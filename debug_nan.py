"""
NaN 디버깅 스크립트 - ECPConv 및 모델 수치 안정성 테스트
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from ultralytics.nn.smoke_modules import ECPConv, ECPConvBlock, EMA, RCM, SmokeMFFPN

def check_nan_inf(tensor, name):
    """텐서에서 NaN/Inf 검사"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
        print(f"     min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")
        return True
    else:
        print(f"  ✅ {name}: OK (min={tensor.min().item():.4f}, max={tensor.max().item():.4f})")
        return False

def test_ecpconv_stability():
    """ECPConv 수치 안정성 테스트"""
    print("\n" + "="*60)
    print("1. ECPConv 단독 테스트")
    print("="*60)
    
    # 다양한 입력 테스트
    test_cases = [
        ("정상 입력", torch.randn(2, 64, 32, 32)),
        ("큰 값", torch.randn(2, 64, 32, 32) * 100),
        ("작은 값", torch.randn(2, 64, 32, 32) * 0.001),
        ("0 포함", torch.zeros(2, 64, 32, 32)),
        ("일부 극값", torch.randn(2, 64, 32, 32)),
    ]
    test_cases[4][1][0, 0, 0, 0] = 1e10  # 극값 삽입
    
    model = ECPConv(64, 128, k=3, s=1)
    model.eval()
    
    for name, x in test_cases:
        print(f"\n[{name}]")
        check_nan_inf(x, "입력")
        with torch.no_grad():
            try:
                out = model(x)
                check_nan_inf(out, "출력")
            except Exception as e:
                print(f"  ❌ 에러 발생: {e}")

def test_ecpconv_gradient():
    """ECPConv 그래디언트 테스트"""
    print("\n" + "="*60)
    print("2. ECPConv 그래디언트 테스트")
    print("="*60)
    
    model = ECPConv(64, 128, k=3, s=1)
    model.train()
    
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    print("\n[그래디언트 검사]")
    check_nan_inf(x.grad, "입력 grad")
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if check_nan_inf(param.grad, f"param.{name}"):
                print(f"     ⚠️ 이 파라미터에서 NaN/Inf 발생!")

def test_scatter_operation():
    """Scatter 연산 테스트 (ECPConv의 핵심 연산)"""
    print("\n" + "="*60)
    print("3. Scatter 연산 테스트 (ECPConv 핵심)")
    print("="*60)
    
    # ECPConv에서 사용하는 scatter 패턴 시뮬레이션
    B, C, H, W = 2, 64, 32, 32
    k_channels = 16
    
    x = torch.randn(B, C, H, W)
    w = torch.rand(B, C)
    
    _, topk_idx = torch.topk(w, k_channels, dim=1)
    batch_idx = torch.arange(B).unsqueeze(1).expand(-1, k_channels)
    
    x_selected = x[batch_idx, topk_idx]
    print(f"선택된 채널 shape: {x_selected.shape}")
    
    # 처리 시뮬레이션
    y_selected = x_selected * 2  # 간단한 연산
    
    # Scatter 복원
    out = x.clone()
    topk_idx_expanded = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
    out.scatter_(1, topk_idx_expanded, y_selected)
    
    check_nan_inf(out, "Scatter 결과")
    
    # 극단적 경우 테스트
    print("\n[극단적 케이스]")
    y_selected_nan = y_selected.clone()
    y_selected_nan[0, 0, 0, 0] = float('nan')
    out2 = x.clone()
    out2.scatter_(1, topk_idx_expanded, y_selected_nan)
    check_nan_inf(out2, "NaN 포함 Scatter")

def test_full_forward():
    """전체 모델 Forward 테스트"""
    print("\n" + "="*60)
    print("4. 전체 모델 Forward 테스트 (실제 학습 시뮬레이션)")
    print("="*60)
    
    from ultralytics import RTDETR
    
    model = RTDETR("smoke-detr-paper.yaml")
    model.model.train()
    
    # 실제 학습과 유사한 입력
    x = torch.rand(2, 3, 640, 640)  # 0~1 범위
    
    print("\n[Forward Pass with Gradient]")
    try:
        # Forward
        out = model.model(x)
        print(f"출력 타입: {type(out)}")
        
        # Loss 시뮬레이션 (간단히 합계)
        if isinstance(out, (list, tuple)):
            loss = sum(o.sum() if isinstance(o, torch.Tensor) else 0 for o in out)
        else:
            loss = out.sum()
        
        print(f"Loss 값: {loss.item()}")
        check_nan_inf(torch.tensor([loss.item()]), "Loss")
        
        # Backward
        loss.backward()
        print("Backward 완료")
        
        # 그래디언트 검사
        nan_count = 0
        for name, param in model.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"  ❌ NaN/Inf in gradient: {name}")
                    nan_count += 1
        
        if nan_count == 0:
            print("  ✅ 모든 그래디언트 정상")
        else:
            print(f"  ⚠️ {nan_count}개 파라미터에서 NaN/Inf 발견")
            
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("NaN 디버깅 진단 시작")
    print("="*60)
    
    test_ecpconv_stability()
    test_ecpconv_gradient()
    test_scatter_operation()
    test_full_forward()
    
    print("\n" + "="*60)
    print("진단 완료")
    print("="*60)
