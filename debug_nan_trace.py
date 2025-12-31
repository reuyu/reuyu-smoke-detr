"""
강력한 NaN 디버깅 스크립트
 - torch.autograd.set_detect_anomaly(True) 활성화
 - 모든 레이어 Forward/Backward Hook으로 NaN 감시
 - 발견 즉시 종료 및 위치 보고
"""
import torch
import torch.nn as nn
import sys
import os

# 현재 경로 추가
sys.path.insert(0, os.getcwd())

from ultralytics import RTDETR

# NaN 발생 시 로그를 남길 전역 변수
nan_found = False

def check_tensor(tensor, name, step_info=""):
    """텐서 검사 및 NaN 발견 시 리포트"""
    global nan_found
    if nan_found: return True
    
    if tensor is None:
        return False
        
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"\n[🚨 NaN/Inf DETECTED] {step_info} -> {name}")
            print(f"    Min: {tensor.min().item()}, Max: {tensor.max().item()}")
            print(f"    Shape: {tensor.shape}")
            nan_found = True
            return True
    elif isinstance(tensor, (tuple, list)):
        for i, t in enumerate(tensor):
            if check_tensor(t, f"{name}[{i}]", step_info):
                return True
    return False

def register_hooks(model):
    """모든 모듈에 Forward/Backward Hook 등록"""
    print("[Hook] 모듈에 감시 장치 등록 중...")
    
    def forward_hook(module, input, output):
        if nan_found: raise SystemExit("NaN 발생으로 인한 강제 종료")
        name = module._get_name()
        # 입력 검사
        check_tensor(input, "INPUT", f"Forward: {name}")
        # 출력 검사
        check_tensor(output, "OUTPUT", f"Forward: {name}")

    def backward_hook(module, grad_input, grad_output):
        if nan_found: raise SystemExit("NaN 발생으로 인한 강제 종료")
        name = module._get_name()
        # Gradient 검사
        check_tensor(grad_input, "GRAD_INPUT", f"Backward: {name}")
        check_tensor(grad_output, "GRAD_OUTPUT", f"Backward: {name}")

    count = 0
    for name, module in model.named_modules():
        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)
        count += 1
    
    print(f"[Hook] 총 {count}개 모듈 감시 중")

def main():
    print("="*60)
    print("🔥 강력한 NaN 추적기 시작")
    print("="*60)
    
    # 1. Anomaly Detection 비활성화 (Inplace 연산 허용)
    torch.autograd.set_detect_anomaly(False)
    print("✅ torch.autograd.set_detect_anomaly(False) (Hook만 사용)")
    
    # 2. 모델 로드
    try:
        model = RTDETR("smoke-detr-paper.yaml")
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 3. Hook 등록 (Model 내부 모듈)
    register_hooks(model.model)
    
    # 4. 학습 시작
    print("\n[🚀 학습 시작 (1 Epoch)]")
    try:
        model.train(
            data="smoke_dataset.yaml",
            epochs=1,
            imgsz=640,
            batch=4,  # 배치 크기
            optimizer="AdamW",
            lr0=0.0001, # 일부러 0.0001로 테스트 (NaN 유발 확인용)
            workers=0,
            plots=False,
            val=False,
            device=0, # GPU 사용
        )
    except SystemExit:
        print("\n🛑 NaN 감지로 인해 학습이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생 (Traceback 확인 필요):")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
