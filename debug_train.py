"""
실제 데이터셋으로 학습하면서 NaN 발생 지점 추적
"""
import torch
import sys
sys.path.insert(0, '.')

from ultralytics import RTDETR

def check_tensor(tensor, name, silent=True):
    """텐서 검사"""
    if tensor is None:
        return False
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"  ❌ NaN/Inf in {name}")
        return True
    return False

# NaN 감지 Hook
nan_detected = {"step": -1, "location": ""}

class NaNDetectorHook:
    def __init__(self, name):
        self.name = name
    
    def __call__(self, module, input, output):
        global nan_detected
        if nan_detected["step"] >= 0:
            return
        
        # 입력 검사
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor) and (torch.isnan(inp).any() or torch.isinf(inp).any()):
                    nan_detected["location"] = f"{self.name} INPUT[{i}]"
                    return
        
        # 출력 검사
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any() or torch.isinf(output).any():
                nan_detected["location"] = f"{self.name} OUTPUT"
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor) and (torch.isnan(out).any() or torch.isinf(out).any()):
                    nan_detected["location"] = f"{self.name} OUTPUT[{i}]"

def main():
    print("="*60)
    print("실제 데이터셋 NaN 디버깅")
    print("="*60)
    
    # 모델 로드
    model = RTDETR("smoke-detr-paper.yaml")
    
    # 모든 레이어에 Hook 등록
    print("\n[Hook 등록 중...]")
    hooks = []
    for name, module in model.model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(NaNDetectorHook(name))
            hooks.append(hook)
    print(f"  {len(hooks)}개 모듈에 Hook 등록 완료")
    
    # 짧은 학습 실행
    print("\n[학습 시작 - NaN 발생 시 즉시 중단]")
    print("="*60)
    
    try:
        results = model.train(
            data="smoke_dataset.yaml",
            epochs=1,              # 1 에포크만
            imgsz=640,
            batch=2,               # 작은 배치
            optimizer="AdamW",
            lr0=0.0001,
            project="runs/debug",
            name="nan_debug",
            exist_ok=True,
            workers=0,
            # classes=[0],  # 제거: 데이터셋이 이미 nc:1로 수정됨
            plots=False,
            verbose=True,
        )
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
    
    # Hook 해제
    for hook in hooks:
        hook.remove()
    
    if nan_detected["step"] >= 0:
        print(f"\n🔍 NaN 발생 위치: {nan_detected['location']}")
    else:
        print("\n✅ NaN 감지되지 않음")

if __name__ == "__main__":
    main()
