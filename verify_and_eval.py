import sys
import os
import torch

# 1. 커스텀 모듈 경로 등록 (현재 프로젝트 경로)
sys.path.append(os.getcwd())

# 2. 커스텀 모듈 Import (필수! 모델 로드 전에 해야 함)
# pickle이 이 모듈을 찾을 수 있게 해줌
try:
    import ultralytics.nn.smoke_modules
    print("[Success] Custom smoke_modules imported.")
except ImportError as e:
    print(f"[Error] Failed to import smoke_modules: {e}")
    # 프로젝트 내의 ultralytics를 바라보고 있다면 경로 조정이 필요할 수 있음
    # 하지만 site-packages에 덮어썼다면 괜찮음.

from ultralytics import RTDETR

def run_evaluation():
    # 3. 모델 경로 (가장 최근 학습 결과)
    # 직접 경로를 지정하거나, glob으로 자동 탐색
    model_path = r"runs\smoke_detr\train_20251231_231425\weights\best.pt"  # 사용자 로그 기준 경로
    
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at: {model_path}")
        return

    print(f"[Info] Loading model from {model_path}...")
    try:
        model = RTDETR(model_path)
    except ModuleNotFoundError as e:
        print(f"[Critical] Still failing to find module: {e}")
        print("Tip: ultralytics 패키지 내부에 smoke_modules.py가 물리적으로 존재해야 합니다.")
        return

    # 4. 평가 실행 (workers=0 필수)
    print("[Info] Starting validation...")
    results = model.val(
        data="smoke_dataset.yaml",
        split="val",
        workers=0,  # Windows Fix
        verbose=True
    )
    
    print("\n" + "="*50)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()
