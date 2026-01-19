import sys
import os
from ultralytics import RTDETR

# 커스텀 모듈 등록
sys.path.append(os.getcwd())
try:
    import ultralytics.nn.smoke_modules 
    print("[Success] Custom smoke_modules imported.")
except ImportError as e:
    print(f"[Error] Failed to import smoke_modules: {e}")

def main():
    print("=" * 60)
    print("Smoke-DETR Extended Training Phase 2 (ECPConv A1 Fixed)")
    print("Base: ecpconv_a1_200ep | Adding: 200 Epochs")
    print("=" * 60)

    # 1. 이전 학습 체크포인트 로드
    prev_run = 'runs/smoke_detr_fixed/ecpconv_a1_200ep'
    last_pt = os.path.join(prev_run, 'weights', 'last.pt')
    
    # Phase 2가 이미 진행 중인지 확인
    current_project = 'runs/smoke_detr_fixed'
    current_name = 'ecpconv_a1_phase2_200ep'
    phase2_last = os.path.join(current_project, current_name, 'weights', 'last.pt')
    
    if os.path.exists(phase2_last):
        print(f"[Info] Resuming Phase 2 from {phase2_last}")
        model = RTDETR(phase2_last)
        resume_mode = True
    elif os.path.exists(last_pt):
        print(f"[Info] Starting Phase 2 from {last_pt}")
        model = RTDETR(last_pt)
        resume_mode = False  # 이미 끝난 모델이므로 새 학습으로 시작
    else:
        print(f"[Error] No checkpoint found at {last_pt}")
        return

    # 2. 추가 학습
    model.train(
        data='smoke_dataset.yaml',
        epochs=200,                  # 200 더
        imgsz=640,
        batch=10,
        lr0=0.0001,
        lrf=0.01,
        workers=0,
        project=current_project,
        name=current_name,
        save=True,
        plots=True,
        exist_ok=True,
        amp=True,
        resume=resume_mode
    )

if __name__ == "__main__":
    main()
