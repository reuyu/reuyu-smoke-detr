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
    print("Smoke-DETR Deep Backbone Training (Resume Capable)")
    print("Backbone: [3, 4, 6, 3] (ResNet-50 Style)")
    print("Dataset: Original | Epochs: 200")
    print("=" * 60)

    # 저장 경로 및 체크포인트 파일 경로 정의
    project_path = 'runs/smoke_detr_deep'
    run_name = 'resnet50_style_200ep'
    last_ckpt = f"{project_path}/{run_name}/weights/last.pt"

    # 1. 체크포인트 존재 여부 확인 후 모델 로드
    if os.path.exists(last_ckpt):
        print(f"[Info] Resuming training from {last_ckpt}...")
        model = RTDETR(last_ckpt)  # .pt 파일로 모델 로드
        resume_training = True
    else:
        print("[Info] Initializing new model from smoke-detr-deep.yaml...")
        model = RTDETR("smoke-detr-deep.yaml")  # 새 구조 로드
        resume_training = False

    # 2. Training 실행
    # resume=True는 .pt 파일을 로드했을 때만 동작하므로 분기 처리 혹은 인자 전달
    if resume_training:
        model.train(resume=True) # resume=True만 주면 기존 설정 그대로 이어감
    else:
        model.train(
            data='smoke_dataset.yaml',
            epochs=200,
            imgsz=640,
            batch=8,
            lr0=0.0001,
            lrf=0.01,
            workers=0,
            project=project_path,
            name=run_name,
            save=True,
            plots=True,
            exist_ok=True,
            amp=True
        )

if __name__ == "__main__":
    main()