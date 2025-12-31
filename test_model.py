import torch
from ultralytics import RTDETR
import os

# --- 1. 기본 설정 (Configuration) ---

# 우리가 훈련시킨 커스텀 모델 가중치 (172 에포크)
MODEL_PATH = 'best.pt' 

# (필수) 테스트할 이미지가 들어있는 폴더 경로
# 이 스크립트와 같은 위치에 'test_images' 폴더를 만드세요.
INPUT_IMAGE_DIR = 'test_images' 

# (필수) 결과 이미지가 저장될 폴더 경로
OUTPUT_DIR = 'test_results'

# 신뢰도 임계값 (F1-Score가 가장 높았던 0.65 근처 값 권장)
CONF_THRESHOLD = 0.65 

# --- 2. 장치 설정 (Device Setup) ---
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

def main():
    print(f"Starting Smoke-DETR performance test on Local PC...")
    print(f"Using device: {DEVICE}")

    # --- 3. 모델 로드 (Model Loading) ---
    # (중요) 이 스크립트는 훈련(training)을 실행했던
    # (라이브러리가 패치된) Python 환경에서 실행해야 합니다.
    try:
        model = RTDETR(MODEL_PATH)
        model.to(DEVICE)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"FATAL: 모델 로드 중 오류 발생: {e}")
        print("--- 확인 사항 ---")
        print("이 스크립트를 훈련(train_teacher.py)에 사용했던")
        print("동일한 가상환경(예: smoke_jetson)에서 실행했는지 확인하세요.")
        return

    # --- 4. 입력/출력 폴더 확인 ---
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f"ERROR: 입력 폴더를 찾을 수 없습니다: {INPUT_IMAGE_DIR}")
        print("스크립트와 같은 위치에 'test_images' 폴더를 만들고 테스트 이미지를 넣어주세요.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Input directory: {INPUT_IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # --- 5. 추론 및 결과 저장 ---
    # model.predict()는 폴더 경로를 받아 배치 처리를 수행합니다.
    try:
        print("Starting batch inference...")
        results = model.predict(
            source=INPUT_IMAGE_DIR,
            device=DEVICE,
            conf=CONF_THRESHOLD,
            save=True,               # 결과를 이미지 파일로 저장
            project=OUTPUT_DIR,      # 저장할 기본 폴더
            name='final_test_run',   # 하위 폴더 이름
            exist_ok=True,           # 덮어쓰기 허용
            verbose=False            # 콘솔 로그 최소화
        )
        
        print(f"Inference complete. Processed {len(results)} images.")
        print(f"Result images saved in: {os.path.join(OUTPUT_DIR, 'final_test_run')}")

    except Exception as e:
        print(f"An error occurred during batch inference: {e}")

if __name__ == "__main__":
    main()
