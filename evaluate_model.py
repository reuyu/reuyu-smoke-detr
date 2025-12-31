import torch
from ultralytics import RTDETR
import os

# --- 1. 기본 설정 (Configuration) ---

# 훈련된 최종 모델 가중치
MODEL_PATH = 'best.pt' 

# 훈련/검증/테스트 경로가 모두 정의된 YAML 파일
DATA_YAML_PATH = r'C:\Users\user\Documents\Projects\ultralytics\smoke_dataset.yaml'

# 결과 그래프와 로그가 저장될 폴더
OUTPUT_DIR = 'evaluation_results'

# --- 2. 장치 설정 (Device Setup) ---
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

def main():
    print(f"Starting Smoke-DETR Final Evaluation...")
    print(f"Using device: {DEVICE}")

    # --- 3. 모델 로드 ---
    # (중요) 훈련(training)을 실행했던 (라이브러리가 패치된)
    # Python 가상환경에서 실행해야 합니다.
    try:
        model = RTDETR(MODEL_PATH)
        model.to(DEVICE)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"FATAL: 모델 로드 중 오류 발생: {e}")
        print("훈련 시 사용했던 동일한 가상환경(예: smoke_jetson)에서 실행했는지 확인하세요.")
        return

    # --- 4. 최종 평가 실행 ---
    # model.predict()가 아닌 model.val()을 사용합니다.
    # split='test' : YAML 파일의 'test:' 경로를 사용하라고 지정합니다.
    print(f"Running evaluation on 'test' split (from {DATA_YAML_PATH})...")
    try:
        metrics = model.val(
            data=DATA_YAML_PATH,
            split='test',            # 'val'이 아닌 'test' 스플릿으로 평가
            device=DEVICE,
            project=OUTPUT_DIR,      # 결과 저장 기본 폴더
            name='final_evaluation', # 하위 폴더 이름
            exist_ok=True,           # 덮어쓰기 허용
            plots=True,             # (True가 기본값) 모든 그래프 생성
            workers=0
        )
        
        print("Evaluation complete.")
        print("="*30)
        print("--- 최종 성능 지표 (Test Set) ---")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50:     {metrics.box.map50:.4f}")
        print(f"Precision:  {metrics.box.p[0]:.4f}") # (첫 번째 클래스 기준)
        print(f"Recall:     {metrics.box.r[0]:.4f}") # (첫 번째 클래스 기준)
        print("="*30)
        print(f"모든 결과 그래프와 로그는 {os.path.join(OUTPUT_DIR, 'final_evaluation')} 폴더에 저장되었습니다.")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()
