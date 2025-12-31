import cv2
import torch
from ultralytics import RTDETR
import time

# --- 1. 기본 설정 (Configuration) ---

# 우리가 훈련시킨 커스텀 모델 가중치
MODEL_PATH = 'best.pt' 

# 0번 웹캠. Jetson에 연결된 카메라 포트(0, 1...) 또는 
# 동영상 파일 경로(예: '/home/jetson/my_test_video.mp4')
SOURCE_VIDEO = 0  

# 신뢰도 임계값 (이 값 이상일 때만 탐지 결과로 인정)
CONF_THRESHOLD = 0.65 # F1-Score가 가장 높았던 0.65 사용 (이전 분석 기반)

# 윈도우 창 이름
WINDOW_NAME = "Smoke-DETR UI (Press 'q' to quit)"

# --- 2. 장치 설정 (Device Setup) ---
DEVICE = 0 if torch.cuda.is_available() else 'cpu'


def main():
    print(f"Starting Smoke-DETR detector...")
    print(f"Using device: {DEVICE}")

    # --- 3. 모델 로드 (Model Loading) ---
    # (매우 중요) 이 코드가 작동하려면 Jetson의 ultralytics 라이브러리가
    # smoke_modules.py와 수정된 tasks.py로 반드시 패치되어 있어야 합니다!
    try:
        model = RTDETR(MODEL_PATH)
        model.to(DEVICE)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"FATAL: 모델 로드 중 심각한 오류 발생: {e}")
        print("--- 확인 사항 ---")
        print("1. 'best.pt'와 'smoke-detr-l.yaml' 파일이 이 스크립트와 같은 폴더에 있나요?")
        print("2. Jetson의 'ultralytics/nn/' 폴더에 'smoke_modules.py'와 'tasks.py'를 덮어썼나요?")
        return

    # --- 4. 비디오 소스 열기 (Video Source) ---
    cap = cv2.VideoCapture(SOURCE_VIDEO)
    if not cap.isOpened():
        print(f"ERROR: 비디오 소스({SOURCE_VIDEO})를 열 수 없습니다.")
        return

    print(f"Video source {SOURCE_VIDEO} opened. Starting UI...")
    
    # UI 창 생성
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # --- 5. 메인 UI 및 추론 루프 (Main Loop) ---
    try:
        while True:
            # 5.1. 카메라/비디오에서 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("Video stream ended. Exiting loop.")
                break # 비디오가 끝나면 루프 종료

            # 5.2. 모델 추론 수행
            # verbose=False로 설정하여 콘솔에 로그가 찍히는 것을 방지
            results = model.predict(
                frame,
                device=DEVICE,
                conf=CONF_THRESHOLD,
                verbose=False 
            )
            
            # results는 리스트 형태이므로, 첫 번째([0]) 결과를 사용
            result = results[0]

            # 5.3. 바운딩 박스 그리기
            # result.plot() 메소드는 바운딩 박스, 클래스, 신뢰도가
            # 모두 그려진 'frame' 이미지를 반환합니다.
            frame_with_boxes = result.plot()

            # (선택 사항) 여기에 커스텀 로직 추가
            if len(result.boxes) > 0:
                # '연기'가 1개 이상 감지되었을 때의 로직
                # 예: 알람 울리기, 경고 텍스트 표시 등
                cv2.putText(
                    frame_with_boxes, 
                    "!! SMOKE DETECTED !!", 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), # BGR (빨간색)
                    2
                )
            
            # 5.4. UI 창에 최종 프레임 표시
            cv2.imshow(WINDOW_NAME, frame_with_boxes)

            # 5.5. 종료 키 ('q') 감지
            # 1ms 대기하며 'q' 키가 눌렸는지 확인
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' key pressed. Exiting...")
                break
    
    except Exception as e:
        print(f"An error occurred during the inference loop: {e}")
    
    finally:
        # --- 6. 리소스 정리 (Cleanup) ---
        cap.release()
        cv2.destroyAllWindows()
        print("Video capture and UI windows closed.")


if __name__ == "__main__":
    main()
