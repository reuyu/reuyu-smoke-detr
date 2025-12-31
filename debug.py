import torch
from ultralytics import RTDETR
import sys
import ultralytics
import os

print(f"✅ 현재 사용 중인 Ultralytics 경로: {os.path.dirname(ultralytics.__file__)}")

def debug_model():
    # 1. 모델 설정 파일 경로 (사용자가 만든 yaml 파일)
    yaml_path = r"C:/Users/user/Documents/Projects/smoke-detr-paper.yaml"
    
    print(f"🔍 [1/3] Loading Model Configuration: {yaml_path}")
    try:
        # 모델 생성 (이 과정에서 tasks.py와 smoke_modules.py가 잘 연결되었는지 확인됨)
        model = RTDETR(yaml_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Tip: tasks.py에 모듈 등록이 잘 되었는지, smoke_modules.py에 오타가 없는지 확인하세요.")
        return

    # 2. 모델 구조 출력 (우리가 만든 모듈 이름이 보여야 함)
    print("\n🔍 [2/3] Checking Model Architecture...")
    found_ecp = False
    found_mffpn = False
    
    # 모델의 모든 레이어를 순회하며 이름 확인
    for name, module in model.model.named_modules():
        class_name = module.__class__.__name__
        if "ECPConvBlock" in class_name:
            found_ecp = True
        if "SmokeMFFPN" in class_name:
            found_mffpn = True
            
    if found_ecp:
        print("✅ 'ECPConvBlock' found in backbone! (백본 교체 성공)")
    else:
        print("❌ 'ECPConvBlock' NOT found. (백본 설정 확인 필요)")

    if found_mffpn:
        print("✅ 'SmokeMFFPN' found in neck! (넥/인코더 교체 성공)")
    else:
        print("❌ 'SmokeMFFPN' NOT found. (헤드/넥 설정 확인 필요)")
        
    # 상세 구조 출력 (필요시 주석 해제하여 확인)
    # model.info(detailed=True) 

    # 3. 가짜 데이터로 Forward Pass 테스트 (형상 맞는지 확인)
    print("\n🔍 [3/3] Testing Forward Pass (Dry Run)...")
    try:
        # [Batch=1, Channel=3, Height=640, Width=640] 의 가짜 이미지 생성
        dummy_input = torch.randn(1, 3, 640, 640).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 예측 실행 (학습 모드가 아닌 추론 모드로 테스트)
        # verbose=False로 로그 줄임
        results = model.predict(source=dummy_input, verbose=False)
        
        print("✅ Forward pass successful! (연산 흐름 정상)")
        print("🎉 Everything looks good. You are ready to train!")
        
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        print("💡 Tip: 채널 수(ch)나 차원(stride) 계산이 맞지 않을 수 있습니다. smoke_modules.py의 forward 부분을 확인하세요.")

if __name__ == "__main__":
    debug_model()