from ultralytics import RTDETR

def main():
    last_ckpt = r"C:\Users\user\Documents\Projects\smoke_jetson_project\teacher_l_model_final8\weights\last.pt"
    data_yaml_path = r"C:\Users\user\Documents\Projects\ultralytics\smoke_dataset.yaml"

    print("Smoke-DETR (Teacher) 모델을 **기존 세션에서 재개(resume)** 합니다...")

    model = RTDETR(last_ckpt)  # ✅ 마지막 체크포인트에서 모델 불러오기

    results = model.train(
        data=data_yaml_path,
        resume=True,  # True 여도 이제 경로가 명확히 지정됨
        workers=0,
        project=r'C:\Users\user\Documents\Projects\smoke_jetson_project',
        name='teacher_l_model_final8',
    )

    print("Teacher 모델 학습 완료!")
    print(f"최종 모델은 {model.trainer.best} 에 저장되었습니다.")

if __name__ == '__main__':
    main()
