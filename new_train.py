
from ultralytics import RTDETR
import os
import glob
import matplotlib.pyplot as plt

def main():

    model_yaml = r"C:/Users/user/Documents/Projects/smoke-detr-paper.yaml"
    data_yaml = r"C:\Users\user\Documents\Projects\smoke_dataset.yaml" 
    project_dir = r"C:\Users\user\Documents\Projects\runs\detect"
    run_name = "smoke_detr_teacher"
    
    last_ckpt = os.path.join(project_dir, run_name, "weights", "last.pt")
    resume_training = False
    
    if os.path.exists(last_ckpt):
        print(f"Resuming Smoke-DETR training from {last_ckpt}...")
        try:
            model = RTDETR(last_ckpt)
            resume_training = True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting fresh.")
            model = RTDETR(model_yaml)
    else:
        print("Starting fresh Smoke-DETR training...")
        model = RTDETR(model_yaml)

    # Train
    # Note: Using batch=2, workers=0 for stability parallel with Baseline
    results = model.train(
        data=data_yaml,
        epochs=200,
        imgsz=640,
        batch=4,
        workers=0,
        device=0,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        resume=resume_training,
        classes=[0], # Filter: Train ONLY on Class 0 (Smoke)
        amp=False    # Disable AMP for stability
    )

    print("Smoke-DETR Training Complete.")
    print(f"Results saved to {os.path.join(project_dir, run_name)}")

    # Validation & Analysis (Basic)
    # Ultralytics automatically saves results.csv and confusion_matrix.png
    # We can explicitly validate if needed, but .train() includes val at end.
    
    # Generate/Move graphs is handled by Ultralytics automatically.

if __name__ == '__main__':
    main()
