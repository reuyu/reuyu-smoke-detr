from ultralytics import RTDETR

def main():
    # Load the baseline model (Pure RT-DETR-L)
    # Using local 'rtdetr-l.pt' if available, otherwise it will download
    model = RTDETR('rtdetr-l.pt') 
    
    print("Starting Baseline RT-DETR-L Training...")
    
    import os
    yaml_path = os.path.abspath('smoke_dataset.yaml')
    print(f"Using dataset config: {yaml_path}")
    with open(yaml_path, 'r') as f:
        print("--- Config Content ---")
        print(f.read())
        print("----------------------")

    # Auto-resume logic: Check for existing checkpoint
    # This allows training to continue if interrupted (e.g., power loss, manual stop)
    last_ckpt = 'runs/detect/baseline_rtdetr_l_smoke/weights/last.pt'
    if os.path.exists(last_ckpt):
        print(f"Resuming training from {last_ckpt}...")
        model = RTDETR(last_ckpt)
        resume_training = True
    else:
        print("Starting fresh training...")
        resume_training = False

    # Train the model
    # Batch size set to 4 to ensure stability. Adjust as needed.
    results = model.train(
        data=yaml_path, # Verified existence
        epochs=200,      # Updated to 200 epochs as requested
        imgsz=640,
        batch=2,         # Lower batch size for safety
        workers=0,       # Avoid Windows multiprocessing issues
        amp=False,       # Disable AMP to avoid potential crashes
        classes=[0],     # Filter: Train ONLY on Class 0 (Smoke check)
        device=0,        
        project='runs/detect',
        name='baseline_rtdetr_l_smoke',
        pretrained=True,  # Use pre-trained COCO weights
        exist_ok=True,    # Overwrite existing run if needed
        resume=resume_training # Enable resumption if checkpoint exists
    )
    
    print("Training Complete. Results saved to runs/detect/baseline_rtdetr_l_smoke")

if __name__ == '__main__':
    main()
