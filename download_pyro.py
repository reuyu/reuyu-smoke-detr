import os
import shutil
from datasets import load_dataset
from tqdm import tqdm

# Define paths matching our project structure
REPO_ID = "pyronear/pyro-sdis"
BASE_DIR = "datasets/pyro_sdis"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

def save_ultralytics_format(dataset_split, split_name):
    """
    Save a dataset split into the Ultralytics format with Class Re-mapping (All -> 0).
    Args:
        dataset_split: The dataset split (e.g., dataset["train"])
        split_name: "train" or "val"
    """
    # Create directories
    os.makedirs(os.path.join(IMAGE_DIR, split_name), exist_ok=True)
    os.makedirs(os.path.join(LABEL_DIR, split_name), exist_ok=True)

    print(f"Processing {split_name} split...")
    for example in tqdm(dataset_split):
        # 1. Save Image
        image = example["image"]  # PIL.Image.Image
        image_name = example["image_name"]
        
        # Ensure compatible filename
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_name += ".jpg"
            
        output_image_path = os.path.join(IMAGE_DIR, split_name, image_name)
        # Check if exists to skip (optional, but good for resume) - Overwrite for safety now
        image.save(output_image_path)

        # 2. Process & Save Label
        annotations = example["annotations"] # String in YOLO format (e.g., "1 0.5 0.5 ...")
        
        # We need to force Class ID to 0 (Smoke) because our model is Single-Class (nc=1).
        # Original might have 0=Fire, 1=Smoke or similar. We treat ALL as Smoke/Target.
        
        new_lines = []
        if annotations:
            for line in annotations.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Replace class_id (parts[0]) with '0'
                    parts[0] = '0'
                    new_lines.append(" ".join(parts))
        
        label_name = os.path.splitext(image_name)[0] + ".txt"
        output_label_path = os.path.join(LABEL_DIR, split_name, label_name)
        
        if new_lines:
            with open(output_label_path, "w") as label_file:
                label_file.write("\n".join(new_lines))
        # If no annotations, we can skip creating the file or create empty (bg image)
        # Yolo handles empty files or missing files as background. 
        # But commonly empty .txt is better for explicit background.
        else:
            with open(output_label_path, "w") as label_file:
                pass # Empty file

def main():
    # Clean up previous failed attempts if needed?
    # shutil.rmtree(BASE_DIR, ignore_errors=True) # Optional
    
    print("Loading dataset from Hugging Face...")
    try:
        dataset = load_dataset(REPO_ID, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Process Train (Original 'train') -> 'train'
    if 'train' in dataset:
        save_ultralytics_format(dataset['train'], "train")
    
    # Process Val (Original 'val' or split from train constraint?)
    # The reference code assumes 'val' key exists.
    if 'val' in dataset:
        save_ultralytics_format(dataset['val'], "val")
    elif 'test' in dataset:
        print("Using 'test' split as 'val'...")
        save_ultralytics_format(dataset['test'], "val")
    else:
        print("No validation split found in dataset keys:", dataset.keys())

    print("\nDataset export complete. Structure:")
    print(f"{BASE_DIR}/")
    print(f"  images/train, val")
    print(f"  labels/train, val")

if __name__ == "__main__":
    main()
