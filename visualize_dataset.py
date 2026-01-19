import os
import random
import cv2
import glob
import matplotlib.pyplot as plt

def visualize_samples(image_dir, label_dir, output_dir="vis_output", num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 목록
    img_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not img_files:
        print("No images found!")
        return

    # 다중 객체가 있는 샘플 찾기 시도
    multi_obj_samples = []
    single_obj_samples = []
    
    # 셔플링해서 랜덤하게 검사
    random.shuffle(img_files)
    
    print("Searching for multi-object samples...")
    for img_path in img_files:
        if len(multi_obj_samples) >= 3 and len(single_obj_samples) >= 3:
            break
            
        basename = os.path.basename(img_path)
        label_name = os.path.splitext(basename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        if len(lines) > 1:
            multi_obj_samples.append(img_path)
        elif len(lines) == 1:
            single_obj_samples.append(img_path)

    # 샘플 합치기
    samples = multi_obj_samples[:3] + single_obj_samples[:2]
    
    print(f"Visualizing {len(samples)} samples (Multi: {len(multi_obj_samples)}, Single: {len(single_obj_samples)})")

    for i, img_path in enumerate(samples):
        # 이미지 로드
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # 라벨 로드
        basename = os.path.basename(img_path)
        label_name = os.path.splitext(basename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            # YOLO format: class x_center y_center w h
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            
            # Convert to pixel coordinates
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Smoke", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 저장
        save_path = os.path.join(output_dir, f"vis_{i}_{basename}")
        cv2.imwrite(save_path, img)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    visualize_samples(
        "datasets/pyro_sdis/images/train",
        "datasets/pyro_sdis/labels/train",
        num_samples=args.num_samples
    )
