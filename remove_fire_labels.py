"""
데이터셋 라벨 파일에서 클래스 1 (fire) 행을 모두 제거
"""
import os
from pathlib import Path

def remove_class_from_labels(dataset_path, class_to_remove=1):
    """라벨 파일에서 특정 클래스 행 제거"""
    
    label_dirs = [
        Path(dataset_path) / "labels" / "train",
        Path(dataset_path) / "labels" / "val",
        Path(dataset_path) / "labels" / "test",
    ]
    
    total_files = 0
    modified_files = 0
    removed_lines = 0
    
    for label_dir in label_dirs:
        if not label_dir.exists():
            print(f"⚠️ 디렉토리 없음: {label_dir}")
            continue
        
        print(f"\n[처리 중] {label_dir}")
        
        for txt_file in label_dir.glob("*.txt"):
            total_files += 1
            
            # 파일 읽기
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            # 클래스 1이 아닌 행만 유지
            new_lines = []
            removed_in_file = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    try:
                        cls = int(parts[0])
                        if cls != class_to_remove:
                            new_lines.append(line)
                        else:
                            removed_in_file += 1
                    except ValueError:
                        new_lines.append(line)
            
            # 변경된 경우만 저장
            if removed_in_file > 0:
                with open(txt_file, 'w') as f:
                    f.writelines(new_lines)
                modified_files += 1
                removed_lines += removed_in_file
    
    print("\n" + "="*60)
    print("처리 완료")
    print("="*60)
    print(f"총 라벨 파일: {total_files}")
    print(f"수정된 파일: {modified_files}")
    print(f"제거된 행(fire 라벨): {removed_lines}")

if __name__ == "__main__":
    dataset_path = "C:/Users/user/Documents/Projects/datasets/smoke"
    
    print("="*60)
    print("클래스 1 (fire) 라벨 제거 스크립트")
    print("="*60)
    print(f"데이터셋 경로: {dataset_path}")
    print(f"제거할 클래스: 1 (fire)")
    
    remove_class_from_labels(dataset_path, class_to_remove=1)
