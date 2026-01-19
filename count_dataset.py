import os

def count_files(directory):
    if not os.path.exists(directory):
        return "Not Found"
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

datasets = {
    "Original Smoke": "datasets/smoke",
    "Pyro-SDIS": "datasets/pyro_sdis"
}

print(f"{'Dataset':<20} | {'Split':<10} | {'Count':<10}")
print("-" * 45)

total_train = 0
total_val = 0

for name, path in datasets.items():
    for split in ["train", "val"]:
        dir_path = os.path.join(path, "images", split)
        count = count_files(dir_path)
        print(f"{name:<20} | {split:<10} | {count:<10}")
        
        if isinstance(count, int):
            if split == "train": total_train += count
            else: total_val += count

print("-" * 45)
print(f"{'TOTAL':<20} | {'train':<10} | {total_train:<10}")
print(f"{'TOTAL':<20} | {'val':<10} | {total_val:<10}")
