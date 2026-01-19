from datasets import load_dataset
import sys

try:
    ds = load_dataset("pyronear/pyro-sdis", trust_remote_code=True)
    item = ds['train'][0]
    annotations = item['annotations']
    print(f"Raw annotations type: {type(annotations)}")
    print(f"Raw annotations content: '{annotations}'")
    
    if isinstance(annotations, str):
        lines = annotations.strip().split('\n')
        for line in lines:
            parts = line.strip().split()
            print(f"Parts: {parts}")
            if len(parts) >= 5:
                try:
                    vals = list(map(float, parts[:5]))
                    print(f"Parsed: {vals}")
                except ValueError as e:
                    print(f"Parse Error: {e}")
    else:
        print("Not a string")

except Exception as e:
    print(f"Global Error: {e}")

