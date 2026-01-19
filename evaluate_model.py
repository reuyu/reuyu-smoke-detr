import sys
import os
import glob
from ultralytics import RTDETR

# 커스텀 모듈 등록
sys.path.append(os.getcwd())
try:
    import ultralytics.nn.smoke_modules 
    print("[Success] Custom smoke_modules imported.")
except ImportError as e:
    print(f"[Error] Failed to import smoke_modules: {e}")

def find_best_model():
    """가장 최근 학습된 best.pt 찾기"""
    # Phase 2 먼저 확인
    phase2_best = 'runs/smoke_detr_fixed/ecpconv_a1_phase2_200ep/weights/best.pt'
    if os.path.exists(phase2_best):
        return phase2_best
    
    # Phase 1 확인
    phase1_best = 'runs/smoke_detr_fixed/ecpconv_a1_200ep/weights/best.pt'
    if os.path.exists(phase1_best):
        return phase1_best
    
    # 다른 폴더에서 찾기
    all_bests = glob.glob('runs/**/best.pt', recursive=True)
    if all_bests:
        all_bests.sort(key=os.path.getmtime, reverse=True)
        return all_bests[0]
    
    return None

def main():
    print("=" * 60)
    print("Smoke-DETR Model Evaluation on Test Dataset")
    print("=" * 60)

    # 1. best.pt 찾기
    best_pt = find_best_model()
    if not best_pt:
        print("[Error] No best.pt found!")
        return
    
    print(f"[Info] Loading model from: {best_pt}")
    model = RTDETR(best_pt)

    # 2. Test 데이터셋으로 평가
    print("\n[Info] Running evaluation on test dataset...")
    results = model.val(
        data='smoke_dataset.yaml',
        split='test',              # test 데이터셋 사용
        imgsz=640,
        batch=10,
        workers=0,                 # Windows Fix
        save_json=True,            # COCO format JSON 저장
        plots=True,                # Confusion Matrix, PR curve 등 저장
        project='runs/evaluation',
        name='test_results',
        exist_ok=True
    )

    # 3. 결과 출력
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@0.5:      {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Precision:    {results.box.mp:.4f}")
    print(f"Recall:       {results.box.mr:.4f}")
    print("=" * 60)
    
    # 4. 결과 저장 경로 안내
    print(f"\n[Info] Results saved to: runs/evaluation/test_results/")
    print("  - confusion_matrix.png")
    print("  - PR_curve.png")
    print("  - F1_curve.png")
    print("  - predictions.json (COCO format)")

if __name__ == "__main__":
    main()
