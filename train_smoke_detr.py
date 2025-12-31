"""
Smoke-DETR Training & Evaluation Script
논문: "Smoke Detection Transformer: An Improved Real-Time Detection Transformer 
       Smoke Detection Model for Early Fire Warning"

학습 설정 (논문 4.1절):
- Input: 640x640
- Epochs: 200
- Optimizer: AdamW
- LR: 0.0001
- Momentum: 0.9
- Weight Decay: 0.0001
"""

import os
import sys
import json
import torch
import datetime
from pathlib import Path
from ultralytics import RTDETR

def train_smoke_detr():
    """Smoke-DETR 모델 학습 (200 에포크)"""
    
    # =========================================
    # 1. 모델 및 설정
    # =========================================
    # model_yaml = "smoke-detr-paper.yaml"  # Smoke-DETR 아키텍처
    
    # =========================================
    # 2. 모델 초기화
    # =========================================
    model_yaml = "smoke-detr-paper.yaml"
    model = RTDETR(model_yaml)
    
    # Gradient Clipping 콜백 추가
    def on_train_batch_end(trainer):
        if trainer.model:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
            
    model.add_callback("on_train_batch_end", on_train_batch_end)
    
    # 2. 데이터셋 설정
    data_yaml = "smoke_dataset.yaml"      # 연기 데이터셋
    
    # 결과 저장 디렉토리
    project_name = "runs/smoke_detr"
    run_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 60)
    print("Smoke-DETR Training Script")
    print("=" * 60)
    print(f"Model Config: {model_yaml}")
    print(f"Dataset: {data_yaml}")
    print(f"Save Dir: {project_name}/{run_name}")
    print("=" * 60)
    
    # =========================================
    # 2. 모델 초기화 (이미 위에서 수행됨)
    # =========================================
    # model = RTDETR(model_yaml) # 중복 제거
    
    # =========================================
    # 3. 학습 (논문 하이퍼파라미터)
    # =========================================
    results = model.train(
        data=data_yaml,
        epochs=200,              # 논문: 200 epochs
        imgsz=640,               # 논문: 640x640
        batch=8,                 # 논문: batch_size=4, 속도 위해 8로 변경
        optimizer="AdamW",       # 논문: AdamW
        lr0=0.0001,             # 학습 안정화를 위해 10배 감소 (0.0001 -> 0.00001)
        lrf=1.0,                 # 논문: Final LR = 1.0 (multiplier)
        momentum=0.9,            # 논문: Momentum = 0.9
        weight_decay=0.0001,     # 논문: Weight Decay = 0.0001
        project=project_name,
        name=run_name,
        exist_ok=True,
        save=True,               # 모델 저장
        save_period=50,          # 50 에포크마다 체크포인트 저장
        plots=True,              # 학습 그래프 생성
        val=True,                # 검증 수행
        patience=0,              # Early stopping 비활성화 (200 에포크 완전 학습)
        verbose=True,
        workers=0,           # Windows multiprocessing 호환성
    )
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    
    return model, results, f"{project_name}/{run_name}"


def evaluate_and_save_results(model, save_dir):
    """검증 데이터셋으로 평가 및 결과 저장"""
    
    print("\n" + "=" * 60)
    print("Starting Evaluation on Validation Dataset")
    print("=" * 60)
    
    # =========================================
    # 1. 검증 데이터셋 평가
    # =========================================
    val_results = model.val(
        data="smoke_dataset.yaml",
        split="val",
        imgsz=640,
        batch=4,
        save_json=True,    # COCO format JSON 저장
        plots=True,        # PR curve, confusion matrix 등 저장
        verbose=True,
    )
    
    # =========================================
    # 2. 결과 추출
    # =========================================
    results_dict = {
        "model": "Smoke-DETR",
        "evaluation_date": datetime.datetime.now().isoformat(),
        "dataset": "smoke_dataset",
        "metrics": {
            "precision": float(val_results.box.mp),        # Mean Precision
            "recall": float(val_results.box.mr),           # Mean Recall
            "mAP50": float(val_results.box.map50),         # mAP@0.5
            "mAP50-95": float(val_results.box.map),        # mAP@0.5:0.95
        },
        "per_class": {}
    }
    
    # 클래스별 결과
    class_names = val_results.names
    for i, name in class_names.items():
        results_dict["per_class"][name] = {
            "precision": float(val_results.box.p[i]) if i < len(val_results.box.p) else 0,
            "recall": float(val_results.box.r[i]) if i < len(val_results.box.r) else 0,
            "ap50": float(val_results.box.ap50[i]) if i < len(val_results.box.ap50) else 0,
            "ap": float(val_results.box.ap[i]) if i < len(val_results.box.ap) else 0,
        }
    
    # =========================================
    # 3. 결과 저장
    # =========================================
    save_path = Path(save_dir)
    
    # JSON 저장
    json_path = save_path / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    # 텍스트 리포트 저장
    report_path = save_path / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Smoke-DETR Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Evaluation Date: {results_dict['evaluation_date']}\n")
        f.write(f"Dataset: {results_dict['dataset']}\n\n")
        f.write("-" * 40 + "\n")
        f.write("Overall Metrics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision:    {results_dict['metrics']['precision']:.4f}\n")
        f.write(f"Recall:       {results_dict['metrics']['recall']:.4f}\n")
        f.write(f"mAP@0.5:      {results_dict['metrics']['mAP50']:.4f}\n")
        f.write(f"mAP@0.5:0.95: {results_dict['metrics']['mAP50-95']:.4f}\n\n")
        f.write("-" * 40 + "\n")
        f.write("Per-Class Metrics\n")
        f.write("-" * 40 + "\n")
        for cls_name, cls_metrics in results_dict["per_class"].items():
            f.write(f"\n{cls_name}:\n")
            f.write(f"  Precision: {cls_metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {cls_metrics['recall']:.4f}\n")
            f.write(f"  AP@0.5:    {cls_metrics['ap50']:.4f}\n")
            f.write(f"  AP:        {cls_metrics['ap']:.4f}\n")
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Precision:    {results_dict['metrics']['precision']:.4f}")
    print(f"Recall:       {results_dict['metrics']['recall']:.4f}")
    print(f"mAP@0.5:      {results_dict['metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {results_dict['metrics']['mAP50-95']:.4f}")
    print("-" * 40)
    print(f"Results saved to: {save_path}")
    print(f"  - {json_path.name}")
    print(f"  - {report_path.name}")
    print("=" * 60)
    
    return results_dict


def main():
    """메인 실행 함수"""
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " Smoke-DETR Training & Evaluation Pipeline ".center(58) + "║")
    print("║" + " Paper: Smoke Detection Transformer ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # 1. 학습
    model, train_results, save_dir = train_smoke_detr()
    
    # 2. 평가 및 결과 저장
    eval_results = evaluate_and_save_results(model, save_dir)
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " All Done! ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\nAll results saved to: {save_dir}")
    
    return eval_results


if __name__ == "__main__":
    main()
