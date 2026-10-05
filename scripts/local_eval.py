#!/usr/bin/env python3
"""赛题5 本地评测 — 在 val 集上计算 mAP@0.5 / mAP@0.5:0.95（OBB）。

复现审核要求复现结果与排行榜误差<10%，本脚本用于提交前自检。

依赖 ultralytics（隔离 venv）。

用法：
    python contest/local_eval.py --weights contest/runs/sea_obb/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_YAML = REPO / "contest" / "datasets" / "yolo_obb" / "data.yaml"


def main():
    ap = argparse.ArgumentParser(description="本地 val 评测（OBB mAP）")
    ap.add_argument("--weights", required=True, help="best.pt 权重")
    ap.add_argument("--data", default=str(DATA_YAML), help="data.yaml")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="评测时置信度阈值（0.001=全量算 AP）")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("缺少 ultralytics，请在隔离 venv 内运行。")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, imgsz=args.imgsz, device=args.device,
        conf=args.conf, iou=args.iou, split="val",
        plots=True,
    )
    print("\n=== 评测结果 ===")
    # OBB metrics
    box = metrics.box
    print(f"mAP@0.5     = {box.map50:.4f}")
    print(f"mAP@0.5:0.95= {box.map:.4f}")
    print(f"Precision   = {box.mp:.4f}")
    print(f"Recall      = {box.mr:.4f}")
    print("--- 各类别 mAP@0.5 ---")
    names = model.names
    for i, ap in enumerate(box.ap50):
        print(f"  {names.get(i, i)}: {ap:.4f}")
    print(f"\n复现校验基准: mAP@0.5:0.95 = {box.map:.4f}")


if __name__ == "__main__":
    main()
