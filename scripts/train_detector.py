#!/usr/bin/env python3
"""赛题5 海面目标检测 — YOLOv8-OBB 训练脚本。

跨视角增广：训练集以卫星俯视（方形）为主，测试集以无人机斜视（宽屏/竖屏）为主。
通过随机透视/旋转/翻转/HSV 增强跨视角泛化能力。

依赖（隔离 venv，不影响 shipmind 主依赖）：
    pip install -r contest/requirements-detector.txt

用法：
    python contest/train_detector.py
    python contest/train_detector.py --model yolov8m-obb.pt --epochs 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ultralytics 在脚本顶层 import，便于缺失时报清晰错误
try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "缺少 ultralytics。请先建隔离 venv：\n"
        "  python -m venv D:\\hao\\shipmind\\.venv-detector\n"
        "  D:\\hao\\shipmind\\.venv-detector\\Scripts\\activate\n"
        "  pip install -r contest/requirements-detector.txt"
    )

REPO = Path(__file__).resolve().parents[1]
DATA_YAML = REPO / "contest" / "datasets" / "yolo_obb" / "data.yaml"


def build_aug_config() -> dict:
    """跨视角域泛化增广配置。

    关键参数（针对 nadir→oblique 域差）：
      - perspective: 强透视扰动，模拟斜视几何畸变
      - degrees: 大角度旋转，模拟视角变化
      - fliplr/flipud/transpose: 横竖翻转，覆盖宽屏/竖屏视角
      - hsv: 色相/饱和/明度抖动，适应光照差异
      - mosaic/mixup: 混合增强，提升小目标鲁棒性
    """
    return {
        "perspective": 0.0005,   # 透视畸变（ultralytics 上限 0.001）
        "degrees": 15.0,         # 大角度旋转（默认 0），模拟视角偏转
        "fliplr": 0.5,
        "flipud": 0.3,           # 上下翻转（默认 0），覆盖竖屏视角
        "transpose": 0.3,        # 转置（宽↔高互换），关键跨视角增广
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.0,
        "scale": 0.5,
        "translate": 0.1,
    }


def main():
    ap = argparse.ArgumentParser(description="赛题5 YOLOv8-OBB 训练")
    ap.add_argument("--model", default="yolov8m-obb.pt",
                    help="预训练权重（yolov8n/s/m/l/x-obb.pt）")
    ap.add_argument("--data", default=str(DATA_YAML),
                    help="data.yaml 路径")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="训练输入尺寸（大图用 1024/1536）")
    ap.add_argument("--batch", default=-1,
                    help="batch（-1=自动，或 8/16/32）")
    ap.add_argument("--device", default="0",
                    help="cuda 设备号（0/0,1）或 cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default=str(REPO / "contest" / "runs"))
    ap.add_argument("--name", default="sea_obb",
                    help="本次实验名（runs/<name> 下）")
    ap.add_argument("--patience", type=int, default=20,
                    help="早停耐心（0=不早停）")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"data.yaml 不存在: {args.data}\n"
                         "请先运行: python contest/labelme_to_yolo_obb.py "
                         "--src \"D:\\hao\\赛题5数据集\\train\" "
                         "--out contest/datasets/yolo_obb")

    model = YOLO(args.model)
    aug = build_aug_config()

    print(f"=== 训练配置 ===")
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")
    print(f"跨视角增广: perspective={aug['perspective']} degrees={aug['degrees']} "
          f"flipud={aug['flipud']} transpose={aug['transpose']}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        exist_ok=False,
        # 跨视角域泛化增广
        **aug,
        # 保存与复现
        seed=42,
        deterministic=True,
        verbose=True,
    )
    print(f"\n训练完成。最佳权重: {REPO / 'contest' / 'runs' / args.name / 'weights' / 'best.pt'}")
    print(f"结果目录: {REPO / 'contest' / 'runs' / args.name}")
    return results


if __name__ == "__main__":
    main()
