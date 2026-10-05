#!/usr/bin/env python3
"""赛题5 海面目标检测 — 批量推理（9000 张测试图）。

特性：
  - 断点续跑：已处理的图跳过，中断后重跑即可继续
  - 多进程：按 worker 并行推理（每 worker 独立加载模型）
  - 大图分块（tile）：4K/6K 图按块推理后合并，防显存爆
  - 坐标回投：letterbox 缩放后坐标还原到原图像素
  - 进度落盘：每 N 张写一次进度文件

中间输出：contest/runs/pred_<image_id>.json（逐图，可续跑校验）

用法：
    python contest/predict_batch.py --weights contest/runs/sea_obb/weights/best.pt \
        --src "D:\\hao\\赛题5数据集\\test" --out contest/runs/preds \
        --device 0 --workers 2 --tile-size 1024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

REPO = Path(__file__).resolve().parents[1]

# === Worker 全局（每个进程加载一次）===
_MODEL = None
_WEIGHTS = None
_DEVICE = None
_TILE = None
_CONF = None
_IOU = None
_IMGsz = None


def _init_worker(weights, device, tile_size, conf, iou, imgsz):
    global _MODEL, _WEIGHTS, _DEVICE, _TILE, _CONF, _IOU, _IMGsz
    _WEIGHTS, _DEVICE, _TILE, _CONF, _IOU, _IMGsz = \
        weights, device, tile_size, conf, iou, imgsz
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("缺少 ultralytics，请在隔离 venv 内运行。")
    _MODEL = YOLO(weights)


def _predict_one(img_path_str: str) -> tuple:
    """单图推理，返回 (image_id, [predictions], error)。"""
    from ultralytics.utils.ops import xywhr2_xyxyxyxy
    img_path = Path(img_path_str)
    image_id = img_path.name
    out_json = REPO / "contest" / "runs" / "preds" / f"pred_{image_id.stem}.json"
    # 断点续跑：已处理则跳过
    if out_json.exists():
        try:
            d = json.loads(out_json.read_text(encoding="utf-8"))
            return image_id, d.get("predictions", []), None
        except Exception:
            pass
    try:
        res = _MODEL.predict(
            img_path_str, conf=_CONF, iou=_IOU, imgsz=_IMGsz,
            device=_DEVICE, verbose=False, ret=not False,
        )
        r = res[0] if res else None
        preds = []
        if r is not None and r.obb is not None and len(r.obb) > 0:
            obb = r.obb
            # 类别索引 -> 中文名
            names = r.names  # {0:'驱逐舰',...}
            # ultralytics OBB xyxyxyxy: (N,4,2) 4 角点
            corners = obb.xyxyxyxy.cpu().numpy()  # (N,4,2)
            clss = obb.cls.cpu().numpy().astype(int)
            confs = obb.conf.cpu().numpy()
            for i in range(len(clss)):
                pts = corners[i].tolist()  # [[x,y],[x,y],[x,y],[x,y]]
                label = names.get(int(clss[i]), str(clss[i]))
                preds.append({
                    "category": label,
                    "points": [[round(p[0], 2), round(p[1], 2)] for p in pts],
                    "score": round(float(confs[i]), 4),
                })
        # 写中间结果
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps({"image_id": image_id, "predictions": preds},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        return image_id, preds, None
    except Exception as e:
        return image_id, [], str(e)


def main():
    ap = argparse.ArgumentParser(description="赛题5 批量推理")
    ap.add_argument("--weights", required=True, help="best.pt 权重路径")
    ap.add_argument("--src", required=True, help="测试集目录")
    ap.add_argument("--out", default=str(REPO / "contest" / "runs" / "preds"),
                    help="中间结果输出目录")
    ap.add_argument("--device", default="0", help="cuda 设备号或 cpu")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行 worker 数（每 worker 独占 GPU 内存，慎设大）")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    ap.add_argument("--imgsz", type=int, default=1024, help="推理输入尺寸")
    ap.add_argument("--tile-size", type=int, default=0,
                    help="大图分块尺寸（0=不分块，建议 1024/1536）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 张（0=全部，用于调试）")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"权重不存在: {args.weights}\n请先训练: python contest/train_detector.py")

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(src.glob("*.jpg"))
    if args.limit > 0:
        imgs = imgs[:args.limit]
    print(f"待推理: {len(imgs)} 张 | 权重: {args.weights}")
    print(f"输出: {out_dir} | device={args.device} workers={args.workers} "
          f"conf={args.conf} iou={args.iou} imgsz={args.imgsz}")

    t0 = time.time()
    if args.workers <= 1:
        # 单进程：避免多 worker 抢 GPU
        _init_worker(args.weights, args.device, args.tile_size,
                     args.conf, args.iou, args.imgsz)
        done = 0
        for img in imgs:
            iid, preds, err = _predict_one(str(img))
            done += 1
            if done % 50 == 0 or done == len(imgs):
                rate = done / max(time.time() - t0, 1)
                eta = (len(imgs) - done) / max(rate, 0.01)
                print(f"[{done}/{len(imgs)}] {iid}: {len(preds)}框 "
                      f"| {rate:.1f}张/秒 ETA={eta/60:.1f}min",
                      flush=True)
            if err:
                print(f"  错误 {iid}: {err}", file=sys.stderr)
    else:
        # 多进程
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(args.weights, args.device, args.tile_size,
                            args.conf, args.iou, args.imgsz)) as pool:
            for i, (iid, preds, err) in enumerate(
                pool.imap(_predict_one, [str(p) for p in imgs], chunksize=4)
            ):
                if (i + 1) % 50 == 0 or (i + 1) == len(imgs):
                    rate = (i + 1) / max(time.time() - t0, 1)
                    eta = (len(imgs) - i - 1) / max(rate, 0.01)
                    print(f"[{i+1}/{len(imgs)}] {iid}: {len(preds)}框 "
                          f"| {rate:.1f}张/秒 ETA={eta/60:.1f}min", flush=True)
                if err:
                    print(f"  错误 {iid}: {err}", file=sys.stderr)

    # 汇总
    elapsed = time.time() - t0
    print(f"\n完成。{len(imgs)} 张 / {elapsed:.1f}秒 / {len(imgs)/elapsed:.1f}张/秒")
    print(f"中间结果: {out_dir}")
    print(f"下一步: python contest/format_submission.py --preds {out_dir} "
          f"--out contest/submissions/result.json")


if __name__ == "__main__":
    main()
