"""LabelMe -> YOLO-OBB 转换器（赛题5 海面目标检测）。

零第三方依赖（仅 Python 标准库）。读取赛题5训练集的 LabelMe JSON 标注，
转换为 YOLOv8-OBB 训练格式（归一化 8 坐标），按图像宽高比分层划分 train/val，
生成 data.yaml。图片以硬链接方式接入（不复制、不修改原始数据）。

旋转框(rotation)与轴对齐框(rectangle)统一处理为 4 角点 OBB。

用法：
    python contest/labelme_to_yolo_obb.py --src "D:\\hao\\赛题5数据集\\train" --out contest/datasets/yolo_obb
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

CLASSES = [
    "驱逐舰", "护卫舰", "渔船", "杂货船", "液体运输船",
    "航母", "保障船", "散装货船", "集装箱船",
]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
EXPECTED_SHAPES = 11638
EXPECTED_CLASSES = 9


def parse_labelme(json_path: Path):
    """读 LabelMe JSON -> (w, h, [(cls_idx, [(x,y)*4])])。

    跳过未知类别 shape 和非 4 点 shape，保留其余。
    返回 None 表示 JSON 损坏或缺字段。
    """
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    w, h = d.get("imageWidth"), d.get("imageHeight")
    if not w or not h:
        return None
    boxes = []
    for s in d.get("shapes", []):
        label = s.get("label", "")
        if label not in LABEL_TO_IDX:
            continue
        pts = s.get("points", [])
        if len(pts) != 4:
            continue
        cls_idx = LABEL_TO_IDX[label]
        clamped = [
            (max(0.0, min(float(x), float(w))),
             max(0.0, min(float(y), float(h))))
            for x, y in pts
        ]
        boxes.append((cls_idx, clamped))
    return w, h, boxes


def to_yolo_obb_line(cls_idx: int, pts, w, h) -> str:
    coords = []
    for x, y in pts:
        coords.append(f"{x / w:.6f}")
        coords.append(f"{y / h:.6f}")
    return f"{cls_idx} " + " ".join(coords)


def aspect_bucket(w, h) -> int:
    """按宽高比分层：0=竖屏 1=方形(俯视) 2=横屏(斜视)。"""
    r = w / h
    if r < 0.99:
        return 0
    if r > 1.01:
        return 2
    return 1


def link_image(src: Path, dst: Path):
    """硬链接图片到目标路径；跨卷失败则 fallback 复制。不修改源文件。"""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="LabelMe -> YOLO-OBB 转换器")
    ap.add_argument("--src", required=True, help="训练集目录（含 jpg+json）")
    ap.add_argument("--out", required=True, help="输出 YOLO 数据集目录")
    ap.add_argument("--val-stride", type=int, default=5,
                    help="每 N 个取 1 个做 val（默认 5=20%%）")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    json_files = sorted(src.glob("*.json"))
    print(f"扫描到 {len(json_files)} 个标注文件")

    # 第一遍：解析 + 按宽高比分层
    parsed = []
    class_counts = collections.Counter()
    skip_bad = skip_empty = 0
    for jf in json_files:
        result = parse_labelme(jf)
        if result is None:
            skip_bad += 1
            continue
        w, h, boxes = result
        if not boxes:
            skip_empty += 1
            continue
        parsed.append((jf, w, h, aspect_bucket(w, h), boxes))
        for cls_idx, _ in boxes:
            class_counts[CLASSES[cls_idx]] += 1

    # 按宽高比分层 + stride 划分 train/val（确保各视角层都有 val 代表）
    by_bucket = collections.defaultdict(list)
    for item in parsed:
        by_bucket[item[3]].append(item)

    train_items, val_items = [], []
    for b in sorted(by_bucket):
        items = sorted(by_bucket[b], key=lambda x: x[0].name)
        for i, item in enumerate(items):
            (val_items if i % args.val_stride == 0 else train_items).append(item)

    # 第二遍：写 labels + 链接 images
    n_shapes = 0
    for split, items in [("train", train_items), ("val", val_items)]:
        for jf, w, h, _, boxes in items:
            stem = jf.stem
            img_src = jf.with_suffix(".jpg")
            if img_src.exists():
                link_image(img_src, out / f"images/{split}/{stem}.jpg")
            lines = [to_yolo_obb_line(c, p, w, h) for c, p in boxes]
            (out / f"labels/{split}/{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            n_shapes += len(boxes)

    # data.yaml
    yaml_lines = [
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASSES)}",
        "names:",
    ]
    for i, c in enumerate(CLASSES):
        yaml_lines.append(f"  {i}: {c}")
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # 自检输出
    print(f"train: {len(train_items)} 张 | val: {len(val_items)} 张")
    print(f"总框数: {n_shapes}（预期 {EXPECTED_SHAPES}）")
    print(f"跳过: bad_json={skip_bad}  empty={skip_empty}")
    print("--- 类别分布 ---")
    for c in CLASSES:
        print(f"  {c}: {class_counts[c]}")
    print(f"类别数: {len(class_counts)}（预期 {EXPECTED_CLASSES}）")
    print(f"data.yaml: {out / 'data.yaml'}")

    ok = n_shapes == EXPECTED_SHAPES and len(class_counts) == EXPECTED_CLASSES
    if ok:
        print("自检通过 ✓")
    else:
        print("自检失败 ✗ — 请检查上方输出")
        sys.exit(1)


if __name__ == "__main__":
    main()
