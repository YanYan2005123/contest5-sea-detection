#!/usr/bin/env python3
"""赛题5 提交格式化器 — 中间推理结果 → 比赛 JSON。

输出严格符合提交规范：
  - 单文件 result_[队伍名].json，UTF-8
  - 顶层为 list，每元素 {image_id, predictions:[{category, points, score}]}
  - image_id 含 .jpg 扩展名，与测试集文件名严格一致
  - category 为中文标签（9 类）
  - points 为 4 顶点 [[x,y]*4]，浮点数保留 2 位
  - score ∈ [0,1]
  - 无目标的图也输出空 predictions（保证完整性）

用法：
    python contest/format_submission.py \
        --preds contest/runs/preds \
        --out contest/submissions/result_Devx.json \
        --team Devx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["驱逐舰", "护卫舰", "渔船", "杂货船", "液体运输船",
           "航母", "保障船", "散装货船", "集装箱船"]
CLASSES_SET = set(CLASSES)


def main():
    ap = argparse.ArgumentParser(description="提交 JSON 格式化")
    ap.add_argument("--preds", required=True,
                    help="predict_batch.py 中间结果目录")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    ap.add_argument("--team", default="Devx", help="队伍名（文件名用）")
    ap.add_argument("--src", default=None,
                    help="测试集目录（提供则按其文件名顺序输出，含空图）")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="低于此分数的框丢弃（0=全保留）")
    args = ap.parse_args()

    preds_dir = Path(args.preds)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 确定 image_id 列表
    if args.src:
        src = Path(args.src)
        image_ids = sorted(p.name for p in src.glob("*.jpg"))
    else:
        image_ids = sorted(p.stem + ".jpg" for p in
                           preds_dir.glob("pred_*.json"))

    print(f"格式化 {len(image_ids)} 张 → {out_path}")
    results = []
    n_boxes = n_filtered = n_bad_label = 0
    for iid in image_ids:
        stem = iid.rsplit(".", 1)[0]
        pred_file = preds_dir / f"pred_{stem}.json"
        preds = []
        if pred_file.exists():
            try:
                d = json.loads(pred_file.read_text(encoding="utf-8"))
                raw = d.get("predictions", [])
            except Exception:
                raw = []
            for p in raw:
                cat = p.get("category", "")
                if cat not in CLASSES_SET:
                    n_bad_label += 1
                    continue
                sc = float(p.get("score", 0.0))
                if sc < args.min_score:
                    n_filtered += 1
                    continue
                pts = p.get("points", [])
                if len(pts) != 4:
                    continue
                # 规范化：浮点 + 2 位小数
                pts_clean = [[round(float(x), 2), round(float(y), 2)]
                             for x, y in pts]
                preds.append({
                    "category": cat,
                    "points": pts_clean,
                    "score": round(sc, 4),
                })
                n_boxes += 1
        results.append({"image_id": iid, "predictions": preds})

    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成。{len(results)} 图 / {n_boxes} 框")
    if n_filtered:
        print(f"  丢弃低分框: {n_filtered}（阈值 {args.min_score}）")
    if n_bad_label:
        print(f"  丢弃未知类别框: {n_bad_label}")
    print(f"输出: {out_path}（{out_path.stat().st_size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
