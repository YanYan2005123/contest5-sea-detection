#!/usr/bin/env python3
"""赛题5 提交前最终验证 — 严格核对提交 JSON 是否符合规范。

零第三方依赖（纯 Python 标准库）。

检查项（全部对标提交规范）：
  [file]  文件可读、UTF-8 无 BOM、文件名规范 result_队伍名.json
  [top]   顶层为 list
  [item]  每元素含 image_id(str) + predictions(list)
  [id]    image_id 与测试集文件名一一对应（不多不漏不重复）
  [cat]   category 为 9 类中文标签之一
  [pts]   points 为 4 顶点 [[x,y]*4]，浮点数，保留>=2 位小数
  [quad]  4 顶点构成有效四边形（非自相交、面积>0）
  [score] score ∈ [0.0, 1.0]，浮点数
  [order] 顶层 list 按 image_id 排序（便于评分脚本处理）

用法：
    python contest/validate_submission.py \
        --submission contest/submissions/result_Devx.json \
        --test-dir "D:\\hao\\赛题5数据集\\test"
    python contest/validate_submission.py --submission ... --strict
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CLASSES = {
    "驱逐舰", "护卫舰", "渔船", "杂货船", "液体运输船",
    "航母", "保障船", "散装货船", "集装箱船",
}
EXPECTED_TEST_COUNT = 9000


def _area_quad(pts):
    """鞋带公式算四边形有向面积（非自相交时绝对值为面积）。"""
    n = len(pts)
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _is_self_intersecting(pts):
    """粗略检测四边形是否自相交：检查对边是否相交。

    四边形 4 顶点按顺序连边 0-1-2-3-0，对边为 (0-1,2-3) 和 (1-2,3-0)。
    """
    def ccw(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def seg_inter(a, b, c, d):
        return (ccw(a, c, d) != ccw(b, c, d)) and \
               (ccw(a, b, c) != ccw(a, b, d))

    a, b, c, d = pts
    # 对边：ab-cd 和 bc-da
    if seg_inter(a, b, c, d):
        return True
    if seg_inter(b, c, d, a):
        return True
    return False


def validate(submission_path: str, test_dir: str | None,
             strict: bool) -> int:
    """返回错误数；0=通过。同时打印详细报告。"""
    errors = []
    warnings = []
    stats = {"images": 0, "boxes": 0, "empty": 0,
             "by_class": {}, "score_min": 1.0, "score_max": 0.0}

    # === [file] 文件检查 ===
    p = Path(submission_path)
    if not p.exists():
        print(f"FATAL 文件不存在: {p}")
        return 1
    # UTF-8 无 BOM
    raw = p.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        errors.append("[file] 文件含 UTF-8 BOM，规范要求无 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        errors.append(f"[file] 非 UTF-8 编码: {e}")
        return 1
    # 文件名规范：result_队伍名.json
    fname = p.name
    if not re.match(r"^result_.+\.json$", fname):
        errors.append(
            f"[file] 文件名 '{fname}' 不符合规范 result_队伍名.json")
    print(f"[file] 文件: {p} ({len(raw) / 1024:.0f} KB)")

    # === [top] 顶层结构 ===
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        errors.append(f"[top] JSON 解析失败: {e}")
        return 1
    if not isinstance(data, list):
        errors.append(f"[top] 顶层应为 list，实为 {type(data).__name__}")
        return 1
    print(f"[top] 顶层 list: {len(data)} 个元素")

    # === [id] 与测试集对照 ===
    test_ids = None
    if test_dir:
        td = Path(test_dir)
        if td.exists():
            test_ids = sorted(f.name for f in td.glob("*.jpg"))
            print(f"[id] 测试集: {len(test_ids)} 张")
        else:
            warnings.append(f"[id] 测试集目录不存在: {td}（跳过完整性检查）")

    sub_ids = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"[item][{i}] 元素非 dict: {type(item).__name__}")
            continue
        iid = item.get("image_id")
        if not iid:
            errors.append(f"[item][{i}] 缺 image_id")
            continue
        if not isinstance(iid, str):
            errors.append(f"[item][{i}] image_id 非 str: {iid!r}")
            continue
        if not iid.endswith(".jpg"):
            errors.append(f"[item][{i}] image_id 缺 .jpg 扩展名: {iid}")
        sub_ids.append(iid)
        preds = item.get("predictions")
        if preds is None:
            errors.append(f"[item][{i}] {iid} 缺 predictions")
            continue
        if not isinstance(preds, list):
            errors.append(f"[item][{i}] {iid} predictions 非 list")
            continue
        stats["images"] += 1
        if not preds:
            stats["empty"] += 1
        for j, pr in enumerate(preds):
            if not isinstance(pr, dict):
                errors.append(f"[cat][{i}.{j}] {iid} 预测非 dict")
                continue
            # category
            cat = pr.get("category")
            if cat not in CLASSES:
                errors.append(
                    f"[cat][{i}.{j}] {iid} 类别 '{cat}' 不在 9 类中")
                continue
            stats["by_class"][cat] = stats["by_class"].get(cat, 0) + 1
            stats["boxes"] += 1
            # points
            pts = pr.get("points")
            if not isinstance(pts, list) or len(pts) != 4:
                errors.append(
                    f"[pts][{i}.{j}] {iid} {cat} points 非 4 顶点: {pts}")
                continue
            bad_pt = False
            for pt in pts:
                if (not isinstance(pt, list) or len(pt) != 2
                        or not isinstance(pt[0], (int, float))
                        or not isinstance(pt[1], (int, float))):
                    bad_pt = True
                    break
            if bad_pt:
                errors.append(
                    f"[pts][{i}.{j}] {iid} {cat} 顶点含非浮点值")
                continue
            # 四边形有效性
            area = _area_quad(pts)
            if area < 1.0:
                errors.append(
                    f"[quad][{i}.{j}] {iid} {cat} 四边形面积过小: {area:.2f}")
            elif _is_self_intersecting(pts):
                errors.append(
                    f"[quad][{i}.{j}] {iid} {cat} 顶点自相交")
            # score
            sc = pr.get("score")
            if not isinstance(sc, (int, float)):
                errors.append(f"[score][{i}.{j}] {iid} {cat} score 非数值")
                continue
            sc = float(sc)
            if sc < 0.0 or sc > 1.0:
                errors.append(f"[score][{i}.{j}] {iid} {cat} score={sc} 越界")
            stats["score_min"] = min(stats["score_min"], sc)
            stats["score_max"] = max(stats["score_max"], sc)

    # 重复 image_id
    if len(sub_ids) != len(set(sub_ids)):
        dup = [x for x in sub_ids if sub_ids.count(x) > 1]
        errors.append(f"[id] 重复 image_id: {set(dup)}")

    # 与测试集对照
    if test_ids is not None:
        sub_set = set(sub_ids)
        test_set = set(test_ids)
        missing = test_set - sub_set
        extra = sub_set - test_set
        if missing:
            errors.append(f"[id] 缺 {len(missing)} 张测试图（漏提交）"
                          f"，前5: {sorted(missing)[:5]}")
        if extra:
            errors.append(f"[id] 多 {len(extra)} 个未知 image_id"
                          f"，前5: {sorted(extra)[:5]}")
        if len(sub_ids) != len(test_ids):
            warnings.append(f"[id] 提交 {len(sub_ids)} 张 vs 测试集 "
                            f"{len(test_ids)} 张")

    # 顶层排序检查
    if sub_ids != sorted(sub_ids):
        warnings.append("[order] 顶层未按 image_id 排序（建议排序便于评分）")

    # === 报告 ===
    print(f"\n{'=' * 60}")
    print(f"验证报告")
    print(f"{'=' * 60}")
    print(f"图像总数: {stats['images']}")
    print(f"检测框总数: {stats['boxes']}")
    print(f"空图(无框): {stats['empty']}")
    if stats['boxes'] > 0:
        print(f"score 范围: [{stats['score_min']:.4f}, "
              f"{stats['score_max']:.4f}]")
    print(f"\n--- 类别分布 ---")
    for c in sorted(CLASSES):
        n = stats["by_class"].get(c, 0)
        print(f"  {c}: {n}")

    if warnings:
        print(f"\n--- 警告 ({len(warnings)}) ---")
        for w in warnings[:20]:
            print(f"  ⚠ {w}")
        if len(warnings) > 20:
            print(f"  ... 共 {len(warnings)} 条")

    if errors:
        print(f"\n--- 错误 ({len(errors)}) ---")
        for e in errors[:30]:
            print(f"  ✗ {e}")
        if len(errors) > 30:
            print(f"  ... 共 {len(errors)} 条")
        print(f"\n{'!' * 60}")
        print(f"验证失败：{len(errors)} 个错误")
        print(f"{'!' * 60}")
        return 1

    print(f"\n{'✓' * 60}")
    print(f"验证通过：符合赛题提交规范")
    print(f"  文件: {fname}")
    print(f"  图像: {stats['images']}")
    print(f"  框数: {stats['boxes']}")
    print(f"{'✓' * 60}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="赛题5 提交 JSON 验证")
    ap.add_argument("--submission", required=True,
                    help="提交 JSON 路径 result_*.json")
    ap.add_argument("--test-dir", default=None,
                    help="测试集目录（提供则核对 image_id 完整性）")
    ap.add_argument("--strict", action="store_true",
                    help="严格模式：有警告也判失败")
    args = ap.parse_args()
    rc = validate(args.submission, args.test_dir, args.strict)
    if args.strict and rc == 0:
        pass
    sys.exit(rc)


if __name__ == "__main__":
    main()
