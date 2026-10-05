# 赛题5：跨大视角下的海面目标检测与识别

> 全国大学生计算机设计大赛 · 海面舰船旋转框检测（OBB）
> 用卫星俯视数据，训出无人机斜视检测能力

## 任务概述

| 项目 | 说明 |
|------|------|
| 任务类型 | 旋转框目标检测（Oriented Bounding Box） |
| 输入 | 卫星俯视图像 + 无人机斜视/平视图像 |
| 输出 | 9 类舰船的 4 顶点旋转检测框 |
| 训练集 | 6000 张（4800 俯视 + 1200 斜视，带标注） |
| 测试集 | 9000 张（无人机斜视视角，无标注） |
| 提交格式 | JSON，含 image_id / category / points / score |

### 9 类目标

驱逐舰、护卫舰、渔船、杂货船、液体运输船、航母、保障船、散装货船、集装箱船

### 核心挑战

卫星俯视（源域，数据充足）与无人机斜视（目标域，标注稀缺）之间存在**视角域鸿沟**：
同一艘船在俯视图像中是甲板平面形态，在斜视图像中是舷侧立面形态，外观差异极大。
本方案通过强跨视角数据增广实现域泛化，无需目标域标注即可迁移。

---

## 目录结构

```
contest5-sea-detection/
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── classes.yaml               # 类别映射（中文 ↔ 索引）
│
├── scripts/                   # 核心代码
│   ├── labelme_to_yolo_obb.py # LabelMe JSON → YOLO-OBB 格式转换
│   ├── train_detector.py      # YOLOv8-OBB 训练（含跨视角增广）
│   ├── predict_batch.py       # 批量推理（断点续跑）
│   ├── format_submission.py   # 推理结果 → 比赛提交 JSON
│   ├── local_eval.py          # 本地 mAP 评测
│   └── validate_submission.py # 提交 JSON 格式校验
│
├── data/
│   └── yolo_obb/
│       ├── data.yaml          # 数据集配置（相对路径）
│       ├── images/            # ⚠️ 需自行放入比赛图片（见 images/README.md）
│       └── labels/           # ✅ 已提供（4799 train + 1201 val）
│
└── runs/                      # 训练/推理产物（gitignore，不入库）
```

---

## 快速开始

### 1. 环境搭建

```powershell
# Python >= 3.10, CUDA >= 11.8, GPU 显存 >= 8GB
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 准备数据

1. 从比赛官网下载数据集
2. 将训练图片按 `data/yolo_obb/images/{train,val}/` 放置（见 `data/yolo_obb/images/README.md`）
3. 标注文件已在 `data/yolo_obb/labels/` 中，无需重新转换

### 3. 训练

```powershell
python scripts/train_detector.py
# 自定义参数：
python scripts/train_detector.py --model yolov8m-obb.pt --epochs 100 --imgsz 1024 --device 0
```

**跨视角增广策略**（已内置）：
- `perspective=0.0005` 透视畸变 → 模拟斜视几何
- `degrees=15` 大角度旋转 → 模拟视角偏转
- `flipud=0.3` + `transpose=0.3` 翻转/转置 → 覆盖横竖屏
- `mosaic=1.0` + `mixup=0.1` 混合增强 → 提升小目标鲁棒性

产物：`runs/sea_obb/weights/best.pt`

### 4. 推理（9000 张测试图）

```powershell
python scripts/predict_batch.py `
    --weights runs/sea_obb/weights/best.pt `
    --src <测试集图片目录> `
    --out runs/preds `
    --device 0 --conf 0.25 --iou 0.7 --imgsz 1024
```

### 5. 生成提交 JSON

```powershell
python scripts/format_submission.py `
    --preds runs/preds `
    --out result_<队伍名>.json `
    --src <测试集图片目录>
```

### 6. 提交前校验

```powershell
python scripts/validate_submission.py `
    --submission result_<队伍名>.json `
    --test-dir <测试集图片目录>
```

校验项：文件编码、JSON 结构、image_id 完整性、9 类合法性、4 顶点有效性、score 范围、排序。

---

## 技术方案

- **模型**：YOLOv8m-OBB（Ultralytics）
- **输入尺寸**：1024×1024
- **训练轮数**：100 epochs（patience=20 早停）
- **增广**：透视畸变 + 大角度旋转 + 上下翻转 + 转置 + mosaic/mixup
- **推理**：conf=0.25, IoU=0.7，断点续跑

## 评测指标

mAP@0.5 / mAP@0.5:0.95 / Precision / Recall

---

## 许可

本仓库代码仅供学术交流与比赛复现使用。数据集图片版权归比赛主办方所有。
