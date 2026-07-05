# Kaggle | MABe Challenge 2025 - 小鼠社会行为识别

本项目为 Kaggle MABe Challenge（小鼠社会行为多分类识别）的 **Top 3%（银牌）** 解决方案核心代码。
本仓库主要包含了基于无标记姿态关键点的时间序列特征工程、针对严重长尾分布的模型优化以及平滑后处理算法。

## 🌟 核心贡献

1. **多尺度特征与时空几何特征工程**：基于小鼠 2D 关键点序列，提取了多窗口（Multi-window）运动学特征、轨迹曲率（Curvature），并创新性地引入**自我中心相对坐标系（Egocentric Coordinates）**来刻画多鼠复杂交互行为。
2. **长尾分布处理**：通过设计包装器动态计算样本比例，自适应调整 XGBoost 的 `scale_pos_weight`，缓解了稀有行为类别的漏报问题。
3. **后处理算法优化**：设计了基于时间滑动窗口（Rolling Window）的平滑滤波与连通域过滤，最终将交叉验证（CV）分数提升至 0.503。

## 📁 仓库结构

```text
├── docs/
│   ├── solution.md                         # 完整方案报告（中文）
│   └── images/mouse_keypoints.png          # 小鼠关键点示意图
├── scripts/
│   ├── train.py                            # 训练脚本
│   ├── inference.py                        # 推理脚本
│   └── run_gpu0.sh                         # Shell 启动脚本
├── src/
│   ├── feature_engineering_single.py       # 52 个单鼠特征
│   ├── feature_engineering_pair.py         # 213 个成对交互特征
│   ├── models.py                           # XGBoost 动态正负样本加权
│   ├── data_processing.py                  # 数据加载与预处理
│   ├── post_precessing.py                  # 平滑与自适应阈值
│   ├── utils.py                            # 通用工具
│   └── utils_mabe2025.py                   # 评估指标、重采样、配置
├── requirements.txt
├── .gitignore
└── README.md
```

## 📁 流程概览

- `src/feature_engineering_*.py`：核心特征提取（包括角度、速度不对称性、相对微颤等超过百余个物理特征）。
- `src/models.py`：封装了具有动态正负样本重平衡的 `StratifiedSubsetClassifierWEval`。
- `src/post_precessing.py`：自适应多分类阈值与时序平滑逻辑。

## ⚙️ 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

从 [Kaggle MABe Challenge](https://kaggle.com/competitions/mabe-mouse-behavior-detection) 下载完整数据集，将数据放入项目根目录的 `input/` 文件夹，结构如下：

```text
input/
└── mabe-mouse-behavior-detection/
    ├── train.csv
    ├── test.csv
    ├── train_tracking/       # 训练集姿态跟踪数据
    ├── train_annotation/     # 训练集标注
    └── test_tracking/        # 测试集姿态跟踪数据
```

### 3. 训练

```bash
python scripts/train.py --gpu 0
```

或使用启动脚本：
```bash
bash scripts/run_gpu0.sh
```

训练脚本支持以下参数：
- `--gpu`：指定 GPU 编号（默认 `0`）
- `--model_name`：模型名称
- `--target_pos_ratio`：正样本目标比例（默认 `0.01`）
- `--single_transform_version` / `--pair_transform_version`：特征版本号

训练完成后，模型文件将保存至 `artifacts/` 目录。

### 4. 推理与提交

```bash
python scripts/inference.py --gpu 0
```

推理脚本会：
1. 加载已保存的模型文件
2. 对测试集进行特征工程
3. 按 `(body_parts_tracked, action)` 组合执行推理
4. 输出 `submission.csv` 提交文件

## 🛠 技术栈

Python | XGBoost | Pandas | NumPy | SciPy | Parquet
