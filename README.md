# Kaggle | MABe Challenge 2025 - Mouse Social Behavior Recognition

本项目为 Kaggle MABe Challenge (小鼠社会行为多分类识别) 的 Top3% (Silver Medal) 解决方案核心代码。
本仓库主要包含了基于无标记姿态关键点的时间序列特征工程、针对严重长尾分布的模型优化以及平滑后处理算法。

## 🌟 My Contributions 
1. **多尺度特征与时空几何特征工程**：基于小鼠 2D 关键点序列，提取了多窗口（Multi-window）运动学特征、轨迹曲率（Curvature），并创新性地引入**自我中心相对坐标系（Egocentric Coordinates）**来刻画多鼠复杂交互行为。
2. **长尾分布处理**：通过设计包装器动态计算样本比例，自适应调整 XGBoost 的 `scale_pos_weight`，缓解了稀有行为类别的漏报问题。
3. **后处理算法优化**：设计了基于时间滑动窗口（Rolling Window）的平滑滤波与连通域过滤，最终将交叉验证（CV）分数提升至 0.503。

## 📁 Repository Structure
```text
├── docs/
│   ├── solution.md                         # Full solution report (Chinese)
│   └── images/mouse_keypoints.png          # Mouse keypoint diagram
├── scripts/
│   ├── train.py                            # Main training script
│   ├── inference.py                        # Inference script
│   └── run_gpu0.sh                         # Shell launcher
├── src/
│   ├── feature_engineering_single.py       # 52 single-mouse features
│   ├── feature_engineering_pair.py         # 213 pairwise interaction features
│   ├── models.py                           # XGBoost with dynamic scale_pos_weight
│   ├── data_processing.py                  # Data loading & preprocessing
│   ├── post_precessing.py                  # Smoothing & adaptive thresholding
│   ├── utils.py                            # General utilities
│   └── utils_mabe2025.py                   # Metrics, resampling, config
├── requirements.txt
├── .gitignore
└── README.md
```

## 📁 Pipeline Overview
- `src/feature_engineering_*.py`: 核心特征提取（包括角度、速度不对称性、相对微颤等超过百余个物理特征）。
- `src/models.py`: 封装了具有动态正负样本重平衡的 `StratifiedSubsetClassifierWEval`。
- `src/post_precessing.py`: 自适应多分类阈值与时序平滑逻辑。
