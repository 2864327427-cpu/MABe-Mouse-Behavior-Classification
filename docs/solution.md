# MABe Challenge - Social Action Recognition in Mice（MABe2025）解决方案报告

**赛题**：小鼠社会行为识别（Social Action Recognition in Mice）
**周期**：2025-09-18 ～ 2025-12-15
**最终成绩**：CV **0.503**；Public LB **0.49**；Private LB **0.47**

---

## 1) 比赛理解

### 1.1 背景与目标

基于顶视视频的无标记姿态/运动捕捉数据，自动识别同居小鼠的 **30+** 种社会/非社会行为，并尽可能达到人类观察者水平且对跨实验室采集差异具备泛化能力。

**补充说明（结合代码实现）**
本方案把任务拆成两条“数据形态”分支：

* **single**：自行为（agent=target/self）
* **pair**：双鼠交互（agent≠target）
  并进一步按 `body_parts_tracked`（不同实验室关键点集合）分区训练与推理，以对齐“不同实验室跟踪部位不同”的核心难点。

---

### 1.2 竞赛数据

本次竞赛使用隐藏测试集。当提交的 notebook 被评分时，实际测试数据（包括一个完整的样本提交）会提供给 notebook。预计隐藏测试集包含约 **200** 个视频。

**文件：**

* **[train/test].csv**：包含小鼠及其录制设置元数据

  * `lab_id`：提供数据实验室的化名。CalMS21、CRIM13 和 MABe22 数据集是公开可用数据集的副本，作为额外训练数据提供。CalMS21 部分跟踪文件存在较多重复（不同人员针对不同行为集标注）。
  * `video_id`：视频唯一标识符。
  * `mouse[1-4] [strain/color/sex/id/age/condition]`：每只小鼠基本信息。
  * `frames per second`、`video duration (sec)`
  * `pix per cm (approx)`
  * `video [width/height]`、`arena [width/height] (cm)`、`arena shape`、`arena type`
  * `body parts tracked`：不同实验室跟踪的身体部位不同。
  * `behaviors labeled`：视频中标注的行为（**稀疏标注**，只对标注到的行为/小鼠计分）。
  * `tracking method`：姿态跟踪方法。

* **[train/test]_tracking/**：关键点轨迹特征数据（长表）

  * `video_frame`：帧号
  * `mouse_id`：小鼠标识
  * `bodypart`：身体部位
  * `[x/y]`：像素坐标

* **train_annotation/**：训练标签（片段级，start/stop）

  * `agent_id`、`target_id`、`action`、`[start/stop]_frame`

* **sample_submission.csv**：提交格式样例

  * `row_id, video_id, agent_id, target_id, action, start_frame, stop_frame`

---

### 1.3 评估指标

官方说明：

> 本次竞赛使用了一种F-Score变体作为评价指标。F分数在每个实验室、每个视频中进行平均，并且仅对特定视频中被标注的具体行为和小鼠进行评分。

对应的核心形式可写为 F_(\beta)（本方案代码默认 (\beta=1)）：
$$
F_{\beta}=\frac{(1+\beta^2)\cdot TP}{(1+\beta^2)\cdot TP+\beta^2\cdot FN+FP}
$$
并在实现中按 **action** 聚合帧级 TP/FP/FN，再对 action 取平均；同时再对 **lab** 做平均（`mouse_fbeta`）。

**结合 `utils_mabe2025.py` 的关键计分细节（与“稀疏标注”强相关）**

* 只对 `behaviors_labeled` 中出现的 (agent,target,action) 键进行计分；不在 active labels 的预测会被忽略（既不加 FP，也不加 TP）。
* 预测按帧集合去重：同一 `prediction_key` 重复预测的帧会被差集剔除。
* 代码中对 behaviors_labeled 的 `"self"` 做了标准化映射（如 `"mouse1,self,dig"` → `"1,1,dig"` / `"1_1_dig"`），避免 self 行为在评测时对不上 key。

---

### 1.4 领域知识入门

这题本质是 **“从姿态点（keypoints）时间序列 → 行为片段检测”**：

* 输入不是原始视频像素，而是每帧若干身体部位的 (x,y) 轨迹（不同实验室部位集合不同）。
* 输出不是逐帧分类分数，而是 **事件片段**（start/stop frame）+ 行为类别 + 主体/目标（agent/target）。
* 最难的两点：

  1. **跨实验室域差异**：帧率、像素尺度、场地大小、跟踪点集合不同；
  2. **稀疏标注与极度不平衡**：很多行为很少发生；且只对标注过的行为计分。

**术语词汇表（10条）**

1. **Keypoints / body parts**：身体部位关键点（如 nose、tail_base）。
2. **Pose tracking**：姿态跟踪方法，输出关键点轨迹。
3. **FPS（frames per second）**：帧率；决定“多少帧≈多少秒”。
4. **Pix per cm (PPM)**：像素到厘米的尺度；用于把不同视频统一到物理尺度。
5. **Single action**：自行为，agent=target（或 target=self）。
6. **Pair action**：交互行为，agent≠target（如 sniff、chase）。
7. **Sparse labels**：稀疏标注；只标注部分行为/部分鼠/部分片段。
8. **Event/segment**：行为事件片段（start_frame, stop_frame）。
9. **Class imbalance**：类别不平衡；正样本比例可能远低于 1%。
10. **Domain shift**：域偏移；不同实验室采集/标注/跟踪导致分布变化。

---

## 2) 解决方案解析

### 2.1 方案流程概览

**(A) 数据预处理**

* 读取 `train.csv/test.csv` 元数据；过滤明显异常/不一致数据：

  * 训练中删除疑似休眠片段（MABe22 + “lights on”）
  * 跳过已知 identity permutation 的视频 `1212811043`
* 按 `body_parts_tracked` 将数据划分为多个 **section**（`body_parts_tracked_dict` 0~8）。
* 每个视频读取 `*_tracking/*.parquet`，透视为宽表并做 **像素→厘米**归一：`pvid = pvid / pix_per_cm_approx`。
* 训练集读取 `train_annotation/*.parquet`，并修复已知 fps 标注错配（AdaptableSnail：25fps pose / 30fps annot → 乘 (25/30) 校正）。

**(B) 模型设计与训练**

* **按 action 单独训练二分类器**（是否为该 action 的帧）：

  * 对每个 section、每个分支（single/pair）、每个 action：抽取该 action 非空标签帧，构造 (X, y)。
* **按 video_id 划分 CV**（默认 valid=10% 视频），避免帧级泄漏。
* **类不平衡处理**：`rebalance_pos_neg` 默认 `downsample_neg`，目标正样本比例 `target_pos_ratio=0.01`。
* **模型：**GPU XGBoost（`tree_method="gpu_hist"`, `n_estimators=2000`, `learning_rate=0.05`, `max_leaves=255`…）

**(C) 推理与后处理**

* 推理阶段按 section + single/pair 生成样本（`generate_mouse_data`），再做对应特征工程（single v52 / pair v213）。
* **按 action 加载对应模型 artifact**（每个 action 一个 pkl），并强制对齐训练期 `feature_names`：缺失列补 0，多余列丢弃。
* 逐帧概率 → 片段：

  * 5 帧 rolling mean 平滑
  * 每帧取 argmax 动作，并用动作阈值（默认 0.27）过滤为背景
  * run-length 编码生成 start/stop
  * 过滤长度 < 3 帧的短片段
* `robustify`：

  * 删除 start>=stop
  * 同一 (video,agent,target) 的片段做时间互斥去重（贪心去重叠）
  * 对“完全无预测”的视频做 fail-safe 填充（按 behaviors_labeled 均分时间段，保证提交不为空）

---

### 2.2 关键技术点

#### 2.2.1 模型选择与原因

* **树模型（XGBoost）+ 大规模手工时序特征**的组合，适配本题的两大约束：

  1. 输入是结构化 keypoints（而非原视频像素），可以高效构造几何/运动学特征；
  2. action 极多且不平衡严重，**“按 action 的独立二分类”**让每个分类器只解决一个稀疏问题，且便于按 action 调 `scale_pos_weight` 与重采样。

#### 2.2.2 特征工程 / 数据增强（核心清单）

本方案的特征工程非常“物理直觉驱动”，并且 **显式做了 FPS 与尺度自适应**：

**Single（`transform_single_v52`）核心簇：**

* **几何距离簇**：同一只鼠的关键点两两距离（组合特征，适配不同 body_parts）。
* **中心运动学**（`add_center_kinematics_v2`）：多 lag 位移、速度/加速度/jerk、多窗口统计、短长对比。
* **曲率/曲折度/状态特征**：curvature、tortuosity、速度状态占比与状态切换次数。
* **长时程与累计路程**：长窗口位置均值、EWM、窗口内路径长度。
* **Arena 空间特征**：归一化坐标、到墙/角/中心距离、近墙占比（提升跨场地泛化）。
* **Grooming 微特征**：nose 相对 body 的“头身解耦”、径向抖动、朝向抖动等。
* **非因果上下文**：关键特征 lag/lead（例如 `nt_dist`, `elong`, `sp_m5` 的 past/future）用于捕捉动作起止边界的对称模式。
* **形态学点云 PCA**：`morph_len/morph_wid/morph_ratio` 抽象“伸展 vs 蜷缩”。

**Pair（`transform_pair_v213`）核心簇：**

* **跨鼠距离网格**：A 的部位 × B 的部位两两距离（交互语义基础）。
* **接近/追逐/协同**：approach rate、lead 指标、chase 强度、速度相关性、速度对齐（余弦相似）。
* **Sniff proxy**：nose→face/body/genital 的代理距离与动态（例如 `snf_dmin_face`, `snf_dmin_genital`）。
* **Pair grooming 特征**：A nose 相对 B center 的高频 jitter + B 的静止程度（区分 grooming vs tussle 等）。
* **A-centric egocentric 坐标**：把 B 的位置投到 A 的前后/左右坐标系（对 follow/chase/approach 很关键）。
* **关键特征 lag/lead**：对 `ab_cd/nn/appr/...` 做过去/未来偏移，增强边界检测与短时模式识别。



#### 2.2.3 训练策略

* **划分**：按 `video_id` 划分 train/valid（10% videos），且在每个 `body_parts_tracked` section 内独立划分并保存视频列表。

* **重采样**：对训练集执行 `downsample_neg`，把正样本比例提升到 `target_pos_ratio`（默认 1%），缓解极端不平衡。

* **不平衡权重**：XGBoost 训练时设置 `scale_pos_weight = n_neg/n_pos`（逐 action 自适应）。

* **逐 action 训练**：如果某 action 在该子集里只有单一类别（全 0 或全 1），直接跳过，避免无意义训练。

  

---

### 2.3 代码解析

#### 2.3.1 `data_processing.py`：样本生成与标注对齐

* **职责**：

  * 逐视频读取 tracking parquet（长表）→ 透视为宽表（MultiIndex 列）→ 像素转厘米；
  * 解析 `behaviors_labeled`，确定该视频需要生成哪些 (agent,target,actions)；
  * 训练集读取 annotation parquet，并把片段转为逐帧布尔标签矩阵。
* **关键输出（generator）**：

  * `yield 'single', single_df, meta_df, y_df` 或 `yield 'pair', pair_df, meta_df, y_df`
* **关键点**：

  * 只为 `behaviors_labeled` 出现过的 pair 组合生成样本（减少无效计算）；
  * 对 AdaptableSnail 的 fps mismatch 做 start/stop 纠正；
  * `_resolve()` 兼容 0/1-based、"mouse3" 等命名差异，提升跨实验室对齐鲁棒性。

#### 2.3.2 `feature_engineering_single.py`：single v52 特征

* **职责**：把单鼠关键点时序转换为大量可学习特征（几何+运动学+空间）。
* **输入/输出**：

  * 输入：`single_mouse`（列：bodypart×{x,y}）、fps、pix_per_cm、arena 信息
  * 输出：`X`（float32 特征表，逐帧）
* **关键参数**：

  * `_scale()`：把“以30fps为基准的帧窗口”缩放到实际 fps，保证时间尺度一致。
  * `add_key_lags=True` + `key_lag_features=["nt_dist","elong","sp_m5"]`：只对少量关键列做显式 lag/lead，控制特征膨胀。

#### 2.3.3 `feature_engineering_pair.py`：pair v213 特征

* **职责**：把双鼠关键点时序转换为交互特征（距离、相对朝向、追逐、嗅探代理、A-centric 坐标等）。
* **输入/输出**：

  * 输入：`mouse_pair`（一级列 A/B，二级 bodypart×{x,y}）、fps、arena/尺度
  * 输出：逐帧特征表 `X`
* **关键点**：

  * `add_sniff_proxy_features_pair()` 把 sniff 相关语义显式化；
  * `add_pair_grooming_features()` 用相对 jitter + target 静止刻画 dominance_groom 等动作；
  * `add_egocentric_position_features_pair()` 输出 `ego_*` 系列特征，增强 follow/chase 等方向性行为。

#### 2.3.4 `models.py`：模型封装与构建

* **职责**：构建可复用的 sklearn 风格分类器封装。
* **核心模型**：`XGBClassifier(tree_method="gpu_hist", n_estimators=2000, learning_rate=0.05, max_leaves=255, ...)`
* **不平衡处理**：在 `fit()` 内设置 `scale_pos_weight=n_neg/n_pos`；并选择 `eval_metric`（极不平衡时倾向 `aucpr`）。

> 代码摘录（来自 `models.py`，体现逐 action 自适应不平衡权重）：

```python
n_pos = max(1, int((y_train == 1).sum()))
n_neg = max(1, len(y_train) - n_pos)
self.estimator.set_params(scale_pos_weight=(n_neg / n_pos))
```

#### 2.3.5 `train.py`：分区、分支、逐 action 训练与产物保存

* **职责**：

  * 读取元数据、按 section 遍历；
  * 调用 `generate_mouse_data` 生成 single/pair 样本；
  * 调用 transform 生成特征（支持 parquet cache）；
  * 对每个 action 单独划分 train/valid、重采样、训练模型；
  * `save_artifacts()` 保存：`models + model_names + feature_names`（用于推理严格对齐列）。
* **关键设计**：按 `body_parts_tracked` 分区训练 + 按 action 训练二分类器。

#### 2.3.6 `inference.py`：按 action 加载模型、特征对齐、输出逐帧概率并片段化

* **职责**：

  * 按 section + single/pair 生成测试样本；
  * 计算特征；
  * 逐 action 推理并平均多个模型（本配置通常只有一个 `xgb1`）；
  * 交给 `predict_multiclass_adaptive` 生成片段，再 `robustify` 做合规化。
* **关键点**：推理阶段用 `feature_names` 对齐训练列集合：缺列补 0，防止不同 section/不同视频导致列不一致。

#### 2.3.7 `post_precessing.py`：从逐帧到片段 + 提交鲁棒化

* `predict_multiclass_adaptive`：

  * rolling mean 平滑（window=5）
  * argmax 多分类（每帧只保留一个动作）
  * 阈值过滤（默认 0.27）
  * RLE 生成 start/stop，并修正跨视频/跨个体边界
  * 丢弃 <3 帧片段
* `robustify`：

  * 移除无效片段、去重叠
  * 对“无预测视频”做 fail-safe 填充（避免空提交导致异常）

#### 2.3.8 `utils_mabe2025.py`：评测实现与兼容修复

* **职责**：复刻并修复赛题 metric 的关键细节，尤其是 `behaviors_labeled` 中 `"self"` 的解析与 key 标准化，避免 self 行为无法计分的问题。

---

### 2.4 结果与总结

#### 2.4.1 最终结果

* **CV**：0.503
* **Public LB**：0.49
* **Private LB**：0.47

从三者关系看（CV > Public > Private），符合该赛题“跨实验室泛化 + 隐藏测试更难”的典型现象：Private 更像真实分布/更强域偏移。

#### 2.4.2 Ablation 量化表

| 组件增量（按代码模块）                                       | 估计CV变化（区间） | 不确定性来源                         |
| ------------------------------------------------------------ | -----------------: | ------------------------------------ |
| 来自kaggle baseline, 基础几何距离 + XGB（single/pair最简版） |              0.449 |                                      |
| 加入 FPS 自适应多窗口运动学（single v52 的 kinematics/tortuosity/state 等） |           ~ +0.016 | 不同行为对时序特征敏感度不同         |
| 加入 Arena 空间特征（wall/corner/center）                    |           ~ +0.008 | 场地信息缺失/噪声会影响              |
| 加入 Sniff proxy + Egocentric pair 特征（pair v213）         |           ~ +0.012 | 取决于 sniff/follow/chase 在测试占比 |
| 类不平衡处理（downsample_neg 到 1% + scale_pos_weight）      |           ~ +0.010 | 各 action 正样本稀疏程度差异巨大     |
| 逐帧→片段后处理（平滑+阈值+去重叠+fail-safe）                |           ~ +0.008 | 阈值与动作碎片化程度强相关           |

**总结一句话**
这套方案用“分区（section）+ 分支（single/pair）+ 逐 action 二分类”的工程化拆解，配合大量 **物理可解释的时序/几何特征** 与严格的提交鲁棒化，把结构化姿态序列问题稳定转化为树模型可学习的表格任务，最终取得 **CV 0.503 / Private 0.47**。

---





## 3) 简历项目模板（可直接粘贴）

**Kaggle MABe Challenge – Social Action Recognition in Mice | 2025.09–2025.12 | Private LB 0.47（CV 0.503）**

* **背景/挑战**：基于顶视视频的无标记姿态关键点序列，检测 30+ 小鼠社会/非社会行为；跨 20+ 实验室采集系统带来强域偏移，且标注稀疏、类别极不平衡。
* **方案亮点**：
  * 将任务拆为 **single（自行为）/pair（交互）** 双分支，并按 `body_parts_tracked` 分区训练，保证特征语义稳定、减少跨实验室缺失与错配。
  * 构建可解释的时序特征体系（FPS 自适应多窗口运动学、曲率/曲折度、arena 空间特征），并引入 **A-centric egocentric 坐标** 与 **sniff 部位代理**，显式刻画追逐/跟随/嗅探等交互语义。
  * 对每个 action 独立训练 GPU XGBoost 二分类器，结合 `scale_pos_weight` 与负样本下采样缓解极端不平衡；推理阶段严格对齐训练特征列并进行片段化后处理（平滑+阈值+去重叠）。

* **量化结果**：CV **0.503**；Public LB **0.49**；Private LB **0.47**。
* **技术栈**：Python，Pandas/Polars，XGBoost(GPU)，特征工程（时序/几何/空间），自定义评测与事件片段后处理，Kaggle Notebook 工程化缓存与产物管理。
