import re
import json
import itertools
import numpy as np
import pandas as pd

# ================== Known annotation issues fix ==================
ANNOT_FPS_FIX_LABS = {'AdaptableSnail'}
ANNOT_FPS_SCALE = 25.0 / 30.0

BAD_IDENTITY_VIDEOS = {('AdaptableSnail', 1212811043)}  # safest to skip in train

def _fix_known_annotation_issues(annot: pd.DataFrame, lab_id, video_id, fps_pose, max_frame):
    annot = annot.copy()

    # 1) Fix fps-mismatch annotations for AdaptableSnail 25fps videos
    # Host: annotations were rescaled to 30fps while pose is at 25fps.
    # Multiply start/stop by 25/30 for better synchronization.
    if (lab_id in ANNOT_FPS_FIX_LABS) and (abs(float(fps_pose) - 25.0) < 0.2):
        annot['start_frame'] = (annot['start_frame'] * ANNOT_FPS_SCALE).round().astype(int)
        annot['stop_frame']  = (annot['stop_frame']  * ANNOT_FPS_SCALE).round().astype(int)

    # 2) Clip to valid range and sanitize
    if max_frame is not None:
        annot['start_frame'] = annot['start_frame'].clip(0, max_frame)
        annot['stop_frame']  = annot['stop_frame'].clip(0, max_frame)

    # ensure start <= stop
    bad = annot['start_frame'] > annot['stop_frame']
    if bad.any():
        tmp = annot.loc[bad, 'start_frame'].copy()
        annot.loc[bad, 'start_frame'] = annot.loc[bad, 'stop_frame']
        annot.loc[bad, 'stop_frame'] = tmp

    return annot

def generate_mouse_data(CFG, dataset, traintest, drop_body_parts, traintest_directory=None, generate_single=True, generate_pair=True):
    """
    基于元数据逐视频生成训练/测试样本。
    读取 tracking parquet → 透视为多层列 → 归一化为厘米。
    解析 behaviors_labeled 得到 (agent,target,action)。
    产出两类样本：
      - 'single'：个体动作，返回(X, meta, y/actions)
      - 'pair'  ：成对交互，返回(X, meta, y/actions)
    训练集返回布尔帧级标签 y；测试集返回动作名列表。
    """
    assert traintest in ['train', 'test', 'valid']  # 仅支持 train/test/valid 三种分割
    if traintest_directory is None:
        # 默认的 tracking 根目录，按 Kaggle 挂载路径拼接
        if traintest == 'valid':
            traintest_directory = f"{CFG.comp_dir}/train_tracking"
        else:
            traintest_directory = f"{CFG.comp_dir}/{traintest}_tracking"

    def _to_num(x):
        # 如果是整数直接返回
        if isinstance(x, (int, np.integer)): return int(x)
        # 解析字符串末尾数字（如 "mouse3" -> 3），便于对齐不同实验室命名
        m = re.search(r'(\d+)$', str(x))
        return int(m.group(1)) if m else None

    # 遍历该关键点集合对应的所有视频行
    for _, row in dataset.iterrows():
        # ------- 基础元数据提取（尽量转为基础标量类型） -------
        lab_id   = row.lab_id
        video_id = row.video_id
        fps      = float(row.frames_per_second)  # 帧率，用于时间尺度自适应
        n_mice   = int(row.n_mice)               # 场景中小鼠数量（1~4）
        arena_w  = float(row.get('arena_width_cm', np.nan))   # 场地宽(cm)
        arena_h  = float(row.get('arena_height_cm', np.nan))  # 场地高(cm)
        sleeping = bool(getattr(row, 'sleeping', False))      # 是否“lights on”推断的休眠状态
        arena_shape = row.get('arena_shape', 'rectangular')   # 场地形状

        # 无行为列表（稀疏标注）则跳过该视频
        if not isinstance(row.behaviors_labeled, str):
            continue

        # ---- tracking表 ----
        # 读取该视频的 tracking（长表：列含 video_frame/mouse_id/bodypart/x/y）
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        # 若关键点较多，则剔除预定义的不稳定关键点，减噪
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        # 透视为宽表，多层列索引：(mouse_id, bodypart) → 值为 x/y
        pvid = vid.pivot(columns=['mouse_id','bodypart'], index='video_frame', values=['x','y'])
        del vid  # 释放内存
        # 调整列层级顺序为 (mouse_id, bodypart, xy)，并排序，便于后续切片
        pvid = pvid.reorder_levels([1,2,0], axis=1).T.sort_index().T
        # 像素转厘米：用 pix_per_cm_approx 做缩放；后续几何量即为 cm 量纲
        pvid = (pvid / float(row.pix_per_cm_approx)).astype('float32', copy=False)

        # tracking 中出现的 mouse_id 标签集合（统一做多种等价表示的集合）
        avail = list(pvid.columns.get_level_values('mouse_id').unique())
        avail_set = set(avail) | set(map(str, avail)) | {f"mouse{_to_num(a)}" for a in avail if _to_num(a) is not None}

        # ---- 主表 中获得的 behaviors ----
        # behaviors_labeled 是字符串化列表，每个元素形如 "mouse1,self,groom" 或 "mouse1,mouse2,chase"
        vb = json.loads(row.behaviors_labeled)
        vb = sorted(list({b.replace("'", "") for b in vb}))  # 去单引号与去重
        vb = pd.DataFrame([b.split(',') for b in vb], columns=['agent','target','action'])
        vb['agent']  = vb['agent'].astype(str)
        vb['target'] = vb['target'].astype(str)
        vb['action'] = vb['action'].astype(str).str.lower()  # 动作统一小写

        # 训练集需要加载逐帧标注 parquet，用于构造帧级布尔矩阵 y
        if traintest == 'train':
            try:
                # 对应路径：train_tracking → train_annotation
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                # 某些视频可能缺标注文件，直接跳过
                continue

            # ---- Known identity-permutation video: safest skip ----
            if (lab_id, int(video_id)) in BAD_IDENTITY_VIDEOS:
                continue

            # ---- Fix fps-mismatch annotation timeline ----
            max_frame = int(pvid.index.max())
            annot = _fix_known_annotation_issues(annot, lab_id, video_id, fps, max_frame)


        def _resolve(agent_str):
            """返回与 agent_str 对应、且实际存在于 pvid 列中的 mouse_id 标签；若无匹配则返回 None。"""
            # 解析末尾数字，构造候选形式：1 基、0 基、字符串、"mouse{n}"、原始
            m = re.search(r'(\d+)$', str(agent_str))
            cand = [agent_str]
            if m:
                n = int(m.group(1))
                cand = [n, n-1, str(n), f"mouse{n}", agent_str]  # 兼容 1-based/0-based/字符串/规范名
            for c in cand:
                if c in avail_set:  # 若在等价集合中
                    # 若刚好与列中元素类型一致，直接返回
                    if c in set(avail): return c
                    # 否则在现有标签里找一个“内容等价”的真实列名（优先保留原始类型）
                    for a in avail:
                        if str(a) == str(c) or f"mouse{_to_num(a)}" == str(c):
                            return a
            return None

        def _mk_meta(index, agent_id, target_id):
            # 构造与特征行对齐的元数据 DataFrame（用于提交与后处理）
            m = pd.DataFrame({
                'lab_id':        lab_id,
                'video_id':      video_id,
                'agent_id':      agent_id,
                'target_id':     target_id,
                'video_frame':   index.astype('int32', copy=False),   # 帧号（行索引）
                'frames_per_second': np.float32(fps),                 # 帧率（float32）
                'sleeping':      sleeping,
                'arena_shape':   arena_shape,
                'arena_width_cm': np.float32(arena_w),
                'arena_height_cm': np.float32(arena_h),
                'n_mice':        np.int8(n_mice),
            })
            # 若干分类字段转为 category，减小内存与便于下游 groupby
            for c in ('lab_id','video_id','agent_id','target_id','arena_shape'):
                m[c] = m[c].astype('category')
            return m

        # -------- 单鼠（自行为）样本 --------
        # 仅处理自行为（target == 'self'）；每个主体独立生成一条时间序列
        if generate_single:
            vb_single = vb.query("target == 'self'") # 主表中“单鼠”的['agent','target','action'] 行
            for agent_str in pd.unique(vb_single['agent']):
                # 将 behaviors 中的主体字符串映射为 pvid 里的真实列标签
                agent_id = _resolve(agent_str)
                if agent_id is None:
                    # 若 tracking 中不存在该主体，则跳过
                    continue
                # 该主体的所有自行为动作集合
                actions = sorted(vb_single.loc[vb_single['agent'].eq(agent_str), 'action'].unique().tolist())
                if not actions:
                    continue

                # single: MultiIndex 列 (bodypart, xy)，行为特征会在 transform_single 中构造
                single = pvid.loc[:, agent_id]
                # 构建与当前主体对应的元数据
                meta_df = _mk_meta(single.index, agent_str, 'self')

                if traintest == 'train':
                    # 训练集：构造帧级布尔标签矩阵 y，列为动作名，索引为帧
                    a_num = _to_num(agent_id)  # 将列标签解析为数值 id，用于和标注对齐
                    y = pd.DataFrame(False, index=single.index.astype('int32', copy=False), columns=actions)
                    # 从标注中筛选 agent=target=该主体 的片段，区间 [start, stop]（闭区间/含 stop）
                    a_sub = annot.query("(agent_id == @a_num) & (target_id == @a_num)")
                    for i in range(len(a_sub)):
                        annot_row = a_sub.iloc[i]
                        a = str(annot_row.action).lower()
                        if a in y.columns:
                            # 将区间内的帧置 True（后续 decode 会转为片段）
                            y.loc[int(annot_row['start_frame']):int(annot_row['stop_frame']), a] = True
                    # 产出生成器元素：类型标识、原始时序、元数据、标签
                    yield 'single', single, meta_df, y
                else:
                    # 测试集：无标签，仅返回动作名列表，供模型按列对齐输出概率
                    yield 'single', single, meta_df, actions

        # -------- 成对（社会行为）样本 --------
        # 双鼠交互仅对 behaviors_labeled 中明确出现过的 (agent,target) 进行生成
        if generate_pair:
            vb_pair = vb.query("target != 'self'") # 主表中“成对”的['agent','target','action'] 行
            if len(vb_pair) > 0:
                # 允许的有标注的 (agent,target) 组合集合
                allowed_pairs = set(map(tuple, vb_pair[['agent','target']].itertuples(index=False, name=None)))

                # 遍历 pvid 中实际存在的 mouse_id 的全排列（有向对）
                for agent_num, target_num in itertools.permutations(
                        np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    # 标准化字符串形式（"mouse{k}"），用于与 behaviors 的文案对齐
                    agent_str = f"mouse{_to_num(agent_num)}"
                    target_str = f"mouse{_to_num(target_num)}"
                    # 若该对不在标注出现过的组合内，则跳过（减少无效计算）
                    if (agent_str, target_str) not in allowed_pairs:
                        continue

                    # 将字符串形式解析回 pvid 中真实列标签
                    agent_id = _resolve(agent_str)
                    target_id = _resolve(target_str)
                    if agent_id is None or target_id is None:
                        # 若缺少任一主体的 tracking，则跳过
                        continue

                    # 该对主体的所有交互动作集合
                    actions = sorted(
                        vb_pair.query("(agent == @agent_str) & (target == @target_str)")['action'].unique().tolist()
                    )
                    if not actions:
                        continue

                    # 拼接两只鼠的时序（列多层：['A'/'B'] × bodypart × xy）
                    pair_xy = pd.concat([pvid[agent_id], pvid[target_id]], axis=1, keys=['A','B'])
                    # 构建与该对主体对应的元数据
                    meta_df = _mk_meta(pair_xy.index, agent_str, target_str)

                    if traintest == 'train':
                        # 训练集：同样构造帧级布尔标签
                        a_num = _to_num(agent_id); b_num = _to_num(target_id)
                        y = pd.DataFrame(False, index=pair_xy.index.astype('int32', copy=False), columns=actions)
                        a_sub = annot.query("(agent_id == @a_num) & (target_id == @b_num)")
                        for i in range(len(a_sub)):
                            annot_row = a_sub.iloc[i]
                            a = str(annot_row.action).lower()
                            if a in y.columns:
                                y.loc[int(annot_row['start_frame']):int(annot_row['stop_frame']), a] = True
                        # 产出生成器元素：类型标识、原始时序、元数据、标签
                        yield 'pair', pair_xy, meta_df, y
                    else:
                        # 测试集：无标签，返回候选动作列表
                        yield 'pair', pair_xy, meta_df, actions


# get something from meta
def _ppm_from_meta(meta_df, fallback_lookup, default_ppm=12.0):
    """
    获取每厘米像素数 (Pixels Per CM)。
    策略：优先从 meta 读取；缺失则查表；再缺失用默认值(MABe22的平均值)。
    """
    col_name = 'pix per cm (approx)' # CSV中的列名
    # 有些元数据列名可能是下划线格式，做个防御性检查
    if col_name not in meta_df.columns:
        if 'pix_per_cm' in meta_df.columns:
            col_name = 'pix_per_cm'
            
    if col_name in meta_df.columns and pd.notnull(meta_df[col_name]).any():
        val = float(meta_df[col_name].iloc[0])
        # 防御性：防止 pix_per_cm 为 0 或 极小值导致除以零爆炸
        if val > 0.1: 
            return val
    
    vid = meta_df['video_id'].iloc[0]
    val = float(fallback_lookup.get(vid, default_ppm))
    return val if val > 0.1 else default_ppm

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    """
    获取视频帧率 FPS。
    策略：优先从当前 meta 数据中读取；若缺失，则回退到预先构建的 lookup 表查找；再失败则用默认值。
    """
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

def _arena_dim_from_meta(meta_df, fallback_lookup, col_name, default_val):
    """
    从 meta 或 lookup 获取 arena 尺寸（cm）。
    """
    if col_name in meta_df.columns and pd.notnull(meta_df[col_name]).any():
        try:
            val = float(meta_df[col_name].iloc[0])
            if val > 1e-3:
                return val
        except Exception:
            pass

    vid = meta_df['video_id'].iloc[0]
    try:
        val = float(fallback_lookup.get(vid, default_val))
    except Exception:
        val = default_val
    return val if val > 1e-3 else default_val

def _arena_shape_from_meta(meta_df, fallback_lookup, default_shape="square"):
    if 'arena_shape' in meta_df.columns and pd.notnull(meta_df['arena_shape']).any():
        return str(meta_df['arena_shape'].iloc[0])

    vid = meta_df['video_id'].iloc[0]
    return str(fallback_lookup.get(vid, default_shape))

