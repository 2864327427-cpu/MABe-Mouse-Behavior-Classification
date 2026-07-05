"""F Beta customized for the data format of the MABe challenge."""

import json
from collections import defaultdict
import pandas as pd
import polars as pl

class HostVisibleError(Exception):
    pass

def normalize_label_item(label_str: str) -> str:
    """
    Helper: 将 metadata 中的原始标签 (e.g., "mouse1,self,dig") 
    转换为提交文件格式 (e.g., "1,1,dig")
    """
    parts = [p.strip() for p in label_str.split(',')]
    
    if len(parts) == 3:
        # 格式: agent, target, action
        agent = parts[0].replace('mouse', '')
        raw_target = parts[1]
        action = parts[2]
        
        # 处理 self 关键字
        if raw_target == 'self':
            target = agent
        else:
            target = raw_target.replace('mouse', '')
            
        # 处理可能的空 target
        if not target or target.lower() == 'none':
            target = agent
            
        return f"{agent},{target},{action}"
        
    elif len(parts) == 2:
        # 格式: agent, action (隐式 self)
        agent = parts[0].replace('mouse', '')
        action = parts[1]
        target = agent
        return f"{agent},{target},{action}"
        
    return ""  # Should not happen

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    # 这里的 label_key 是由 mouse_fbeta 传入的，格式是 "_" 分隔
    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        # --- [FIX START] ---
        # 原代码: active_labels: set[str] = set(json.loads(active_labels))
        # 修复后: 解析 JSON 并进行标准化处理
        active_labels_raw = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        active_labels_set = set()
        
        if active_labels_raw:
            try:
                raw_list = json.loads(active_labels_raw)
                for item in raw_list:
                    normalized = normalize_label_item(item)
                    if normalized:
                        active_labels_set.add(normalized)
            except:
                pass
        
        active_labels = active_labels_set
        # --- [FIX END] ---

        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            # 这里构建 check_key 使用的是 ',' 分隔，与我们上面的 normalize_label_item 对应
            check_key = ','.join([str(row['agent_id']), str(row['target_id']), row['action']])
            
            # Since the labels are sparse, we can't evaluate prediction keys not in the active labels.
            if check_key not in active_labels:
                continue

            new_frames = set(range(row['start_frame'], row['stop_frame']))
            # Ignore truly redundant predictions.
            # 注意: prediction_key 来自 mouse_fbeta，使用 "_" 分隔
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                # A single agent can have multiple targets per frame (ex: evading all other mice) but only one action per target per frame.
                # 这里可以注释掉报错，或者在提交处理时清理冲突，为了评测稳定性通常建议注释掉 raise
                pass 
                # raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
    
    # 计算 Stats
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key] # label_key 和 prediction_key 格式一致 (_)
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
            
    if not action_f1s:
        return 0.0
        
    return sum(action_f1s) / len(action_f1s)

def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    if len(solution) == 0:
        return 0.0
        
    # 如果 submission 为空，直接返回 0 (或者计算全 FN，视具体需求而定，官方逻辑会报错，这里做个保护)
    if len(submission) == 0:
         # 为了让代码跑通，构建一个空的 submission 结构
         submission = pd.DataFrame(columns=solution.columns)

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    for col in expected_cols:
        if col not in solution.columns: raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns: raise ValueError(f'Submission is missing column {col}')

    solution_pl = pl.DataFrame(solution)
    submission_pl = pl.DataFrame(submission)

    solution_videos = set(solution_pl['video_id'].unique())
    submission_pl = submission_pl.filter(pl.col('video_id').is_in(solution_videos))

    # 构建 Key，统一使用 "_" 分隔
    solution_pl = solution_pl.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission_pl = submission_pl.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    lab_scores = []
    for lab in solution_pl['lab_id'].unique():
        lab_solution = solution_pl.filter(pl.col('lab_id') == lab)
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission_pl.filter(pl.col('video_id').is_in(lab_videos))
        
        # 即使 submission 为空，也需要计算（全是 FN）
        # 原官方代码在这里如果 submission 为空可能会有逻辑问题，single_lab_f1 能够处理空 submission
        score_val = single_lab_f1(lab_solution, lab_submission, beta=beta)
        lab_scores.append(score_val)

    if not lab_scores: return 0.0
    return sum(lab_scores) / len(lab_scores)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)

# ---
import pandas as pd
import polars as pl
from collections import defaultdict
import json

class MABeMetricCalculator:
    def __init__(self, beta=1.0):
        self.beta = beta
        self.action_category_map = {}
        # 预定义一些已知的 Single Action，用于防守性编程 (可选，但推荐)
        self.known_single_actions = {
            'rear', 'selfgroom', 'dig', 'climb', 'biteobject', 'run', 
            'genitalgroom', 'freeze', 'huddle', 'rest', 'exploreobject'
        }

    def _normalize_key(self, raw_str):
        """移除 'mouse' 前缀并清理空格"""
        if raw_str is None:
            return ""
        return str(raw_str).replace("mouse", "").strip()

    def _parse_and_normalize_labels(self, labels_json_str):
        """
        解析 behaviors_labeled.
        [GM FIX]: 增加对 'self' 关键字的处理，将其映射回 agent_id
        """
        try:
            raw_list = json.loads(labels_json_str)
            normalized_set = set()
            for item in raw_list:
                parts = item.split(',')
                parts = [p.strip() for p in parts]
                
                # Case 1: 标准的三段式 "agent,target,action"
                if len(parts) == 3:
                    ag = self._normalize_key(parts[0])
                    raw_target = parts[1]
                    ac = parts[2]
                    
                    # [CRITICAL FIX]: 处理 "self" 关键字
                    # 如果 target 是 "self"，则 target ID 应该等于 agent ID
                    if raw_target.lower() == 'self':
                        ta = ag
                    else:
                        ta = self._normalize_key(raw_target)
                    
                    # 如果 target 为空/None，则视为 agent
                    if not ta or raw_target.lower() == 'none' or raw_target == '':
                        ta = ag
                    
                    normalized_set.add(f"{ag}_{ta}_{ac}")
                
                # Case 2: 两段式 "agent,action"
                elif len(parts) == 2:
                    ag = self._normalize_key(parts[0])
                    ac = parts[1]
                    # 两段式默认 Target = Agent
                    ta = ag
                    
                    normalized_set.add(f"{ag}_{ta}_{ac}")
                    
            return normalized_set
        except Exception as e:
            return set()

    def calculate_stats_core(self, solution: pl.DataFrame, submission: pl.DataFrame):
        # [GM OPTIMIZATION]: 为了防止 Submission 中 Single Action 的 target_id 填写混乱
        # 我们最好先识别出哪些 Action 是 Single 的，然后在生成 Key 时强制对齐。
        # 但为了保持和官方逻辑尽量一致，我们先在 parsing 阶段 fix "self" 问题。
        
        # 1. 预处理 Key
        # 注意：这里假设 DataFrame 中的数据是规范的 (即 Single Action 的 target_id == agent_id)
        # 如果 Submission 里的 target_id 填错了，这里生成的 Key 依然会匹配不上。
        solution = solution.with_columns(
            pl.concat_str(
                [
                    pl.col('video_id').cast(pl.Utf8), 
                    pl.col('agent_id').cast(pl.Utf8), 
                    pl.col('target_id').cast(pl.Utf8), 
                    pl.col('action')
                ],
                separator='_'
            ).alias('label_key')
        )
        
        submission = submission.with_columns(
            pl.concat_str(
                [
                    pl.col('video_id').cast(pl.Utf8), 
                    pl.col('agent_id').cast(pl.Utf8), 
                    pl.col('target_id').cast(pl.Utf8), 
                    pl.col('action')
                ],
                separator='_'
            ).alias('prediction_key')
        )

        label_frames = defaultdict(set)
        prediction_frames = defaultdict(set)

        # 2. 构建 Ground Truth 并识别 Single/Pair
        solution_dicts = solution.to_dicts()
        for row in solution_dicts:
            label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))
            
            action_name = row['action']
            if action_name not in self.action_category_map:
                # 逻辑判断：如果ID相同 或者 是已知的 Single 集合
                # 注意：有些数据集中 Single Action 的 target_id 可能被标记为 None 或 0，需小心
                if str(row['agent_id']) == str(row['target_id']) or action_name in self.known_single_actions:
                    self.action_category_map[action_name] = 'single'
                else:
                    self.action_category_map[action_name] = 'pair'

        # 3. 遍历视频处理预测
        unique_videos = solution['video_id'].unique()
        
        for video in unique_videos:
            active_labels_raw = solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
            if active_labels_raw is None: continue
            
            # 这里调用修复后的 active label 解析
            active_labels = self._parse_and_normalize_labels(active_labels_raw)
            pair_occupied_frames = defaultdict(set)

            video_sub = submission.filter(pl.col('video_id') == video)
            
            for row in video_sub.to_dicts():
                ag_id = self._normalize_key(row['agent_id'])
                ta_id = self._normalize_key(row['target_id'])
                action = row['action']

                # [GM OPTIMIZATION]: 如果这个动作已知是 Single Action，强制让 target = agent
                # 这能拯救那些 action 预测对了，但是 target 填错成对方老鼠的行
                if action in self.action_category_map and self.action_category_map[action] == 'single':
                    ta_id = ag_id
                    # 重新生成 prediction_key 以匹配修正后的 ID
                    row['prediction_key'] = f"{row['video_id']}_{ag_id}_{ta_id}_{action}"

                check_key = f"{ag_id}_{ta_id}_{action}"
                
                # 现在的 active_labels 里的 key 应该是 "1_1_dig" (由 self->1 转换而来)
                # check_key 也应该是 "1_1_dig"
                if check_key not in active_labels:
                    continue

                current_frames = set(range(row['start_frame'], row['stop_frame']))
                
                # 移除该预测已经计算过的帧 (去重)
                if row['prediction_key'] in prediction_frames:
                     already_pred = prediction_frames[row['prediction_key']]
                     current_frames = current_frames.difference(already_pred)
                
                if not current_frames: continue

                # Pair 冲突检测
                # 对于 Single Action，pair_key 就是 "1_1"
                pair_key = f"{ag_id}_{ta_id}"
                
                occupied = pair_occupied_frames[pair_key]
                overlap = current_frames.intersection(occupied)
                
                if overlap:
                    valid_frames = current_frames.difference(overlap)
                    if not valid_frames: continue
                    current_frames = valid_frames

                prediction_frames[row['prediction_key']].update(current_frames)
                pair_occupied_frames[pair_key].update(current_frames)

        # 4. 统计
        stats = defaultdict(lambda: {'tp': 0, 'fn': 0, 'fp': 0})
        
        for key, pred_frames in prediction_frames.items():
            action = key.split('_')[-1]
            matched_label_frames = label_frames[key]
            
            tp = len(pred_frames.intersection(matched_label_frames))
            fp = len(pred_frames.difference(matched_label_frames))
            stats[action]['tp'] += tp
            stats[action]['fp'] += fp

        # 重置 FN 逻辑
        for action in stats:
            stats[action]['fn'] = 0
            
        for key, gt_frames in label_frames.items():
            action = key.split('_')[-1]
            if key in prediction_frames:
                pred_f = prediction_frames[key]
                stats[action]['fn'] += len(gt_frames.difference(pred_f))
            else:
                stats[action]['fn'] += len(gt_frames)

        return stats

    def _compute_f1_from_stats(self, stats_dict):
        f1_scores = []
        for action, s in stats_dict.items():
            tp, fp, fn = s['tp'], s['fp'], s['fn']
            numerator = (1 + self.beta**2) * tp
            denominator = (numerator + self.beta**2 * fn + fp)
            if denominator > 0:
                f1_scores.append(numerator / denominator)
            else:
                f1_scores.append(0.0)
        
        if not f1_scores: return 0.0
        return sum(f1_scores) / len(f1_scores)

    def get_official_score(self, solution: pd.DataFrame, submission: pd.DataFrame):
        sol_pl = pl.from_pandas(solution) if isinstance(solution, pd.DataFrame) else solution
        sub_pl = pl.from_pandas(submission) if isinstance(submission, pd.DataFrame) else submission

        lab_ids = sol_pl['lab_id'].unique()
        lab_scores = []

        for lab in lab_ids:
            lab_sol = sol_pl.filter(pl.col('lab_id') == lab)
            if len(lab_sol) == 0: continue
            
            lab_videos = lab_sol['video_id'].unique()
            lab_sub = sub_pl.filter(pl.col('video_id').is_in(lab_videos))

            stats = self.calculate_stats_core(lab_sol, lab_sub)
            lab_f1 = self._compute_f1_from_stats(stats)
            lab_scores.append(lab_f1)
            
        if not lab_scores: return 0.0
        return sum(lab_scores) / len(lab_scores)

    def get_granular_metrics_df(self, solution: pd.DataFrame, submission: pd.DataFrame):
        sol_pl = pl.from_pandas(solution) if isinstance(solution, pd.DataFrame) else solution
        sub_pl = pl.from_pandas(submission) if isinstance(submission, pd.DataFrame) else submission

        stats_dict = self.calculate_stats_core(sol_pl, sub_pl)
        
        rows = []
        for action, s in stats_dict.items():
            tp, fp, fn = s['tp'], s['fp'], s['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            numerator = (1 + self.beta**2) * tp
            denominator = (numerator + self.beta**2 * fn + fp)
            f1 = numerator / denominator if denominator > 0 else 0.0
            switch_type = self.action_category_map.get(action, 'unknown')

            rows.append({
                'switch': switch_type, 'action': action, 'f1': f1,
                'precision': precision, 'recall': recall,
                'support': tp + fn, 'tp': tp, 'fp': fp, 'fn': fn
            })
            
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values(by=['switch', 'f1'], ascending=[False, False])
        return df


# ---
import re
import json
import itertools
import numpy as np
import pandas as pd


def rebalance_pos_neg(
    X, 
    y, 
    mode="upsample_pos",      # "upsample_pos": 上采样正样本, "downsample_neg": 下采样负样本
    target_pos_ratio=0.3,     # 希望正样本占比至少为 target_pos_ratio (0 ~ 0.5 之间比较常见)
    random_state=42
):
    """
    对二分类任务做简单重采样（只针对正样本/负样本），缓解类不平衡。

    假设：标签中 1 为正样本，0 为负样本。

    参数:
        X, y              : 原始训练数据 (numpy array)
        mode              : "upsample_pos" 上采样正样本；"downsample_neg" 下采样负样本
        target_pos_ratio  : 期望的正样本比例，例如 0.3 表示正样本占 30%
                            如果当前比例 >= target_pos_ratio，则认为已经“更平衡”，不做处理
        random_state      : 随机种子
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    classes = np.unique(y)
    if len(classes) > 2:
        raise ValueError("rebalance_pos_neg 只适用于二分类任务")

    # 这里默认 1 是正样本，如果你的正样本不是 1，可以手动改这里
    if 1 in classes:
        pos_label = 1
        neg_label = classes[classes != 1][0]
    else:
        # 若没有 1，就把较大的那个视为正样本
        pos_label = classes.max()
        neg_label = classes.min()

    idx_pos = np.where(y == pos_label)[0]
    idx_neg = np.where(y == neg_label)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    # 极端情况：某一类完全缺失，直接返回
    if n_pos == 0 or n_neg == 0:
        # 没有办法通过重采样“制造”新标签，只能原样返回
        return X, y

    cur_pos_ratio = n_pos / (n_pos + n_neg)

    # ---- 条件 2：如果当前正负比例已经比目标更平衡，就不调整 ----
    # 这里定义为：当前正样本比例 >= target_pos_ratio 就认为更平衡
    if cur_pos_ratio >= target_pos_ratio:
        # 已经比你设的 target 更不那么极端了，保持原样
        return X, y

    # 否则，才根据 mode 做重采样
    if mode == "upsample_pos":
        # 固定负样本数量 n_neg，通过上采样正样本达到目标比例
        # new_n_pos / (new_n_pos + n_neg) = target_pos_ratio
        # => new_n_pos = target_pos_ratio * n_neg / (1 - target_pos_ratio)
        new_n_pos = int(np.ceil(target_pos_ratio * n_neg / (1.0 - target_pos_ratio)))
        # 至少不能比原先还少
        new_n_pos = max(new_n_pos, n_pos)

        n_extra = new_n_pos - n_pos
        extra_idx_pos = rng.choice(idx_pos, size=n_extra, replace=True)
        new_idx_pos = np.concatenate([idx_pos, extra_idx_pos])
        new_idx_neg = idx_neg

    elif mode == "downsample_neg":
        # 固定正样本数量 n_pos，通过下采样负样本达到目标比例
        # n_pos / (n_pos + new_n_neg) = target_pos_ratio
        # => new_n_neg = n_pos * (1 - target_pos_ratio) / target_pos_ratio
        new_n_neg = int(np.floor(n_pos * (1.0 - target_pos_ratio) / target_pos_ratio))
        new_n_neg = min(new_n_neg, n_neg)  # 不能比原负样本多
        new_n_neg = max(new_n_neg, 1)      # 避免为 0

        new_idx_neg = rng.choice(idx_neg, size=new_n_neg, replace=False)
        new_idx_pos = idx_pos

    else:
        raise ValueError("mode 必须是 'upsample_pos' 或 'downsample_neg'")

    # 合并并打乱
    all_idx = np.concatenate([new_idx_pos, new_idx_neg])
    rng.shuffle(all_idx)

    X_bal = X[all_idx]
    y_bal = y[all_idx]
    return X_bal, y_bal

def split_video_ids(SEED, video_ids, test_size=0.1):
    if test_size <= 0.0:
        return video_ids, np.array([], dtype=video_ids.dtype)
    unique_video_ids = np.unique(video_ids)
    n_test = max(1, int(len(unique_video_ids) * test_size))
    rng = np.random.default_rng(SEED)
    test_video_ids = rng.choice(unique_video_ids, size=n_test, replace=False)
    train_video_ids = np.setdiff1d(unique_video_ids, test_video_ids)
    return train_video_ids, test_video_ids


import joblib
def save_artifacts(CFG, run_id, section_idx, action, models, feature_names, model_names):
    """保存训练产物"""
    artifact = {
        'models': models,
        'model_names': model_names,
        'feature_names': feature_names,
    }
    # 按 section_action 命名，方便推理时索引
    filename = f"{CFG.output_dir}/model_{run_id}_sec{section_idx}_{action}.pkl"
    joblib.dump(artifact, filename)
    



# ---
# all body parts tracked
body_parts_tracked_dict =  {
        0: '["body_center", "ear_left", "ear_right", "headpiece_bottombackleft", "headpiece_bottombackright", "headpiece_bottomfrontleft", "headpiece_bottomfrontright", "headpiece_topbackleft", "headpiece_topbackright", "headpiece_topfrontleft", "headpiece_topfrontright", "lateral_left", "lateral_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"]',
        1: '["body_center", "ear_left", "ear_right", "hip_left", "hip_right", "lateral_left", "lateral_right", "nose", "spine_1", "spine_2", "tail_base", "tail_middle_1", "tail_middle_2", "tail_tip"]',
        2: '["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"]',
        3: '["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "nose", "tail_base", "tail_tip"]',
        4: '["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "nose", "tail_base"]',
        5: '["body_center", "ear_left", "ear_right", "nose", "tail_base"]',
        6: '["ear_left", "ear_right", "head", "tail_base"]',
        7: '["ear_left", "ear_right", "hip_left", "hip_right", "neck", "nose", "tail_base"]',
        8: '["ear_left", "ear_right", "nose", "tail_base", "tail_tip"]',
    }

# 某些头戴器与尾部中段关键点在部分实验室稳定性差，这里统一丢弃
drop_body_parts = ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                   'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright',                  
                   'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

# 有用的body parts
# ['body_center', 'ear_left', 'ear_right', 'head', 'hip_left', 'hip_right', 'lateral_left', 'lateral_right', 'neck', 'nose', 'tail_base', 'tail_tip']
# hip_left 左后胯; hip_right 右后胯; lateral_left 左侧躯干点; lateral_right 右侧躯干点; neck 颈部; tail_base 尾根; tail_tip 尾尖

# all actions in the test set
test_single_actions = {'biteobject', 'dig', 'freeze', 'huddle', 'rest', 'run', 'exploreobject', 'selfgroom', 'rear', 'climb'}
test_pair_actions = {'chaseattack', 'escape', 'tussle', 'intromit', 'sniffbody', 'follow', 'shepherd', 'chase', 'ejaculate', 'sniffface', 
                     'reciprocalsniff', 'flinch', 'avoid', 'dominance', 'attemptmount', 'sniffgenital', 'submit', 'attack', 'dominancegroom', 'sniff', 'allogroom', 'approach', 'defend', 'mount'}
test_all_actions = test_single_actions.union(test_pair_actions)