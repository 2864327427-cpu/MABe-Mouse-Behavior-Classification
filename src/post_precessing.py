verbose = True
from collections import defaultdict
import numpy as np
import pandas as pd
import json

def predict_multiclass_adaptive(pred, meta, action_thresholds=defaultdict(lambda: 0.27)):
    """
    自适应多分类预测函数：结合了针对每个动作的自适应阈值和时间平滑处理。
    将逐帧的概率预测转换为提交所需的事件片段（Start-Stop）格式。
    """
    
    # Apply temporal smoothing
    # 1. 时间平滑：使用窗口大小为5的滚动均值平滑预测概率，以减少单帧噪声波动
    pred_smoothed = pred.rolling(window=5, min_periods=1, center=True).mean()
    
    # 获取每一帧预测概率最大的动作索引（Argmax）
    ama = np.argmax(pred_smoothed, axis=1)
    
    # 获取每一帧的最大概率值
    max_probs = pred_smoothed.max(axis=1)
    
    # 2. 自适应阈值过滤
    # 初始化掩码，全为 False
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    # 遍历每个动作类别
    for i, action in enumerate(pred_smoothed.columns):
        # 找到预测为当前动作 i 的所有帧
        action_mask = (ama == i)
        # 获取该动作对应的阈值（默认为 0.27）
        threshold = action_thresholds.get(action, 0.27)
        # 更新掩码：只有当“预测为该动作”且“概率 >= 阈值”时，才标记为有效
        threshold_mask |= (action_mask & (max_probs >= threshold))
    
    # 应用掩码：未通过阈值的帧标记为 -1（即无动作/背景类），通过的保留原动作索引
    ama = np.where(threshold_mask, ama, -1)
    # 转换为 Series 并对齐索引
    ama = pd.Series(ama, index=meta.video_frame)
    
    # 3. 状态变化检测（Run-Length Encoding 逻辑）
    # 检测当前帧动作是否与前一帧不同，用于识别动作的开始和结束边界
    changes_mask = (ama != ama.shift(1)).values
    # 提取发生变化的动作索引和对应的元数据
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    
    # 筛选出有效动作的开始点（值 >= 0，即非 -1 的状态）
    # mask 用于从变化点中选出“动作开始”的那些行
    mask = ama_changes.values >= 0
    # 确保最后一项不作为开始项（防止越界或未闭合）
    mask[-1] = False
    
    # 4. 构建初步的提交 DataFrame
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values], # 将索引映射回动作名称
        'start_frame': ama_changes.index[mask],           # 动作开始帧
        # stop_frame 暂时取“下一次状态变化”的帧索引
        # 注意：这里简单的 shift 逻辑在跨视频或跨个体会时有问题，需要在下面循环中修正
        'stop_frame': ama_changes.index[1:][mask[:-1]]   

        # 'start_frame' like Index([0, 4], dtype='int64')
        # 'stop_frame' like Index([3, 5], dtype='int64')
    })
    
    # 提取用于边界检查的“下一个片段”的元数据信息
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    
    # 5. 修正 Stop Frame (边界处理)
    # 遍历生成的片段，检查是否跨越了视频或动物个体的边界
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        
        # 如果不是最后一行，且检测到下一个变化点属于不同的视频、不同的主体或目标
        if i < len(stop_video_id):
            if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
                # 说明当前片段是该视频/个体的最后一个动作，结束帧应为该视频的最大帧数+1
                new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
        else:
            # 处理最后一行数据，直接设为视频末尾
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
    
    # 6. 过滤短时噪声
    # 计算持续时间，过滤掉持续时间少于 3 帧的极短片段
    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)
    
    # 最后的完整性检查：确保结束帧永远大于开始帧
    if len(submission_part) > 0:
        assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    
    if verbose: print(f'  actions found: {len(submission_part)}')
    return submission_part

def robustify(CFG, submission, dataset, traintest, traintest_directory=None):
    """
    对预测结果进行后处理，确保提交文件的合规性与完整性。
    
    主要功能：
    1. 清洗数据：移除起始帧大于等于结束帧的无效片段。
    2. 解决冲突：去除同一主体对在时间上的重叠预测。
    3. 兜底填充：对无任何预测结果的视频，基于元数据生成占位片段，防止空提交。

    参数:
        submission (pd.DataFrame): 原始预测结果。
        dataset (pd.DataFrame): 视频元数据 (test.csv)。
        traintest (str): 'train' 或 'test'。
        traintest_directory (str, optional): tracking文件路径。
    """
    # 1. 路径配置：若未指定，使用默认的比赛数据路径
    if traintest_directory is None:
        if traintest in ['train', 'valid']:
            traintest_directory = f"{CFG.comp_dir}/train_tracking"
        else:
            traintest_directory = f"{CFG.comp_dir}/test_tracking"

    # 2. 过滤无效区间：确保 start_frame < stop_frame
    old_submission = submission.copy()
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")
        
    # 3. 去除重叠：对同一行为主体对，按时间顺序贪心移除重叠片段，确保时间轴互斥
    #    逻辑：按开始时间排序，如果当前片段的开始时间早于上一个保留片段的结束时间，则视为重叠并丢弃。
    old_submission = submission.copy()
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0  # 记录上一个有效片段的结束帧
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop_frame:
                mask[i] = False  # 发生重叠，标记为丢弃
            else:
                last_stop_frame = row['stop_frame'] # 更新结束帧边界
        group_list.append(group[mask])
        
    submission = pd.concat(group_list)
    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")
    
    # 4. 兜底填充 (Fail-safe)：处理模型未输出任何预测的“僵尸”视频
    #    逻辑：读取视频全长，将所有可能的行为在时间轴上均分填充，确保该视频有提交记录。
    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        # 跳过公开数据集副本 (MABe22)，通常无需参与特定测试
        if lab_id.startswith('MABe22'):
            continue
        
        video_id = row['video_id']
        # 若视频已有预测结果，则跳过
        if (submission.video_id == video_id).any():
            continue
        
        # 若元数据中无行为标注信息，无法生成填充数据，跳过
        if type(row.behaviors_labeled) != str:
            continue

        print(f"Video {video_id} has no predictions.")
        
        # 读取 tracking 文件以获取视频准确的起止帧
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        # 解析该视频可能包含的行为列表 (agent, target, action)
        vid_behaviors = json.loads(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors})) # 去重与清洗
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
    
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1
    
        # 均分填充策略：将视频总时长按行为数量均分，依次填入
        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    # 合并填充数据与原始预测
    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])
        print("ERROR: Filled empty videos")

    # 重置索引，输出最终格式
    submission = submission.reset_index(drop=True)
    
    return submission