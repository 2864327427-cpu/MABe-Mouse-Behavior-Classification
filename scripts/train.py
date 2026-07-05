import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
import itertools
import warnings
import json
import os, random
import gc, re, math
import lightgbm
from collections import defaultdict
import joblib
import datetime
import polars as pl
from time import perf_counter 
from sklearn.base import clone
import argparse

import xgboost
import lightgbm
import catboost

from utils_mabe2025 import score, body_parts_tracked_dict, drop_body_parts, test_single_actions, test_pair_actions, test_all_actions
from utils_mabe2025 import save_artifacts, rebalance_pos_neg, split_video_ids
from utils import IS_KAGGLE_ENV, get_time_suffix, init_logger, get_timediff
from data_processing import generate_mouse_data, _fps_from_meta, _ppm_from_meta, _arena_dim_from_meta, _arena_dim_from_meta, _arena_shape_from_meta
from feature_engineering_single import transform_single_v52
from feature_engineering_pair import transform_pair_v213
from models import StratifiedSubsetClassifierWEval, build_all_models

pd.set_option('display.max_columns', None)      # 不限制列数
pd.set_option('display.expand_frame_repr', False)  # 不再自动换块打印
warnings.filterwarnings('ignore')
USE_GPU = True
print(f'Using GPU: {USE_GPU}')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--single_transform_version', type=int, default=52, help='single transform version')
parser.add_argument('--pair_transform_version', type=int, default=0, help='pair transform version')
parser.add_argument('--gpu', type=str, default='1', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--model_name', type=str, default='xgb1', help='model type: lgbm/xgb/cat')
parser.add_argument('--target_pos_ratio', type=float, default=0.01, help='target positive ratio after rebalancing')

args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
start_time = time.time()
suffix = get_time_suffix()
SEED = args.seed

class CFG:
    input_dir = "/kaggle/input" if IS_KAGGLE_ENV else "./input/"
    comp_dir = "/kaggle/input/MABe-mouse-behavior-detection" if IS_KAGGLE_ENV else f"{input_dir}/mabe-mouse-behavior-detection"
    output_dir = f"/kaggle/working/output/lb449_{suffix}" if IS_KAGGLE_ENV else f"./output/lb449_{suffix}"
    cache_dir = f"{output_dir}/lb449_cache" if IS_KAGGLE_ENV else f"./input/lb449_cache"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    use_generated_data_cache = True
    use_single_X_cache = True
    use_pair_X_cache = True

    test_size = 0.1
    split_method = 'video_id'  # 'video_id' or 'random'
    use_models_list = [args.model_name] # ['lgbm_225', 'lgbm_150', 'lgbm_100', 'xgb_180', 'cat_120', 'xgb1', 'xgb2', 'cat_bay']
    train_samples = None # 1_500_000, 750_000, None
    
    rebalance_mode = 'downsample_neg'  # 'upsample_pos', 'downsample_neg', or None
    target_pos_ratio = args.target_pos_ratio # 目标正样本比例，仅在 rebalance_mode 不为 None 时生效


if args.single_transform_version == 52:
    transform_single = transform_single_v52

if args.pair_transform_version == 213:
    transform_pair = transform_pair_v213


# --- SEED EVERYTHING -----
os.environ["PYTHONHASHSEED"] = str(SEED)
rnd = np.random.RandomState(SEED)
random.seed(SEED)
np.random.seed(SEED)

LOGGER = init_logger(log_file=f"{CFG.output_dir}/train.log")
LOGGER.info(f"xgboost version: {xgboost.__version__}") # 2.0.3
LOGGER.info(f"lightgbm version: {lightgbm.__version__}") # 4.5.0
LOGGER.info(f"catboost version: {catboost.__version__}") # 1.2.8

# 将CFG打印到日志 
LOGGER.info("Configuration:")
for attr in dir(CFG):
    if not attr.startswith("__") and not callable(getattr(CFG, attr)):
        LOGGER.info(f"  {attr}: {getattr(CFG, attr)}")
LOGGER.info(f"Single Transform Version: v{args.single_transform_version}")
LOGGER.info(f"Pair Transform Version: v{args.pair_transform_version}")
LOGGER.info("\n\n")

# %%
def train_func(section_idx, switch_tr, X_tr, label, meta, train_video_ids, valid_video_ids, train_samples=None):
    # 创建模型
    models = build_all_models(seed=SEED, use_gpu=USE_GPU, use_models_list=CFG.use_models_list)

    # 记录特征列名
    feature_names = list(X_tr.columns)

    # --- 训练循环 ---
    action_count = len(label.columns)
    for a_id, action in enumerate(label.columns):
        LOGGER.info(f"\n>> Section: {section_idx} | {switch_tr} | Action: {action} ({a_id+1}/{action_count}) <<")

        if action not in test_all_actions:
            LOGGER.info(f"Skipping action {action} (not in test set)")
            continue

        # 获取该动作非空值的掩码
        action_mask = ~label[action].isna().values

        X_action = X_tr[action_mask].copy()
        y_action = label[action][action_mask].copy().values.astype(int)
        meta_action = meta[action_mask].copy()
        video_ids = meta_action['video_id'].values
        
        if len(np.unique(y_action)) < 2:
            LOGGER.info(f"[1 Class Warning] Skipping action: {action} , y_action unique: {np.unique(y_action)}")
            continue
        
        if CFG.split_method == 'video_id':
            # 根据 video_ids 划分训练与验证索引
            train_idx_mask = np.isin(video_ids, train_video_ids)
            valid_idx_mask = np.isin(video_ids, valid_video_ids)
        elif CFG.split_method == 'random':
            # 随机划分训练与验证索引
            rng = np.random.default_rng(SEED)
            permuted_indices = rng.permutation(len(y_action))
            n_valid = int(len(y_action) * CFG.test_size)
            valid_idx_mask = np.zeros(len(y_action), dtype=bool)
            valid_idx_mask[permuted_indices[:n_valid]] = True
            train_idx_mask = ~valid_idx_mask

        train_idx = np.where(train_idx_mask)[0]
        valid_idx = np.where(valid_idx_mask)[0]

        if CFG.split_method == 'random':
            # save train_idx and valid_idx for random split
            with open(f"{CFG.output_dir}/sec{section_idx}_{action}_train_valid_idx.pkl", "wb") as f:
                pickle.dump((train_idx, valid_idx), f)
                
        if train_samples is not None and len(train_idx) > train_samples:
            # 如果训练样本过多，进行下采样以加快训练速度
            rng = np.random.default_rng(SEED)
            train_idx = rng.choice(train_idx, size=train_samples, replace=False)

        LOGGER.info(f"Data Split - Train: {len(train_idx)}, Valid: {len(valid_idx)}, Valid Ratio: {len(valid_idx)/(len(train_idx)+len(valid_idx)):.1%}")
        LOGGER.info(f"Positive: {np.sum(y_action)}, Negative: {len(y_action) - np.sum(y_action)}, Positive Ratio: {np.sum(y_action)/len(y_action):.1%}")

        # 准备训练与验证数据
        X_train = X_action.iloc[train_idx].to_numpy(np.float32, copy=False)
        y_train = np.asarray(y_action[train_idx])

        # 进行样本重采样以平衡正负样本比例
        if CFG.rebalance_mode is not None:
            X_train, y_train = rebalance_pos_neg(
                X_train, y_train,
                mode=CFG.rebalance_mode,
                target_pos_ratio=CFG.target_pos_ratio,
                random_state=SEED,
            )
            LOGGER.info(f"After rebalancing - Train samples: {len(y_train)}, Positive: {np.sum(y_train)}, Negative: {len(y_train) - np.sum(y_train)}, Positive Ratio: {np.sum(y_train)/len(y_train):.1%}")


        if valid_idx is not None and len(valid_idx) > 0:
            X_valid = X_action.iloc[valid_idx].to_numpy(np.float32, copy=False)
            y_valid = np.asarray(y_action[valid_idx])
        else:
            X_valid = y_valid = None

        LOGGER.info(f"Feature NUM: {X_train.shape[1]}")
        
        # 训练所有模型
        trained_models_for_action = []
        trained_model_names_for_action = []
        for model_idx, mdl in enumerate(models):
            model_name = CFG.use_models_list[model_idx]
            m_clone = clone(mdl)
            t0 = perf_counter()
            m_clone.fit(X_train, y_train, X_valid, y_valid)
            
            LOGGER.info(f"Sec{section_idx} | {model_idx}.{model_name:<12} | {switch_tr} | Action={action} | {perf_counter()-t0:.1f}s")
            trained_models_for_action.append(m_clone)
            trained_model_names_for_action.append(model_name)

        # 保存当前 Action 的所有模型集成
        if trained_models_for_action:
            save_artifacts(
                CFG=CFG,
                run_id=suffix,
                section_idx=section_idx,
                action=action,
                models=trained_models_for_action,
                feature_names=feature_names,
                model_names=trained_model_names_for_action,
            )

    # 训练完成，清理内存
    del X_tr; gc.collect()

# %%
train = pd.read_csv(f'{CFG.comp_dir}/train.csv')

# drop likely-sleeping MABe22 clips: condition == "lights on"
train = train.loc[~(train['lab_id'].astype(str).str.contains('MABe22', na=False) &
                    train['mouse1_condition'].astype(str).str.lower().eq('lights on'))].copy()

train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)

BAD_IDENTITY_TRAIN_VIDS = {1212811043}  # AdaptableSnail
train = train.loc[~(train['video_id'].isin(BAD_IDENTITY_TRAIN_VIDS))].copy()



test = pd.read_csv(f'{CFG.comp_dir}/test.csv')
test['sleeping'] = (
    test['lab_id'].astype(str).str.contains('MABe22', na=False) &
    test['mouse1_condition'].astype(str).str.lower().eq('lights on')
)
test['n_mice'] = 4 - test[['mouse1_strain','mouse2_strain','mouse3_strain','mouse4_strain']].isna().sum(axis=1)


for section, body_parts_tracked_str in body_parts_tracked_dict.items():
    body_parts_tracked = json.loads(body_parts_tracked_str)
    LOGGER.info(f"\nSection: {section}. Raw Body Parts {len(body_parts_tracked)}: {body_parts_tracked}")

    # 统一删除不稳定关键点
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
        LOGGER.info(f"After dropping unstable parts: {len(body_parts_tracked)}: {body_parts_tracked}")
    
    # 筛选出当前身体部位配置的行
    train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
    LOGGER.info(f"Train subset size: {len(train_subset)} videos")

    # 划分train/valid video_ids, 并保存
    train_video_ids, valid_video_ids = None, None
    if CFG.split_method == 'video_id':
        train_video_ids, valid_video_ids = split_video_ids(SEED, train_subset['video_id'].values, test_size=CFG.test_size)
        LOGGER.info(f"Train video IDs: {len(train_video_ids)}, Valid video IDs: {len(valid_video_ids)}")
        with open(f"{CFG.output_dir}/section_{section}_train_valid_video_ids.pkl", "wb") as f:
            pickle.dump((train_video_ids, valid_video_ids), f)
    elif CFG.split_method == 'random':
        pass

    # 构建 FPS 查找表
    _fps_lookup = (
        train_subset[['video_id', 'frames_per_second']]
        .drop_duplicates('video_id')
        .set_index('video_id')['frames_per_second'].to_dict()
    ) # {video_id: fps}

    # === 新增：构建 PPM 查找表 ===
    # 优先使用 'pix per cm (approx)'，如果不存在则找 'pix_per_cm'
    _ppm_lookup = (
        train_subset[['video_id', 'pix_per_cm_approx']]
        .drop_duplicates('video_id')
        .set_index('video_id')['pix_per_cm_approx'].to_dict()
    )
    # === Arena lookup ===
    _arw_lookup = (
        train_subset[['video_id', 'arena_width_cm']]
        .drop_duplicates('video_id')
        .set_index('video_id')['arena_width_cm'].to_dict()
    )
    _arh_lookup = (
        train_subset[['video_id', 'arena_height_cm']]
        .drop_duplicates('video_id')
        .set_index('video_id')['arena_height_cm'].to_dict()
    )
    _arshape_lookup = (
        train_subset[['video_id', 'arena_shape']]
        .drop_duplicates('video_id')
        .set_index('video_id')['arena_shape'].to_dict()
    )

    # 默认值（用该 section 的中位数更稳）
    _default_aw = float(train_subset['arena_width_cm'].median()) if 'arena_width_cm' in train_subset.columns else 40.0
    _default_ah = float(train_subset['arena_height_cm'].median()) if 'arena_height_cm' in train_subset.columns else 40.0


    # 获取 定位数据/元数据/标签
    if CFG.use_generated_data_cache and os.path.exists(f"{CFG.cache_dir}/section_{section}_generated_data_cache.pkl"):
        LOGGER.info(f"Loading cached generated data for section {section}...")
        with open(f"{CFG.cache_dir}/section_{section}_generated_data_cache.pkl", "rb") as f:
            (single_list, single_label_list, single_meta_list, pair_list, pair_label_list, pair_meta_list) = pickle.load(f)
    else:
        LOGGER.info(f"Generating data for section {section}...")
        # 初始化列表，用于收集各视频生成的“时序数据 + 标签/元数据”
        single_list, single_label_list, single_meta_list = [], [], []
        pair_list, pair_label_list, pair_meta_list = [], [], []
        # 遍历子集中的每个视频，生成特征数据
        for switch, data, meta, label in generate_mouse_data(CFG, train_subset, 'train', drop_body_parts):
            if switch == 'single':
                # 自行为（self）分支收集
                single_list.append(data) # 特征原始数据: index=frame, columns=[[body_part],x,y]
                single_meta_list.append(meta) # 元数据: index=frame, 含video_id, animal_id等
                single_label_list.append(label) # 标签: index=frame, columns=[action1, action2, ...] (布尔值)
            else:
                # 双鼠交互（pair）分支收集
                pair_list.append(data) # 特征原始数据: columns=[('A',[body_part]),('B',[body_part]),x,y]
                pair_meta_list.append(meta) # 元数据: 同上
                pair_label_list.append(label) # 标签: 同上
        # save
        LOGGER.info(f"Caching generated data for section {section}...")
        with open(f"{CFG.cache_dir}/section_{section}_generated_data_cache.pkl", "wb") as f:
            pickle.dump((single_list, single_label_list, single_meta_list, pair_list, pair_label_list, pair_meta_list), f)

    LOGGER.info(f"Generated Single data: {len(single_list)} [video,agent,target] pair")
    LOGGER.info(f"Generated Pair data: {len(pair_list)} [video,agent,target] pair")

    # ===================== 单鼠（self）分支处理 =====================
    if len(single_list) > 0:
        single_feats_parts = []
        if CFG.use_single_X_cache and os.path.exists(f"{CFG.cache_dir}/section_{section}_single_X_cache.pq"):
            LOGGER.info(f"Loading cached single X for section {section}...")
            X_tr = pd.read_parquet(f"{CFG.cache_dir}/section_{section}_single_X_cache.pq")
        else:
            LOGGER.info(f"Generating single X for section {section}...")
            # 逐个[video,agent,target] single 特征工程
            for data_i, meta_i in zip(single_list, single_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                ppm_i = _ppm_from_meta(meta_i, _ppm_lookup, default_ppm=12.4)
                arena_w_i = _arena_dim_from_meta(meta_i, _arw_lookup, 'arena_width_cm', _default_aw)
                arena_h_i = _arena_dim_from_meta(meta_i, _arh_lookup, 'arena_height_cm', _default_ah)
                arena_shape_i = _arena_shape_from_meta(meta_i, _arshape_lookup, default_shape="square")

                Xi = transform_single(
                    data_i, 
                    body_parts_tracked, 
                    fps_i, 
                    pix_per_cm=ppm_i, 
                    arena_width_cm=arena_w_i,
                    arena_height_cm=arena_h_i,
                    arena_shape=arena_shape_i,
                    add_key_lags=True, 
                    key_lags_base=(5, 10), 
                    key_lag_features=["nt_dist", "elong", "sp_m5"]
                    ).astype(np.float32) 
                single_feats_parts.append(Xi)
            X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True) # 合并
            X_tr.to_parquet(f"{CFG.cache_dir}/section_{section}_single_X_cache.pq") # save
            del single_feats_parts

        single_meta  = pd.concat(single_meta_list,  axis=0, ignore_index=True)
        single_label = pd.concat(single_label_list, axis=0, ignore_index=True)
        
        del single_list, single_meta_list, single_label_list
        gc.collect()

        LOGGER.info(f"\nSection: {section}, Single X.shape: {X_tr.shape}")
        LOGGER.info(f"Single Actions: {list(single_label.columns)}")

        train_func(section, 'single', X_tr, single_label, single_meta, train_video_ids, valid_video_ids, train_samples=CFG.train_samples)

        del X_tr, single_label, single_meta
        gc.collect()

    # ===================== 双鼠（pair）分支处理 =====================
    if len(pair_list) > 0:
        pair_feats_parts = []
        if CFG.use_pair_X_cache and os.path.exists(f"{CFG.cache_dir}/section_{section}_pair_X_cache.pq"):
            LOGGER.info(f"Loading cached pair X for section {section}...")
            X_tr = pd.read_parquet(f"{CFG.cache_dir}/section_{section}_pair_X_cache.pq")
        else:
            LOGGER.info(f"Generating pair X for section {section}...")
            # 逐个[video,agent,target] pair 特征工程
            for data_i, meta_i in zip(pair_list, pair_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                ppm_i = _ppm_from_meta(meta_i, _ppm_lookup, default_ppm=12.4)
                arena_w_i = _arena_dim_from_meta(meta_i, _arw_lookup, 'arena_width_cm', _default_aw)
                arena_h_i = _arena_dim_from_meta(meta_i, _arh_lookup, 'arena_height_cm', _default_ah)
                arena_shape_i = _arena_shape_from_meta(meta_i, _arshape_lookup, default_shape="square")
                Xi = transform_pair(
                    data_i, 
                    body_parts_tracked, 
                    fps_i, 
                    pix_per_cm=ppm_i,
                    arena_width_cm=arena_w_i,
                    arena_height_cm=arena_h_i,
                    arena_shape=arena_shape_i,
                    ).astype(np.float32)
                pair_feats_parts.append(Xi)
            X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True) # 合并
            X_tr.to_parquet(f"{CFG.cache_dir}/section_{section}_pair_X_cache.pq") # save
            del pair_feats_parts

        pair_meta  = pd.concat(pair_meta_list,  axis=0, ignore_index=True)
        pair_label = pd.concat(pair_label_list, axis=0, ignore_index=True)

        del pair_list, pair_meta_list, pair_label_list
        gc.collect()

        LOGGER.info(f"\nSection: {section}, Pair X.shape: {X_tr.shape}")
        LOGGER.info(f"Pair Actions: {list(pair_label.columns)}")
        
        train_func(section, 'pair', X_tr, pair_label, pair_meta, train_video_ids, valid_video_ids, train_samples=CFG.train_samples)

        del X_tr, pair_label, pair_meta
        gc.collect()

LOGGER.info(f"\n=== Training completed in {get_timediff(start_time, time.time())} ===")
print(f"SUFFIX={suffix}")

