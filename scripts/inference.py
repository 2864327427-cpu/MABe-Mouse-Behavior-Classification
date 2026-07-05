import os
import sys
sys.path.append("/kaggle/input/mabe-py-files")

IS_KAGGLE_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

verbose = True

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
import joblib # 用于模型保存
import datetime
import polars as pl
from scipy import signal, stats
from typing import Dict, Optional, Tuple
from time import perf_counter 
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from glob import glob

from utils_mabe2025 import score, body_parts_tracked_dict, drop_body_parts
from utils import IS_KAGGLE_ENV, get_time_suffix, init_logger, get_timediff
from data_processing import generate_mouse_data, _fps_from_meta, _ppm_from_meta, _arena_dim_from_meta, _arena_dim_from_meta, _arena_shape_from_meta
from feature_engineering_single import transform_single_v52
from feature_engineering_pair import transform_pair_v213
from models import StratifiedSubsetClassifierWEval, build_all_models
from post_precessing import predict_multiclass_adaptive, robustify

# 使用cuda1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pd.set_option('display.max_columns', None)      # 不限制列数
pd.set_option('display.expand_frame_repr', False)  # 不再自动换块打印

warnings.filterwarnings('ignore')
USE_GPU = True
print(f'Using GPU: {USE_GPU}')

import xgboost
import lightgbm
import argparse

start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--suffix', type=str, default="1213232300", help='suffix for model dir') ####
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--single_transform_version', type=int, default=52, help='single transform version')
parser.add_argument('--pair_transform_version', type=int, default=213, help='pair transform version')
args, _ = parser.parse_known_args()
suffix = args.suffix
SEED = args.seed
RUN_MODE = 'test' # 'test' or 'valid'

print(f"RUN_MODE: {RUN_MODE}, suffix: {suffix}, SEED: {SEED}")

class CFG:
    input_dir = "/kaggle/input" if IS_KAGGLE_ENV else "./input/"
    comp_dir = "/kaggle/input/MABe-mouse-behavior-detection" if IS_KAGGLE_ENV else f"{input_dir}/mabe-mouse-behavior-detection"
    train_dir = f"/kaggle/input/lb449_{suffix}" if IS_KAGGLE_ENV else f"./output/lb449_{suffix}"
    output_dir = f"/kaggle/working/output/lb449_{suffix}_valid" if IS_KAGGLE_ENV else f"./output/lb449_{suffix}_valid"
    cache_dir = f"{output_dir}/lb449_cache" if IS_KAGGLE_ENV else f"./input/lb449_cache"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

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

LOGGER.info(f"Single Transform Version: v{args.single_transform_version}")
LOGGER.info(f"Pair Transform Version: v{args.pair_transform_version}")


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


def inference_func(section, body_parts_tracked_str, switch_tr):
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    model_paths = glob(f"{CFG.train_dir}/model_{suffix}_sec{section}_*.pkl")

    # 现在把 feature_names 也一起读出来，后续按 action 级别对齐特征
    sec_model_list = []
    for model_path in model_paths:
        print(f"  {model_path}")
        action_name = model_path.split("_")[-1].replace(".pkl", "")
        artifact = joblib.load(model_path)
        models = artifact['models']
        feature_names = artifact.get('feature_names', None)  # 兼容旧artifact
        sec_model_list.append((action_name, models, feature_names))

    subset = test[test.body_parts_tracked == body_parts_tracked_str]

    generator = generate_mouse_data(
        CFG, subset, RUN_MODE, drop_body_parts=drop_body_parts,
        generate_single=(switch_tr == 'single'),
        generate_pair=(switch_tr == 'pair')
    )

    fps_lookup = (
        subset[['video_id', 'frames_per_second']]
        .drop_duplicates('video_id')
        .set_index('video_id')['frames_per_second']
        .to_dict()
    )

    ppm_lookup = (
        subset[['video_id', 'pix_per_cm_approx']]
        .drop_duplicates('video_id')
        .set_index('video_id')['pix_per_cm_approx']
        .to_dict()
    )

    # === Arena lookup ===
    _arw_lookup = (
        subset[['video_id', 'arena_width_cm']]
        .drop_duplicates('video_id')
        .set_index('video_id')['arena_width_cm'].to_dict()
    )
    _arh_lookup = (
        subset[['video_id', 'arena_height_cm']]
        .drop_duplicates('video_id')
        .set_index('video_id')['arena_height_cm'].to_dict()
    )
    _arshape_lookup = (
        subset[['video_id', 'arena_shape']]
        .drop_duplicates('video_id')
        .set_index('video_id')['arena_shape'].to_dict()
    )

    # 默认值（用该 section 的中位数更稳）
    _default_aw = float(subset['arena_width_cm'].median()) if 'arena_width_cm' in subset.columns else 40.0
    _default_ah = float(subset['arena_height_cm'].median()) if 'arena_height_cm' in subset.columns else 40.0

    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
        ppm_i = _ppm_from_meta(meta_te, ppm_lookup, default_ppm=12.4)

        arena_w_i = _arena_dim_from_meta(meta_te, _arw_lookup, 'arena_width_cm', _default_aw)
        arena_h_i = _arena_dim_from_meta(meta_te, _arh_lookup, 'arena_height_cm', _default_ah)
        arena_shape_i = _arena_shape_from_meta(meta_te, _arshape_lookup, default_shape="square")
        if switch_te == 'single':
            X_te = transform_single(
                data_te, 
                body_parts_tracked, 
                fps_i, 
                pix_per_cm=ppm_i, 
                arena_width_cm=arena_w_i,
                arena_height_cm=arena_h_i,
                arena_shape=arena_shape_i,
                add_key_lags=True, 
                key_lags_base=(5, 10), 
                key_lag_features=["nt_dist", "elong", "sp_m5"]
                ) 
        else:
            X_te = transform_pair(
                data_te, 
                body_parts_tracked, 
                fps_i, 
                pix_per_cm=ppm_i,
                arena_width_cm=arena_w_i,
                arena_height_cm=arena_h_i,
                arena_shape=arena_shape_i,
                )
        del data_te

        pred = pd.DataFrame(index=meta_te.video_frame)

        # 为了避免每个 action 都重复 reindex 的开销，做一个 cache
        aligned_cache = {}

        for action, trained, feature_names in sec_model_list:
            if action not in actions_te:
                continue
            print(f"use action: {action}")
            # 推理时强制对齐训练期的特征列（列集合 + 顺序）
            if feature_names is not None:
                key = tuple(feature_names)
                X_use = aligned_cache.get(key)
                if X_use is None:
                    # 缺失列补 0，多余列丢弃，并保证顺序一致
                    X_use = X_te.reindex(columns=feature_names, fill_value=0.0).astype(np.float32)
                    aligned_cache[key] = X_use
            else:
                # 兼容旧模型：没有保存 feature_names 就退化为原逻辑
                X_use = X_te

            probs = []
            for mi, mdl in enumerate(trained):
                probs.append(mdl.predict_proba(X_use)[:, 1])

            pred[action] = np.mean(probs, axis=0)

        del X_te
        gc.collect()

        if pred.shape[1] != 0:
            submission_list.append(predict_multiclass_adaptive(pred, meta_te))

# -------

submission_list = []

for section, body_parts_tracked_str in body_parts_tracked_dict.items():
    LOGGER.info(f"Sec{section} single")
    inference_func(section, body_parts_tracked_str, 'single')
    LOGGER.info(f"Sec{section} pair")
    inference_func(section, body_parts_tracked_str, 'pair')

    gc.collect()
    LOGGER.info("")

submission = pd.concat(submission_list, ignore_index=True)

submission_robust = robustify(CFG, submission, test, 'test')
submission_robust.index.name = 'row_id'
submission_robust.to_csv('submission.csv')
LOGGER.info(f"\nSubmission created: {len(submission_robust)} predictions")