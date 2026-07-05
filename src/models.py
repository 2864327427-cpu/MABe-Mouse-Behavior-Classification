import pandas as pd
import numpy as np
from typing import Optional

from sklearn.base import ClassifierMixin, BaseEstimator, clone

import xgboost as xgb
from xgboost import XGBClassifier


class StratifiedSubsetClassifierWEval(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, es_rounds: "int|str" = "auto", es_metric: str = "auto"):
        self.estimator = estimator # 底层的基础模型（如 XGBClassifier, CatBoostClassifier）
        self.es_rounds = es_rounds # 早停（Early Stopping）的耐心轮数 (Patience)。'auto' 表示自动计算。
        self.es_metric = es_metric # 用于早停监控的评估指标。'auto' 表示根据正样本比例自动选择。
 
    # -------------------------- API --------------------------
    def fit(self, X_train, y_train, X_valid, y_valid):
        # 1. 计算正样本比例 (Positive Rate)
        pos_rate = None
        if y_valid is not None and len(y_valid) > 0:
            pos_rate = float(np.mean(y_valid == 1))

        # 2. 自动选择评估指标与早停耐心值
        # 极度不平衡时使用 PRAUC，否则使用 Logloss
        metric = self._choose_metric(pos_rate)

        # XGBoost：设置 scale_pos_weight = 负样本数 / 正样本数
        n_pos = max(1, int((y_train == 1).sum()))
        n_neg = max(1, len(y_train) - n_pos)
        self.estimator.set_params(scale_pos_weight=(n_neg / n_pos))
        self.estimator.set_params(eval_metric=metric)

        # 4. 执行训练
        self.estimator.fit(X_train, y_train)
        
        # 保存类别标签，兼容 sklearn 接口
        self.classes_ = getattr(self.estimator, "classes_", np.array([0, 1]))
        # 保存中间变量供调试
        self._pos_rate_ = pos_rate
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.estimator.predict_proba(X)

    def predict(self, X: pd.DataFrame):
        return self.estimator.predict(X)

    # -------------------------- helpers --------------------------
    def _choose_metric(self, pos_rate=0.01) -> str:
        if self.es_metric != "auto":
            return self.es_metric
        # 异常情况或无正样本，回退到 Logloss
        if pos_rate is None or pos_rate == 0.0 or pos_rate == 1.0:
            return "logloss"

        return "aucpr"

def build_all_models(seed, use_gpu, use_models_list):
    models = []

    eval_kwargs = {
        "es_rounds": "auto",  # 自动计算 Early Stopping rounds
        "es_metric": "auto"   # 自动选择 PRAUC
    }

    if 'xgb3' in use_models_list:
        xgb3 = XGBClassifier(
            # raw n_estimators=2000, learning_rate=0.05
            random_state=seed, booster="gbtree", tree_method="gpu_hist",
            n_estimators=3000, learning_rate=0.05, grow_policy="lossguide",
            max_leaves=255, max_depth=0, min_child_weight=10, gamma=0.0,
            subsample=0.90, colsample_bytree=1.00, colsample_bylevel=0.85,
            reg_alpha=0.0, reg_lambda=1.0, max_bin=256,
            single_precision_histogram=True, verbosity=0,
        )
        models.append(StratifiedSubsetClassifierWEval(xgb3, **eval_kwargs))

    if 'xgb1' in use_models_list:
        # 模型 6: GPU XGBoost Large
        # 原配置: n_samples / 2.0
        xgb1 = XGBClassifier(
            random_state=seed, booster="gbtree", tree_method="gpu_hist",
            n_estimators=2000, learning_rate=0.05, grow_policy="lossguide",
            max_leaves=255, max_depth=0, min_child_weight=10, gamma=0.0,
            subsample=0.90, colsample_bytree=1.00, colsample_bylevel=0.85,
            reg_alpha=0.0, reg_lambda=1.0, max_bin=256,
            single_precision_histogram=True, verbosity=0,
        )
        models.append(StratifiedSubsetClassifierWEval(xgb1, **eval_kwargs))
    
    return models
