import torch
import numpy as np
from torch import nn
import random
import os
from datetime import datetime
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from deepdiff import DeepDiff
from types import SimpleNamespace
from glob import glob
from sklearn.metrics import roc_auc_score
import sys


IS_KAGGLE_ENV = sum(['KAGGLE' in k for k in os.environ]) > 0
IS_KAGGLE_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
sys.path.append('./input/rsna-intracranial-aneurysm-detection')

def sep():
    print("-"*100)

def get_timediff(time1,time2):
    minute_,second_ = divmod(time2-time1,60)
    return f"{int(minute_):02d}:{int(second_):02d}"

def current_date_time():
    # Format the current date and time as "YYYY-MM-DD HH:MM:SS"
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_time_suffix(length=10):
    return datetime.now().strftime("%Y%m%d%H%M%S")[-1*length:]

def init_logger(log_file=f'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.propagate = False
    return logger
