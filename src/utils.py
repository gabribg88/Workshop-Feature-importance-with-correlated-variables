import os
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, \
                            average_precision_score,precision_score, recall_score, f1_score


def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def create_folds(train, features, target, num_folds, num_repeats=None, shuffle=True, seed=42):
    folds = []
    if num_repeats is None:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=seed)
        for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(skf.split(train[features], train[target])):
            folds.append((train_fold_idx, valid_fold_idx))
    else:
        rskf = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=seed)
        for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(rskf.split(train[features], train[target])):
            folds.append((train_fold_idx, valid_fold_idx))
    return folds

