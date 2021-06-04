#!/usr/bin/python
# coding: utf-8

import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader

def seed_everything(seed=2323):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def time_to_str(t, mode="min"):
    if mode == "min":
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return "%2d hr %02d min" % (hr, min)
    elif mode == "sec":
        t = int(t)
        min = t // 60
        sec = t % 60
        return "%2d min %02d sec" % (min, sec)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.avg = (self.avg * self.count + val) / (self.count + 1)
        self.count += 1