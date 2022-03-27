#!/usr/bin/python
# coding: utf-8

import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


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


def metrics_calculator(predictions: list, labels: list, conds: list, mode: str="obj"):
    assert(len(labels) == len(predictions))
    y_pred = torch.cat(predictions, dim=0)
    y_true = torch.cat(labels, dim=0)
    conds = torch.cat(conds, dim=0)
    
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred.argmax(axis=1)
    y_true = y_true.cpu().numpy()
    
    if mode != 'type':
        total_a, right_a = 0., 0.
        total_b, right_b = 0., 0.
        for i, p in enumerate(y_pred):
            t = y_true[i]
            flag = int(conds[i] % 2)
            
            total_a += ((p + t) * (flag == 0))
            right_a += ((p * t) * (flag == 0))
            total_b += ((p + t) * (flag == 1))
            right_b += ((p * t) * (flag == 1))

        
        f1_a = 2.0 * right_a / total_a
        f1_b = 2.0 * right_b / total_b
        f1 =  (f1_a + f1_b) / 2

        return f1_a, f1_b, f1
    else:
        f1 = f1_score(y_true, y_pred, average="macro")
        return f1

