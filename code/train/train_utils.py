#!/usr/bin/python
# coding: utf-8

import os

import numpy as np
import torch
from sklearn.metrics import f1_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def evaluate(predictions: list, labels: list, conds: list, mode: str="obj"):
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

def load_state(model, model_path):
    if os.path.exists(model_path):
        state = torch.load(model_path)
        epoch = state["epoch"]
        model.load_state_dict(state["model"])
        print(f"Restore model, epoch: {epoch}")
        return model, epoch
    else:
        print(f"Not found {model_path} model")
        return model, 1


def save_state(model, epoch, model_path):
    torch.save({"model": model.state_dict(), "epoch": epoch}, str(model_path))
