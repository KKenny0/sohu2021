#!/usr/bin/python
# coding: utf-8

import numpy as np
import torch

from train.train_utils import model_device
from utils.utils import AverageMeter
from sklearn.metrics import f1_score

def sigmoid(x):
    res = 1 / (1+np.e**(-x))
    return res

def evaluate(predictions, labels, conds):
    assert(len(labels) == len(predictions))
    y_pred = torch.cat(predictions, dim=0)
    y_ture = torch.cat(labels, dim=0)
    conds = torch.cat(conds, dim=0)
    
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred.argmax(axis=1)
    y_true = y_ture.cpu().numpy()
    
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
    print(f1_a, f1_b, f1)
    return f1_a, f1_b, f1


class Tester(object):
    def __init__(
        self,
        model,
        test_loader,
        device,
        logger=None,
        mode="valid",
        verbose=1,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.logger = logger
        self.mode = mode
        self.verbose = verbose


    def eval(self):
        test_loader = self.test_loader

        test_pred = []
        label_all = []
        conds_all = []

        input_cols = ['batch_input_ids', 'batch_segment_ids', 'batch_input_mask']
        for batch in test_loader:
            self.model.eval()
            labels = batch["batch_la_id"].to(self.device)
            conds = batch["batch_cond"].to(self.device)
            batch = tuple(batch[col].to(self.device) for col in input_cols)

            with torch.no_grad():
                inputs = {
                    "input_ids_1": batch[0],
                    "token_type_ids_1": batch[1],
                    "attention_mask_1": batch[2],
                    "conds": conds
                }
                obj, type_logits = self.model(**inputs)

            test_pred.append( obj )
            label_all.append( labels )
            conds_all.append( conds )

        if self.mode != "infer":
            evaluate(test_pred, label_all, conds_all)
        else:
            return test_pred, label_all, conds_all

    def test(self):
        test_probs = []
        q_ids = []

        test_loader = self.test_loader
        input_cols = ['batch_input_ids', 'batch_segment_ids', 'batch_input_mask']
        for batch in test_loader:
            self.model.eval()
            q_id = batch["batch_la_id"]
            conds = batch["batch_cond"].to(self.device)
            batch = tuple(batch[col].to(self.device) for col in input_cols)

            with torch.no_grad():
                inputs = {
                    "input_ids_1": batch[0],
                    "token_type_ids_1": batch[1],
                    "attention_mask_1": batch[2],
                    "conds": conds
                }
                obj, type_logits = self.model(**inputs)
            
            q_ids.append(q_id)
            test_probs.append(obj)
        return test_probs, q_ids
