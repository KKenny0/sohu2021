#!/usr/bin/python
# coding: utf-8

import os
import warnings
from timeit import default_timer as timer

import torch
import numpy as np

from callback.modelcheckpoint import ModelCheckpoint
from datasets.sohu_dataset import SohuDataset
from models.net import SMNet as SMN
from train.tester import Tester
from bucket_iterator import BucketIterator
from utils.logger import init_logger
from utils.utils import seed_everything, time_to_str
import config.args as args

warnings.filterwarnings("ignore")

seed_everything(args.SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
    res = 1 / (1+np.e**(-x))
    return res

def evaluate(predictions, labels, conds):
    assert(len(labels) == len(predictions))

    y_pred = predictions.argmax(axis=1)
    y_true = labels
    
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
    print("F1-A: {}, F1-B: {}, F1: {}".format(f1_a, f1_b, f1))

def data_generator(model_name:str, vocab_path: str):
    dataset = SohuDataset(
        max_len=args.MAX_LENGTH,
        vocab_path=vocab_path,
        mode='valid',
        model_name=model_name
    )
    loader = BucketIterator(
        data=dataset,
        batch_size=args.BATCH_SIZE,
        sort_key=lambda x: len(x[2]),
        shuffle=False,
        sort=False,
        mode="valid"
    )
    return loader

def do_eval(model_name: str, vocab_path: str, log_dir: str, checkpoint_dir: str):
    print('Evaluating on '+ model_name)

    model_path = {
        "wobert": "models/wobert",
        "wwm": "models/wwm",
        "roberta": "models/roberta"
    }

    loader = data_generator(model_name, vocab_path)

    args.DOMAIN1_MODEL_PATH = model_path[model_name]
    model = SMN(args)

    logger = init_logger(log_name=model_name, log_dir=log_dir)
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        mode="max",
        monitor="val_acc",
        save_best_only=True,
        best_model_name=args.BEST_MODEL_NAME,
        arch=model_name,
        logger=logger,
    )
    restore_list = model_checkpoint.restore(model)
    model = restore_list[0]
    model.to(device)

    tester = Tester(
        model=model,
        test_loader=loader,
        device=device,
        logger=logger,
        mode="infer"
    )
    pred, label, conds = tester.eval()

    pred = torch.cat(pred, dim=0)
    label = torch.cat(label, dim=0)
    conds = torch.cat(conds, dim=0)

    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    conds = conds.cpu().detach().numpy()

    return pred, label, conds
    

print("---------- Model Evaluating ... ----------")
start_time = timer()
pred1, label1, conds1 = do_eval(model_name="wobert", vocab_path="models/wobert/vocab.txt", log_dir="output/logs_wobert", checkpoint_dir="output/ckpts_wobert")
pred2, _, _ = do_eval(model_name="wwm", vocab_path="models/wwm/vocab.txt", log_dir="output/logs_wwm", checkpoint_dir="output/ckpts_wwm")
pred3, _, _ = do_eval(model_name="roberta", vocab_path="models/roberta/vocab.txt", log_dir="output/logs_roberta", checkpoint_dir="output/ckpts_roberta")


a_mixpred = (0.2100*pred1+0.3400*pred2+0.4500*pred3)/3
b_mixpred = (0.2400*pred1+0.0900*pred2+0.6700*pred3)/3

mixpred = np.zeros_like(a_mixpred)
for i, val in enumerate(a_mixpred):
    if int(conds1[i] % 2) == 0:
        mixpred[i] = val
    else:
        mixpred[i] = b_mixpred[i]

evaluate(mixpred, label1, conds1)

print(f'Took {time_to_str((timer() - start_time), "sec")}')
