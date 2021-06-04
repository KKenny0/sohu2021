#!/usr/bin/python
# coding: utf-8

import os
import csv
import warnings
from timeit import default_timer as timer

import torch
import numpy as np

from datasets.sohu_dataset import SohuDataset
from bucket_iterator import BucketIterator
from train.tester import Tester

from callback.modelcheckpoint import ModelCheckpoint
from models.net import SMNet as SMN

from utils.logger import init_logger
from utils.utils import seed_everything, time_to_str
import config.args as args

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed_everything(args.SEED)
logger = init_logger(log_name=args.ARCH, log_dir=args.LOG_DIR)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sigmoid(x):
    res = 1 / (1+np.e**(-x))
    return res

def data_generator(vocab_path):
    dataset = SohuDataset(
        max_len=args.MAX_LENGTH,
        vocab_path=vocab_path,
        mode="test",
    )
    loader = BucketIterator(
        data=dataset,
        batch_size=args.BATCH_SIZE,
        sort_key=lambda x: len(x[2]),
        shuffle=False,
        sort=True,
        mode="test"
    )
    return loader


def do_predict(model_name: str, vocab_path: str, log_dir: str, checkpoint_dir: str):
    res = {}
    f_id = []

    model_path = {
        "wobert": "models/wobert",
        "wwm": "models/wwm",
        "roberta": "models/roberta"
    }

    print('Predicting on '+ model_name)

    loader = data_generator(vocab_path)

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
    test_prob, q_ids = tester.test()

    for ids in q_ids:
        for q_id in ids:
            f_id.append(q_id)
    test_prob = torch.cat(test_prob, dim=0)
    pred = test_prob.cpu().detach().numpy()
    for p, q_id in zip(pred, f_id):
        if q_id not in res.keys():
            res[q_id] = []
        res[q_id].append(p)

    return res, f_id
    

print("---------- Model Predicting ... ----------")
start_time = timer()

res1, q_ids = do_predict(model_name="wobert", vocab_path="models/wobert/vocab.txt", log_dir="output/logs_wobert", checkpoint_dir="output/ckpts_wobert")
res2, _ = do_predict(model_name="wwm", vocab_path="models/wwm/vocab.txt", log_dir="output/logs_wwm", checkpoint_dir="output/ckpts_wwm")
res3, _ = do_predict(model_name="roberta", vocab_path="models/roberta/vocab.txt", log_dir="output/logs_roberta", checkpoint_dir="output/ckpts_roberta")

mix3pred = {k: res1[k]+res2[k]+res3[k] for k in res1.keys()&res2.keys()&res3.keys()}

final_pred = {}
A_weights = [0.2100, 0.3400, 0.4500]
B_weights = [0.2400, 0.0900, 0.6700]
for q_id, vals in mix3pred.items():
    a_s = np.zeros_like(vals[0])
    b_s = np.zeros_like(vals[0])

    for i, val in enumerate(vals):
        a_s += (A_weights[i]*val) / len(vals)
        b_s += (B_weights[i]*val) / len(vals)

    a_s = sigmoid(a_s)
    b_s = sigmoid(b_s)

    if "a" in q_id:
        final_pred[q_id] = a_s
    else:
        final_pred[q_id] = b_s


# ------------- Make submission file ---------------
print("Making submission file...")
f = open(args.OUTPUT_PATH, 'w', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["id", "label"])

for q_id, ans in final_pred.items():
    ans = np.argmax(ans)
    csv_writer.writerow([q_id, ans])

f.close()

print(f'Took {time_to_str((timer() - start_time), "sec")}')
