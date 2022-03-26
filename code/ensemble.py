import os
import sys
import csv
import logging
import argparse
from time import strftime, localtime
import torch
import numpy as np

from models.net import SMNet as SMN
from datasets.sohu_dataset import SohuDataset
from bucket_iterator import BucketIterator

from train import Trainer as Instructor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

MODEL_LIST = ["RoBERTa", "WoBERT", "WWM"]


def metrics_calculator(predictions, labels, conds):
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
    return f1_a, f1_b, f1


def evaluate(args, model_type):
    saving_path = os.path.join(args.save_path, model_type)
    model = SMN.from_pretrained(saving_path)
    model.to(args.device)

    vocab_path = os.path.join(saving_path, "vocab.txt")
    testset = SohuDataset(args, vocab_path, mode="valid", model_name=model_type)
    test_data_loader = BucketIterator(
                        data=testset,
                        batch_size=args.batch_size,
                        sort_key=lambda x: len(x[2]),
                        shuffle=False,
                        sort=False)

    test_logits, test_q_l, test_cond = Instructor._eval(model, test_data_loader, args.device, infer=True)
    test_logits = torch.cat(test_logits, dim=0).cpu().numpy()
    test_q_l = torch.cat(test_q_l, dim=0).numpy()
    test_cond = torch.cat(test_cond, dim=0).cpu().numpy()
    return test_logits, test_q_l, test_cond


def test(args):
    logger.info(">"*100)
    logger.info("Time: {}".format(strftime("%y%m%d-%H%M", localtime())))

    test_logits_1, test_label_1, test_cond_1 = evaluate(args, MODEL_LIST[0])
    test_logits_2, _, _ = evaluate(args, MODEL_LIST[1])
    test_logits_3, _, _ = evaluate(args, MODEL_LIST[2])

    a_mixpred = (0.2100*test_logits_1 + 0.3400*test_logits_2 + 0.4500*test_logits_3) / 3
    b_mixpred = (0.2400*test_logits_1 + 0.0900*test_logits_2 + 0.6700*test_logits_3) / 3
    mixpred = np.zeros_like(a_mixpred)
    for i, val in enumerate(a_mixpred):
        if int(test_cond_1[i] % 2) == 0:
            mixpred[i] = val
        else:
            mixpred[i] = b_mixpred[i]
    f1_a, f1_b, f1 = metrics_calculator(mixpred, test_label_1, test_cond_1)

    a_model_weight_info = "{}={}, {}={}, {}={}".format(MODEL_LIST[0], 0.2100, MODEL_LIST[1], 0.3400, MODEL_LIST[2], 0.4500)
    b_model_weight_info = "{}={}, {}={}, {}={}".format(MODEL_LIST[0], 0.2400, MODEL_LIST[1], 0.0900, MODEL_LIST[2], 0.6700)
    logger.info("Ensemble Information:")
    logger.info(">> Type A model and weight: {}.".format(a_model_weight_info))
    logger.info(">> Type B model and weight: {}.".format(b_model_weight_info))
    logger.info("Final ensemble result:")
    logger.info(">> Test result: test_f1_a: {:.4f}, test_f1_b: {:.4f}, test_f1: {:.4f}".format(f1_a, f1_b, f1))


def infer(args):
    test_logits_1, test_qid_1, _ = evaluate(args, MODEL_LIST[0])
    test_logits_2, _, _ = evaluate(args, MODEL_LIST[1])
    test_logits_3, _, _ = evaluate(args, MODEL_LIST[2])

    fids = []
    res1, res2, res3 = {}, {}, {}
    for qid in test_qid_1:
        for _id in qid:
            fids.append(_id)
    for logits1, logits2, logits3, qid in zip(test_logits_1, test_logits_2, test_logits_3, fids):
        if qid not in res1.keys():
            res1[qid] = []
            res2[qid] = []
            res3[qid] = []
        res1[qid] = logits1
        res2[qid] = logits2
        res3[qid] = logits3
        
    final_pred = {}
    a_weights = [0.2100, 0.3400, 0.4500]
    b_weights = [0.2400, 0.0900, 0.6700]
    mix3pred = {k: res1[k]+res2[k]+res3[k] for k in res1.keys()&res2.keys()&res3.keys()}
    for qid, vals in mix3pred.items():
        a_s = np.zeros_like(vals[0])
        b_s = np.zeros_like(vals[0])
        for i, val in enumerate(vals):
            a_s += (a_weights[i]*val) / len(vals)
            b_s += (b_weights[i]*val) / len(vals)
        if "a" in qid:
            final_pred[qid] = a_s
        else:
            final_pred[qid] = b_s

    with open(args.result_path, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id", "label"])
        for qid, ans in final_pred.items():
            ans = np.argmax(ans)
            csv_writer.writerow([qid, ans])  


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs", type=str)
    parser.add_argument("--save_path", default="output", type=str)
    parser.add_argument("--result_path", default="output/submission.csv", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--do_eval", action="store_true", help="Whether to run model ensemble on the test set")
    parser.add_argument("--do_infer", action="store_true", help="Whether to run model ensemble on the online data")
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_file = '{}/log-ensemble'.format(args.log)
    logger.addHandler(logging.FileHandler(log_file))

    if args.do_eval:
        test(args)
    elif args.do_infer:
        infer(args)


if __name__ == "__main__":
    main()