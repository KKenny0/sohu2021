#!/usr/bin/python
# coding: utf-8
import logging
import argparse
import os
import sys
import csv

from apex import amp
from time import strftime, localtime
import subprocess

import numpy as np
import torch
from torchKbert import BertAdam
from torchKbert.modeling import BertConfig
from loss.loss import FocalLoss
from torch.nn import CrossEntropyLoss
from models.net import SMNet as SMN
from models.EMA import EMA
from models.adversarial import FGM

from datasets.sohu_dataset import SohuDataset
from bucket_iterator import BucketIterator

from utils.utils import (
    seed_everything, 
    metrics_calculator)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))    


class Trainer(object):
    def __init__(self, args):
        self.args = args
        logger.info("Hyper Parameters")
        logger.info(args)

        config = BertConfig.from_json_file(os.path.join(args.bert_model, "config.json"))
        config.num_labels = 2
        self.model = SMN.from_pretrained(args.bert_model, config=config)
      
        self.vocab_path = os.path.join(args.bert_model, "vocab.txt")
        self.trainset = SohuDataset(args, self.vocab_path, mode="train")
        self.valset = SohuDataset(args, self.vocab_path, mode="valid")
        
        self._reset()

    def _reset(self):
        self.model = self.model.to(self.args.device)
        
        if self.args.do_ema:
            self.ema = EMA(self.model, 0.9999)
            
        if self.args.do_adv:
            self.fgm = FGM(self.model, emb_name=self.args.adv_name, epsilon=self.args.adv_epsilon)
            
    def summary(self):
        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(
            "trainable parameters: {:4}M".format(params / 1000 / 1000)
        )
        #logger.info(self.model)

    def _save_model(self, args, model):
        # If we save using the predefined name, we can load using `from_pretrained`
        output_model_file = os.path.join(args.saving_path, "pytorch_model.bin")
        torch.save(model.state_dict(), output_model_file)

        output_config_file = os.path.join(args.saving_path, "config.json")
        config = model.config
        with open(output_config_file, "w", encoding="utf-8") as fwrite:
            fwrite.write(config.to_json_string())
        output_args_file = os.path.join(args.saving_path, "training_args.bin")
        torch.save(args, output_args_file)
        subprocess.run(['cp', self.vocab_path, os.path.join(args.saving_path, 'vocab.txt')])

    def _eval(self, model, data_loader, device, infer=False):
        eval_pred, eval_conds, eval_correct = [], [], []

        input_cols = ['batch_input_ids', 'batch_segment_ids', 'batch_input_mask']
        model.eval()
        for _, batch in enumerate(data_loader):
            if infer:
                labels = batch["batch_la_id"]
            else:
                labels = batch["batch_la_id"].to(device)
            conds = batch["batch_cond"].to(device)
            batch = tuple(batch[col].to(device) for col in input_cols)

            with torch.no_grad():
                inputs = {"input_ids_1": batch[0],
                          "token_type_ids_1": batch[1],
                          "attention_mask_1": batch[2],
                          "conds": conds}       
                logits, _ = self.model(**inputs)

            eval_pred.append(logits)
            eval_conds.append(conds)
            eval_correct.append(labels)
        
        if infer:
            return eval_pred, eval_correct, eval_conds

        eval_f1_a, eval_f1_b, eval_f1 = metrics_calculator(eval_pred, eval_correct, eval_conds)
        return eval_f1_a, eval_f1_b, eval_f1

    def _train(self, criterion1, criterion2, train_loader, val_loader, optimizer):
        max_f1 = -1
        new_epoch = 0
        global_step = 0

        results = {}
        input_cols = ['batch_input_ids', 'batch_segment_ids', 'batch_input_mask']
        for epoch in range(self.args.num_epochs):
            logger.info("-" * 100)
            logger.info("Epoch {} precess: ".format(epoch))
            n_total, loss_total = 0, 0
            train_pred, train_conds, train_correct = [], [], []
            self.model.train()
            for step, batch in enumerate(train_loader):
                global_step += 1
                labels = batch["batch_la_id"].to(self.args.device)
                conds = batch["batch_cond"].to(self.args.device)
                train_batch = tuple(batch[col].to(self.args.device) for col in input_cols)
                batch_size = train_batch[0].size(0)
                inputs = {"input_ids_1": batch[0],
                         "token_type_ids_1": batch[1],
                         "attention_mask_1": batch[2],
                         "conds": conds}
                logits, type_logits = self.model(**inputs)

                loss1 = criterion1(logits, labels)
                loss2 = criterion2(type_logits, conds)
                loss = loss1 + loss2

                if self.args.gradient_accum > 1:
                    loss = loss / self.args.gradient_accum

                n_total += batch_size
                loss_total += loss.item() * batch_size

                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.fgm is not None:
                    self.fgm.attack()
                    adv_obj, adv_type_logits = self.model(**inputs)
                    loss_adv1 = criterion1(adv_obj, labels)
                    loss_adv2 = criterion2(adv_type_logits, conds)
                    loss_adv = loss_adv1 + loss_adv2
                    if self.fp16:
                        with amp.scale_loss(loss_adv, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_adv.backward()
                    self.fgm.restore()

                if (step+1) % self.gradient_accum == 0:
                    optimizer.step()
                    self.ema.update()
                    optimizer.zero_grad()

                train_pred.append(logits)
                train_conds.append(conds)
                train_correct.append(labels)
                if global_step % self.args.log_step == 0:
                    train_loss = loss_total / n_total
                    _, _, train_f1 = metrics_calculator(train_pred, train_correct, train_conds)
                    logger.info("   epoch-step: {}-{}, loss: {:.4f}, train f1: {:.4f}".format(epoch, global_step, train_loss, train_f1))

            self.ema.apply_shadow()
            val_f1_a, val_f1_b, val_f1 = self._eval(self.model, val_loader, device=self.args.device)
            self.ema.restore()
            logger.info(">> Epoch {} eval results -> val_f1_a: {:.4f}, val_f1_b: {:.4f}, val_f1: {:.4f}".format(epoch, val_f1_a, val_f1_b, val_f1))
            results["epoch{}_val_f1_a".format(epoch)] = val_f1_a
            results["epoch{}_val_f1_b".format(epoch)] = val_f1_b
            results["epoch{}_val_f1".format(epoch)] = val_f1

            if val_f1 > max_f1:
                max_f1 = val_f1
                new_epoch = epoch
                if not os.path.exists(self.args.saving_path):
                    os.makedirs(self.args.saving_path)
                self._save_model(self.args, self.model)
                results["max_val_f1"] = max_f1

            if epoch-new_epoch > self.args.early_stop:
                break

        logger.info("-"*100)
        logger.info("{} training summary.".format(strftime("%y%m%d-%H%M", localtime())))
        for k, v in results.items():
            logger.info("{}={}".format(k, v))
           
    def train(self):
        criterion1 = FocalLoss()
        criterion2 = CrossEntropyLoss()

        train_loader = BucketIterator(
                        data=self.trainset, 
                        batch_size=self.args.batch_size, 
                        sort_key=lambda x: len(x[2]),
                        shuffle=True,
                        sort=True)
        val_loader = BucketIterator(
                        data=self.valset, 
                        batch_size=self.args.batch_size, 
                        sort_key=lambda x: len(x[2]),
                        shuffle=False,
                        sort=False)

        model_params = list(filter(lambda x: x[1].requires_grad is not False, self.model.named_parameters()))
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        param_opt = [
            {
                "params": [p for n, p in model_params if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        num_train_opt_steps = (
            int(len(train_loader) / self.args.gradient_accum / self.args.batch_size * self.args.num_epoch)
        )
        optimizer = BertAdam(param_opt, lr=self.args.learning_rate, warmup=self.args.warmup_prop, t_total=num_train_opt_steps)

        if self.args.fp16:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O2", verbosity=0)

        self._train(criterion1, criterion2, train_loader, val_loader, optimizer)


def test(args):
    logger.info(">"*100)
    logger.info(args)
    config = BertConfig.from_json_file(os.path.join(args.saving_path, "config.json"))
    logger.info(config)

    model = SMN.from_pretrained(args.saving_path)
    model.to(args.device)

    vocab_path = os.path.join(args.saving_path, "vocab.txt")
    testset = SohuDataset(args, vocab_path, mode="test")
    test_data_loader = BucketIterator(
                        data=testset,
                        batch_size=args.batch_size,
                        sort_key=lambda x: len(x[2]),
                        shuffle=False,
                        sort=False)
    test_f1_a, test_f1_b, test_f1 = Trainer._eval(model, test_data_loader, args.device)
    logger.info(">> Final test result: test_f1_a: {:.4f}, test_f1_b: {:.4f}, test_f1: {:.4f}".format(test_f1_a, test_f1_b, test_f1))


# For predicting test data and generating submission file.
# The official test data does not include labels.
def infer(args):
    model = SMN.from_pretrained(args.saving_path)
    model.to(args.device)

    vocab_path = os.path.join(args.saving_path, "vocab.txt")
    testset = SohuDataset(args, vocab_path, mode="infer")
    test_data_loader = BucketIterator(
                        data=testset,
                        batch_size=args.batch_size,
                        sort_key=lambda x: len(x[2]),
                        shuffle=False,
                        sort=False,
                        mode="test")
    test_logits, test_qid, _ = Trainer._eval(model, test_data_loader, args.device, infer=True)

    data_id = []
    result = {}
    for qids in test_qid:
        for _id in qids:
            data_id.append(_id)
    test_logits = torch.cat(test_logits, dim=0)
    test_logits = test_logits.cpu().detach().numpy()
    for l, fid in zip(test_logits, data_id):
        if fid not in result.keys():
            result[fid] = []
        result[fid].append(l)

    with open(args.result_path, "w", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id", "label"])
        for fid, l in result.items():
            pred = np.argmax(l)
            csv_writer.writerow([fid, pred])


def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs", type=str)
    parser.add_argument("--result_path", default="output/submission.csv", type=str)
    parser.add_argument("--bert_model", default="models/RoBERTa", type=str)
    parser.add_argument("--saving_path", default="output/RoBERTa", type=str)
    parser.add_argument("--learning_rate", default="4e-5", type=float)
    parser.add_argument("--warmup_prop", default=0.05, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--gradient_accum", default=1, type=int)
    parser.add_argument("--num_epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--early_stop", default=3, type=int)
    parser.add_argument("--log_step", default=40, type=int)
    parser.add_argument("--seed", default=2323, type=int)
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--do_ema", action="store_true", help="Whether to use exponential moving average")
    parser.add_argument("--do_adv", action="store_true", help="Whether to do adversarial training")
    parser.add_argument("--adv_epsilon", default=1.0, type=float, help="Epsilon for adversarial training")
    parser.add_argument("--adv_name", default="word_embeddings", type=str, help="Name for adversarial layer")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set")
    parser.add_argument("--do_infer", action="store_true", help="Whether to infer on online data")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    seed_everything(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.log):
        os.makedirs(args.log)

    log_file = '{}/log-{}-{}'.format(args.log, args.bert_model.split("/")[-1], strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    if args.do_train:
        trainer = Trainer(args)
        trainer.summary()
        trainer.train()
    elif args.do_eval:
        test(args)
    elif args.do_infer:
        infer(args)


if __name__=="__main__":
    main()
