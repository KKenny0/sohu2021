#!/usr/bin/python
# coding: utf-8

import json
import numpy as np

from torchKbert.tokenization import BertTokenizer
from torch.utils.data import Dataset

import jieba
jieba.initialize()


class SohuDataset(Dataset):
    def __init__(self, args, vocab_path, mode="train", model_name=None):
        self.max_len = args.max_seq_len
        self.vocab_path = vocab_path
        self.mode = mode
        self.model_name = args.bert_model.split("/")[-1] if model_name is None else model_name
        self.reset()

    def reset(self):
        if self.model_name == "WoBERT":
            self.tokenizer = BertTokenizer(vocab_file=self.vocab_path, pre_tokenizer=lambda s: jieba.cut(s, HMM=False))
        else:
            self.tokenizer = BertTokenizer(vocab_file=self.vocab_path)
        self.examples = []
        self.build_examples()

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def read_data(self):
        variants = [
            u'短短匹配A类',
            u'短短匹配B类',
            u'短长匹配A类',
            u'短长匹配B类',
            u'长长匹配A类',
            u'长长匹配B类',
        ]
        
        data_all = []
        for i, var in enumerate(variants):
            key = "labelA" if "A" in var else "labelB"
            if self.mode == "train":
                fs = [
                    'datasets/%s/%s/train.txt' % (self.mode, var),
                    'datasets/%s/%s/train_r1.txt' % (self.mode, var),
                    'datasets/%s/%s/ext.txt' % (self.mode, var),
                    'datasets/%s/%s/train_r3.txt' % (self.mode, var),
                    'datasets/%s/%s/valid_g1.txt' % (self.mode, var)
                ]
            elif self.mode == "valid":
                fs = [
                    'datasets/%s/%s/valid.txt' % (self.mode, var)
                ]
            elif self.mode == "test": # load the local test data (w/ label)
                fs = [
                    'datasets/%s/%s/onehalf_new_test.txt' % (self.mode, var)
                ]
            for f in fs:
                with open(f) as f:
                    for l in f:
                        l = json.loads(l)
                        instance = {}
                        source = l['source']
                        target = l['target']
                        if len(source) > len(target) and (i==2 or i==3):
                            t = source
                            source = target
                            target = t
                        instance['source'] = source.replace(" ", "").replace("\n", "")
                        instance['target'] = target.replace(" ", "").replace("\n", "")
                        instance['cond'] = i
                        try:
                            instance['label'] = int(l[key])
                        except:
                            instance['label'] = int(l["label"])
                        data_all.append(instance)
        return data_all

    def read_test_data(self):
        variants = [
            u'短短匹配A类',
            u'短短匹配B类',
            u'短长匹配A类',
            u'短长匹配B类',
            u'长长匹配A类',
            u'长长匹配B类',
        ]
        data_all = []
        for i, var in enumerate(variants):
            f = 'datasets/%s/%s/test_with_id.txt' % (self.mode, var)
            with open(f) as f:
                for l in f:
                    l = json.loads(l)
                    instance = {}
                    source = l['source']
                    target = l['target']
                    instance['source'] = source
                    instance['target'] = target
                    instance['cond'] = i
                    instance['id'] = l['id']
                    data_all.append(instance)
            
        return data_all

    def build_features(self, source, target, cond, label_or_id):
        cond_tokens = {
            0: ["[SSA]"],
            1: ["[SSB]"],
            2: ["[SLA]"],
            3: ["[SLB]"],
            4: ["[LLA]"],
            5: ["[LLB]"]
        }
        cond_token = cond_tokens[cond]

        max_len = self.max_len-6
        if self.model_name != "wobert":
            tokens_s = self.tokenizer.tokenize(source, pre_tokenize=False)
            tokens_t = self.tokenizer.tokenize(target, pre_tokenize=False)
        else:
            tokens_s = self.tokenizer.tokenize(source)
            tokens_t = self.tokenizer.tokenize(target)
        sequences = [tokens_s, tokens_t]
        while True:
            lengths = [len(s) for s in sequences]
            if sum(lengths) > max_len:
                i = np.argmax(lengths)
                sequences[i].pop()
            else:
                break       

        tokens_s, tokens_t = sequences[0], sequences[1]
        tokens = ["[CLS]"] + cond_token + ["[<S>]"] + tokens_s + ["[</S>]"] + ["[<T>]"] + tokens_t + ["[</T>"]
        segment_ids = [0] * (len(cond_token) + len(tokens_s) + 3) + [1] * (len(tokens_t) + 2)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)     

        return (
            label_or_id,
            cond,
            input_ids,
            segment_ids,
            input_mask
        )

    def build_examples(self):
        if self.mode != 'infer':
            data_all = self.read_data()
            for _, x in enumerate(data_all):
                cond = x['cond']
                label_or_id = x['label']
                self.examples.append(self.build_features(x['source'], x['target'], cond, label_or_id))
        else: # read the official test data
            data_all = self.read_test_data()
            for _, x in enumerate(data_all):
                cond = x['cond']
                label_or_id = x['id']
                self.examples.append(self.build_features(x['source'], x['target'], cond, label_or_id))
