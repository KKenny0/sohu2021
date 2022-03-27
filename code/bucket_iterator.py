# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy as np

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='tokens', shuffle=True, sort=True, mode="train"):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.mode = mode
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)
        
    def sort_and_pad(self, data, batch_size):
        self.data_nums = len(data)
        
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            if isinstance(self.sort_key, str):
                sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
            else:
                sorted_data = sorted(data, key=self.sort_key)
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_la_id= []
        batch_cond= []
        batch_input_ids= []
        batch_segment_ids=  []
        batch_input_mask = []
        
        max_in_batch = max([len(x[2]) for x in batch_data])
        for item in batch_data:
            la_id, cond, input_ids, segment_ids, input_mask = \
                item[0], item[1], item[2], item[3], item[4]
            
            padding = [0] * (max_in_batch - len(input_ids))
            input_ids = input_ids + padding
            segment_ids = segment_ids + padding
            input_mask = input_mask + padding

            batch_la_id.append(la_id)
            batch_cond.append(cond)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(segment_ids)
            batch_input_mask.append(input_mask)

        if self.mode != 'infer':
            return { \
                    'batch_la_id': torch.tensor(batch_la_id), \
                    'batch_cond': torch.tensor(batch_cond), \
                    'batch_input_ids': torch.tensor(batch_input_ids), \
                    'batch_segment_ids': torch.tensor(batch_segment_ids), \
                    'batch_input_mask': torch.tensor(batch_input_mask), \
                }
        else:
            return { \
                    'batch_la_id': batch_la_id, \
                    'batch_cond': torch.tensor(batch_cond), \
                    'batch_input_ids': torch.tensor(batch_input_ids), \
                    'batch_segment_ids': torch.tensor(batch_segment_ids), \
                    'batch_input_mask': torch.tensor(batch_input_mask), \
                } 

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

    def __len__(self):
        return self.data_nums
