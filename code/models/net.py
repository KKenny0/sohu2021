#!/usr/bin/python
# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchKbert.modeling import BertModel, BertPreTrainedModel

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        #self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))

    def forward(self, hidden_state: torch.Tensor, mask: torch.Tensor, type_embed: torch.Tensor):
        fc = self.fc + type_embed
        q = (hidden_state @ fc).squeeze(dim=-1)
        q = q.masked_fill((1-mask).bool(), torch.tensor(-1e4))
        w = F.softmax(q, dim=-1).unsqueeze(dim=1)
        h = w @ hidden_state
        return h.squeeze(dim=1)

class AttentionClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super(AttentionClassifier, self).__init__()
        self.attn = Attention(hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor, type_embed: torch.Tensor):
        h = self.attn(hidden_states, mask, type_embed)
        out = self.fc(h)
        return out

class SMNet(BertPreTrainedModel):
    def __init__(self, config):
        super(SMNet, self).__init__(config)
        self.domain1_encoder = BertModel(config)

        self.proj = nn.Linear(config.hidden_size, 3*config.hidden_size)
        self.task_clf = nn.ModuleList([AttentionClassifier(3*config.hidden_size, config.num_labels) for i in range(2)])
        self.type_clf = nn.Linear(config.hidden_size, 6)

    def forward(
        self,
        input_ids_1: torch.Tensor = None,
        token_type_ids_1: torch.Tensor = None,
        attention_mask_1: torch.Tensor = None,
        conds: torch.Tensor = None,
    ):

        seq, _ = self.domain1_encoder(
                    input_ids=input_ids_1, 
                    token_type_ids=token_type_ids_1,
                    attention_mask=attention_mask_1,
                    output_all_encoded_layers=True)
        seq_sum = torch.cat(
                    (
                        seq[-1],
                        seq[-2],
                        seq[-3]
                    ), dim=-1)

        type_embed = seq[-1][:,1,:]
        type_out = self.type_clf(type_embed)
        
        type_embed = self.proj(type_embed)
        logits = []
        for i, cond in enumerate(conds):
            logit = self.task_clf[int(cond%2)](seq_sum[i].unsqueeze(0), attention_mask_1[i].unsqueeze(0), type_embed[i].unsqueeze(-1))
            logits.append(logit)
        logits = torch.cat(logits, dim=0)

        return logits, type_out


        

