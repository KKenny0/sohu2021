#!/usr/bin/python
# coding: utf-8

import os
import warnings
from timeit import default_timer as timer

import torch
from torchKbert import BertAdam

from datasets.sohu_dataset import SohuDataset
from bucket_iterator import BucketIterator
from train.trainer import Trainer

from loss.loss import FocalLoss
from torch.nn import CrossEntropyLoss
from models.net import SMNet as SMN
from callback.modelcheckpoint import ModelCheckpoint

from utils.logger import init_logger
from utils.utils import seed_everything, time_to_str
import config.args as args

warnings.filterwarnings("ignore")


seed_everything(args.SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------- Training -------------
def data_generator(model_name: str, vocab_path: str):
    train_dataset = SohuDataset(
        max_len=args.MAX_LENGTH,
        vocab_path=vocab_path,
        mode='train',
        model_name=model_name
    )
    valid_dataset = SohuDataset(
        max_len=args.MAX_LENGTH,
        vocab_path=vocab_path,
        mode='valid',
        model_name=model_name
    )
    if model_name == 'roberta':
        args.BATCH_SIZE = 8
    train_loader = BucketIterator(
        data=train_dataset,
        batch_size=args.BATCH_SIZE,
        sort_key=lambda x: len(x[2]),
        shuffle=True,
        sort=True
    )
    valid_loader = BucketIterator(
        data=valid_dataset,
        batch_size=args.BATCH_SIZE,
        sort_key=lambda x: len(x[2]),
        shuffle=False,
        sort=True
    )

    return train_loader, valid_loader

def build_model(vocab_path: str, model_name: str):
    model_path = {
        "wobert": "models/wobert",
        "wwm": "models/wwm",
        "roberta": "models/roberta"
    }

    train_loader, valid_loader = data_generator(model_name, vocab_path)

    args.DOMAIN1_MODEL_PATH = model_path[model_name]
    if model_name == 'roberta':
        args.HIDDEN_SIZE = 1024
        args.BATCH_SIZE = 8
        args.GRADIENT_ACCUMULATION_STEPS = 4
    model = SMN(args)
    param_optimizer = list(filter(lambda x: x[1].requires_grad is not False, model.named_parameters()))
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    num_train_optimization_steps = (
        int(len(train_loader) / args.GRADIENT_ACCUMULATION_STEPS / args.BATCH_SIZE * args.NUM_EPOCHS)
    )
    print(num_train_optimization_steps)

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.LEARNING_RATE,
        warmup=args.WARMUP_PROPORTION,
        t_total=num_train_optimization_steps,
    )

    return model, optimizer, train_loader, valid_loader


def do_train(model_name: str, vocab_path:str, log_dir: str, checkpoint_dir: str):
    model, optimizer, train_loader, valid_loader = build_model(vocab_path, model_name)

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

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        batch_size=args.BATCH_SIZE,
        num_epochs=args.NUM_EPOCHS,
        device=device,
        criterion=FocalLoss(),
        type_criterion=CrossEntropyLoss(),
        gradient_accumulation_steps=args.GRADIENT_ACCUMULATION_STEPS,
        model_checkpoint=model_checkpoint,
        logger=logger,
        resume=args.RESUME,
        FP16=args.FP16,
        adv=args.do_adv,
        ema=args.do_ema,
    )

    start_time = timer()
    logger.info("---------- {} Model Training ----------".format(model_name))
    trainer.summary()
    trainer.train()
    logger.info(f'Took {time_to_str((timer() - start_time), "sec")}')


do_train(model_name="wobert", vocab_path="models/wobert/vocab.txt", log_dir="output/logs_wobert", checkpoint_dir="output/ckpts_wobert")

do_train(model_name="wwm", vocab_path="models/wwm/vocab.txt", log_dir="output/logs_wwm", checkpoint_dir="output/ckpts_wwm")

#do_train(model_name="roberta", vocab_path="models/roberta/vocab.txt", log_dir="output/logs_roberta", checkpoint_dir="output/ckpts_roberta")