#!/usr/bin/python
# coding: utf-8


import torch
import torch.nn.functional as F
import numpy as np
from apex import amp
from tqdm import tqdm
from timeit import default_timer as timer


from models.EMA import EMA
from models.advesarial import FGM

import config.args as args
from train.train_utils import evaluate, model_device
from utils.utils import AverageMeter, seed_everything, time_to_str


class Trainer(object):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        batch_size,
        num_epochs,
        device,
        criterion,
        type_criterion,
        gradient_accumulation_steps,
        model_checkpoint,
        logger,
        resume,
        FP16,
        adv,
        ema,
        verbose=1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.criterion = criterion
        self.type_criterion = type_criterion
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.model_checkpoint = model_checkpoint
        self.logger = logger
        self.resume = resume
        self.fp16 = FP16
        self.adv = adv
        self.ema = ema
        self.verbose = verbose
        self._reset()

    def _reset(self):
        self.start_epoch = 0
        self.global_step = 0

        self.model = self.model.to(self.device)

        if self.resume:
            resume_list = self.model_checkpoint.restore(
                self.model
            )
            self.model = resume_list[0]
        
        # 是否使用 EMA （指数平均移动）
        if self.ema:
            print("EMA working")
            self.ema = EMA(self.model, 0.9999)
            
        # 是否使用对抗训练
        if self.adv:
            print("ADV working")
            self.fgm = FGM(self.model, emb_name=args.adv_name, epsilon=args.adv_epsilon)

        # 是否使用半精度浮点数加速
        if self.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level="O2", verbosity=0
            )
            

    def summary(self):
        model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(
            "trainable parameters: {:4}M".format(params / 1000 / 1000)
        )
        # self.logger.info(self.model)

    def _save_info(self, val_acc):
        state = {
            "state_dict": self.model.state_dict(),
            "val_acc": round(val_acc, 4),
        }
        return state

    def _valid_epoch(self):
        valid_loss = AverageMeter()
        valid_pred = []
        label_all = []
        valid_conds = []
        valid_type_pred = []

        input_cols = ['batch_input_ids', 'batch_segment_ids', 'batch_input_mask']
        for step, batch in enumerate(self.valid_loader):
            self.model.eval()
            labels = batch["batch_la_id"].to(self.device)
            conds = batch["batch_cond"].to(self.device)
            batch = tuple(batch[col].to(self.device) for col in input_cols)
            batch_size = batch[0].size(0)

            with torch.no_grad():
                inputs = {
                    "input_ids_1": batch[0],
                    "token_type_ids_1": batch[1],
                    "attention_mask_1": batch[2],
                    "conds": conds
                }
                obj, type_logits = self.model(**inputs)
                loss = self.criterion(obj, labels)
                type_loss = self.type_criterion(type_logits, conds)
                total_loss = loss+type_loss

            valid_type_pred.append( type_logits )
            valid_pred.append( obj )
            label_all.append( labels )
            valid_conds.append(conds)
            valid_loss.update(total_loss.item(), batch_size)
        
        valid_log = {
            "val_loss": valid_loss.avg, 
            "val_preds": valid_pred, 
            "val_labels": label_all, 
            "val_conds": valid_conds,
            "val_type_preds": valid_type_pred
        }

        return valid_log

    def _train_epoch(self, start_time):
        train_loss = AverageMeter()

        input_cols = ['batch_input_ids', 'batch_segment_ids', 'batch_input_mask']
        for step, batch in tqdm(enumerate(self.train_loader)):
            self.model.train()
            labels = batch["batch_la_id"].to(self.device)
            conds = batch["batch_cond"].to(self.device)
            batch = tuple(batch[col].to(self.device) for col in input_cols)
            batch_size = batch[0].size(0)

            inputs = {
                    "input_ids_1": batch[0],
                    "token_type_ids_1": batch[1],
                    "attention_mask_1": batch[2],
                    "conds": conds
                }

            obj, type_logits = self.model(**inputs)
            loss = self.criterion(obj, labels)
            type_loss = self.type_criterion(type_logits, conds)
            total_loss = loss + type_loss

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            train_loss.update(total_loss.item(), batch_size)

            # 是否使用半精度浮点数加速
            if self.fp16:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            if self.fgm is not None:
                self.fgm.attack()
                adv_obj, adv_type_logits = self.model(**inputs)
                loss_adv = self.criterion(adv_obj, labels)
                type_loss_adv = self.type_criterion(adv_type_logits, conds)
                total_loss_adv = loss_adv + type_loss_adv
                if self.fp16:
                    with amp.scale_loss(total_loss_adv, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss_adv.backward()
                self.fgm.restore()

            if (step + 1) % self.gradient_accumulation_steps == 0:  # using gradient accumulation to simulate batch
                self.optimizer.step()
                self.ema.update()
                self.optimizer.zero_grad()

            self.global_step += 1
            if (step+1) % (self.gradient_accumulation_steps*40) == 0:  # print every batch
                rate = self.optimizer.get_lr()

                now_epoch = (
                    self.global_step
                    * self.batch_size
                    / len(self.train_loader)
                )
                self.logger.info(
                    f"{rate[0]:.7f} "
                    f"{self.global_step / 1000:5.2f} "
                    f"{now_epoch:6.2f}  | "
                    f"{train_loss.avg:.4f}            | "
                    f'{time_to_str((timer() - start_time), "sec")}  '
                )
                #print(timer()-start_time)

        #train_log = {"loss": train_loss.avg, "train_pred": train_pred, "train_labels": train_labels}
        train_log = {"loss": train_loss.avg}

        return train_log

    def train(self):
        self.logger.info("     rate  step  epoch  |   loss  val_loss  |  time")
        self.logger.info("-" * 68)

        max_acc = -np.Inf
        min_loss = np.Inf
        new_epoch = 0

        start_time = timer()
        for epoch in range(self.num_epochs):
            seed_everything(epoch * 1000 + epoch)

            train_log = self._train_epoch(start_time)
            self.ema.apply_shadow()
            valid_log = self._valid_epoch()
            log = dict(train_log, **valid_log)
            val_preds = log["val_preds"]
            label_all = log["val_labels"]
            val_conds = log["val_conds"]
            val_type_preds = log["val_type_preds"]
            f1_a, f1_b, val_f1 = evaluate(val_preds, label_all, val_conds)
            type_f1 = evaluate(val_type_preds, val_conds, val_conds, mode='type')

            if val_f1 > max_acc and self.model_checkpoint:
                new_epoch = epoch
                max_acc = val_f1
                state = self._save_info(val_acc=val_f1)
                self.model_checkpoint.step(state=state)

            self.ema.restore()


            rate = self.optimizer.get_lr()
            now_epoch = (
                self.global_step
                * self.batch_size
                / len(self.train_loader)
            )

            asterisk = " "
            if log["val_loss"] < min_loss:
                min_loss = log["val_loss"]
                asterisk = "*"

            self.logger.info(
                f"{rate[0]:.7f} "
                f"{self.global_step / 1000:5.2f} "
                f"{now_epoch:6.2f}  | "
                f'{log["loss"]:.4f}    '
                f'{log["val_loss"]:.4f} {asterisk} |'
                f'{time_to_str((timer() - start_time), "sec")}  '
            )

            self.logger.info(
                f"val f1_a: {f1_a:.4f}  "
                f"val f1_b: {f1_b:.4f}  "
                f"val f1: {val_f1:.4f}  "
                f"val type-f1: {type_f1:.4f}  "
            )

            if epoch - new_epoch > 1:
                break
