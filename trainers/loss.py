import logging
import math

import torch.nn as nn
import torch
import torch.nn.functional as F

from models.base import BaseModel

class SoftLoss:
    r"""
        no sample
    """

    def __init__(self, mentor: BaseModel, args) -> None:
        self.mentor = mentor
        self.alpha = args.alpha
        self.T = args.T
        self.num_item = args.num_items
        self.device = args.device

        self.softmaxed_mentor = args.softmaxed_mentor # the output from mentor has been softmaxed ?

        self.mentor.eval()

        self.nan = 0
        self.not_nan = 1
        self.debug = 40
        self.accum_iter = 0

    def debug_summary(self):
        logging.debug(f"loss nan summary: nan {self.nan} times, not nan {self.not_nan} times, ratio nan / (nan + not_nan) = {1.0 * self.nan / (self.nan + self.not_nan)}")

    def calculate_loss(self, model: BaseModel, batch):
        self.accum_iter += 1
        seqs = batch[0]
        labels = batch[1]

        with torch.no_grad():
            soft_target = self.mentor(batch)

        # output = (predict logits, predict loss, reg loss, ...)
        output = model.calculate_loss(batch)

        pred_logits = output[0]
        pred_loss = output[1]
        reg_loss = 0.

        if len(output) > 2:
            reg_loss = sum(output[2:])
        
        if len(pred_logits.size()) == 3:
            # batch_size * L * n_item, mask or auto regressive

            cl = labels[labels > 0]
            pred = pred_logits[labels > 0]

            assert(len(soft_target.size()) == 3)

            soft_target = soft_target[labels > 0]
        else:
            # batch_size * n_item, next item type

            cl = labels[labels > 0]
            pred = pred_logits

            if len(soft_target.size()) == 3:
                # B * L * N
                soft_target = soft_target[:, -1, :].squeeze()

        assert(soft_target.size() == pred_logits.size())

        cl_onehot = F.one_hot(cl, num_classes=self.num_item + 1)

        if self.softmaxed_mentor:
            soft_target = 0.5 * (soft_target + cl_onehot)
        else:
            soft_target = 0.5 * ((soft_target / self.T).softmax(dim=-1) + cl_onehot)

        if self.accum_iter % 1000 < 2 and self.accum_iter != 0:
            if self.debug != 0:
                with torch.no_grad():
                    self.debug -= 1
                    
                    logging.debug(f"soft_target max: {soft_target.max()}, argmax {soft_target.argmax()}")
                    logging.debug(f"pred max in softmax: {pred.softmax(dim=-1).max()}, argmax {pred.softmax(dim=-1).argmax()}")

        KL_loss = F.kl_div(F.log_softmax(pred[:, 1:], dim=-1), soft_target[:, 1:], reduction='batchmean')

        if ~torch.isinf(KL_loss):
            loss = (1 - self.alpha) * pred_loss + self.alpha * KL_loss + reg_loss
            self.not_nan += 1
        else:
            loss = pred_loss + reg_loss
            self.nan += 1
        
        return loss

class BasicLoss:
    def __init__(self, **kwargs) -> None:
        pass

    def debug_summary(self):
        pass

    def calculate_loss(self, model: BaseModel, batch):
        # output = (predict logits, predict loss, reg loss, ...)
        output = model.calculate_loss(batch)

        return sum(output[1:])
