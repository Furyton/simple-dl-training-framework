# -*- coding: utf-8 -*-
# @Time   : 2020/9/21
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/2
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
Caser
################################################
Reference:
    Jiaxi Tang et al., "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding" in WSDM 2018.
Reference code:
    https://github.com/graytowne/caser_pytorch
"""

import logging
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, xavier_normal_, constant_
from models.base import BaseModel
from models.loss import RegLoss

class CaserModel(BaseModel):
    r"""Caser is a model that incorporate CNN for recommendation.
    Note:
        We did not use the sliding window to generate training instances as in the paper, in order that
        the generation method we used is common to other sequential models.
        For comparison with other models, we set the parameter T in the paper as 1.
        In addition, to prevent excessive CNN layers (ValueError: Training loss is nan), please make sure the parameters MAX_ITEM_LIST_LENGTH small, such as 10.
    """

    def __init__(self, args, dataset, device, max_len, temp: float=1.0):
        super(CaserModel, self).__init__(dataset, device, max_len, temp)

        # load parameters info
        self.embedding_size = args.embed_size
        self.n_h = args.nh
        self.n_v = args.nv

        self.dropout_prob = args.hidden_dropout
        self.reg_weight = args.reg_weight
        self.loss_type = args.loss_type

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_user, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_item + 1, self.embedding_size, padding_idx=0)

        # vertical conv layer
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.max_len, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_len)]
        self.conv_h = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(i, self.embedding_size)) for i in lengths
        ])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size + self.embedding_size, self.embedding_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()
        self.reg_loss = RegLoss()

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        else:
            logging.critical(f"{self.loss_type} has not been implemented yet.")
            raise NotImplementedError
        # if self.loss_type == 'BPR':
        #     self.loss_fct = BPRLoss()
        # elif self.loss_type == 'CE':
        #     self.loss_fct = nn.CrossEntropyLoss()
        # else:
        #     raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 1.0 / module.embedding_dim)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    @classmethod
    def code(cls):
        return 'caser'

    def forward(self, batch):
        # Embedding Look-up
        # use unsqueeze() to get a 4-D input for convolution layers. (batch_size * 1 * max_length * embedding_size)
        item_seq = batch[0]
        # user = batch[4]

        item_seq_emb = self.item_embedding(item_seq).unsqueeze(1)
        # user_emb = self.user_embedding(user).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        # z = self.ac_fc(self.fc1(out))
        # x = torch.cat([z, user_emb], 1)
        # seq_output = self.ac_fc(self.fc2(x))
        # the hidden_state of the predicted item, size:(batch_size * hidden_size)
        seq_output = self.ac_fc(self.fc1(out))

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # logits size:(batch_size * n_item + 1)

        return logits / self._temperature

    def reg_loss_conv_h(self):
        r"""
        L2 loss on conv_h
        """
        loss_conv_h = 0
        for name, parm in self.conv_h.named_parameters():
            if name.endswith('weight'):
                loss_conv_h = loss_conv_h + loss_conv_h * parm.norm(2)
        return self.reg_weight * loss_conv_h

    def calculate_loss(self, batch):
        labels = batch[1]
        cl = labels.squeeze()

        logits = self.forward(batch)

        predict_loss = self.loss_fct(logits, cl)

        # reg_loss = self.reg_loss([self.user_embedding.weight, self.item_embedding.weight, self.conv_v.weight, self.fc1.weight, self.fc2.weight])
        reg_loss = self.reg_loss([self.item_embedding.weight, self.conv_v.weight, self.fc1.weight, self.fc2.weight])



        return logits, predict_loss, self.reg_weight * reg_loss + self.reg_loss_conv_h()

    @torch.no_grad()
    def predict(self, batch):
        candidates = batch[1]

        scores = self.forward(batch)
        scores = scores.gather(1, candidates)

        return scores

    @torch.no_grad()
    def full_sort_predict(self, batch):
        scores = self.forward(batch)
        return scores
