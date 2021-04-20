# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_D.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from utils.helpers import truncated_normal_

from models.discriminator import CNNDiscriminator

# dis_filter_sizes = [2, 3, 4, 5]
# dis_num_filters = [300, 300, 300, 300]

n_heads = 4
n_transformer_layers = 3
class OurGAN_D(nn.Module):
    def __init__(self, embed_dim, max_seq_len, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(OurGAN_D, self).__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)

        # Returns BxTxD
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=n_heads),
            n_transformer_layers,
            norm=nn.LayerNorm(self.embed_dim)
        )

        # TODO consider adding activation/normalization?
        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim * self.max_seq_len, self.embed_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, 100),
            nn.LeakyReLU(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

        self.init_params()

        self.pos_encoding = self.positional_encoding()
    
    def positional_encoding(self):
        # From Assignment 3
        pos_indices = torch.arange(self.max_seq_len)[..., None]
        dim_indices = torch.arange(self.embed_dim//2)[None, ...]
        exponents = (2*dim_indices).float()/(self.embed_dim)
        trig_args = pos_indices / (10000**exponents)
        sin_terms = torch.sin(trig_args)
        cos_terms = torch.cos(trig_args)

        pos_encodings = torch.zeros((self.max_seq_len, self.embed_dim))
        pos_encodings[:, 0::2] = sin_terms
        pos_encodings[:, 1::2] = cos_terms

        if self.gpu:
            pos_encodings = pos_encodings.cuda()

        return pos_encodings

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp) # batch_size * max_seq_len * embed_dim

        seqlen = inp.size(1)
        emb = emb + self.pos_encoding[:seqlen]

        trans = self.transformer(emb) # batch * max_seq_len * embed_dim

        x = self.fc1(trans.flatten(start_dim=1)) #life is a highway.
        x = self.fc2(x)
        x = self.fc3(x)
        # logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        # # cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        # pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        # pred = torch.cat(pools, 1)
        # pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        # highway = self.highway(pred)
        # pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        # pred = self.feature2out(self.dropout(pred))
        # logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return x

    # def get_feature(self, inp):
    #     """
    #     Get feature vector of given sentences
    #     :param inp: batch_size * max_seq_len // * vocab_size
    #     :return: batch_size * feature_dim
    #     """
    #     emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
    #     convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
    #     pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
    #     pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
    #     highway = self.highway(pred)
    #     pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

    #     return pred

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.dis_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.dis_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.dis_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)
