# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : relgan_instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from models.RelGAN_D import RelGAN_D
from models.RelGAN_G import RelGAN_G
from utils.data_utils import GenDataIter
from utils.helpers import get_fixed_temperature, get_losses
from utils.metrics import BLEU
from utils.text_process import tensor_to_tokens


class RelGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(RelGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = RelGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = RelGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx,
                            gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.dis_criterion = nn.CrossEntropyLoss()  # For SeqGAN CNN Discriminator

        # DataLoader
        self.oracle_data = GenDataIter(cfg.train_data)
        self.test_data = GenDataIter(cfg.test_data)
        self.gen_data = GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size))

        # Metrics
        self.bleu3 = BLEU(test_text=tensor_to_tokens(self.gen_data.target, self.index_word_dict),
                          real_text=tensor_to_tokens(self.test_data.target, self.index_word_dict),
                          gram=3)

    def _run(self):
        # ==========PRE-TRAINING (GENERATOR)==========
        if not cfg.gen_pretrain:
            self._print('\nStarting Generator MLE Training...\n')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pretrain_generator: {}\n'.format(cfg.pretrained_gen_path))

        bleu3_score, gen_nll = self.cal_metrics()
        self._print('Initial generator: BLEU-3 = %.4f, gen_NLL = %.4f,\n' % (
            bleu3_score, gen_nll))

        # # ==========ADVERSARIAL TRAINING==========
        self._print('\nStarting Adversarial Training...\n')
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            self.sig.update()
            if self.sig.adv_sig:
                g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
                d_loss = self.adv_train_discriminator(cfg.ADV_d_step)  # Discriminator
                self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature

                progress.set_description(
                    'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.gen.temperature))

                # TEST
                if adv_epoch % cfg.adv_log_step == 0:
                    bleu3_score, gen_nll = self.cal_metrics()
                    self._print(
                        '[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f, BLEU-3 = %.4f, gen_NLL = %.4f,\n' % (
                            adv_epoch, g_loss, d_loss, bleu3_score, gen_nll))

                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self._print('\n>>> Stop by adv_signal! Finishing adversarial training...\n')
                progress.close()
                break

    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                # =====Train=====
                pre_loss = self.train_gen_epoch(self.gen, self.oracle_data.loader, self.mle_criterion, self.gen_opt)

                # =====Test=====
                if epoch % cfg.pre_log_step == 0:
                    bleu3_score, gen_nll = self.cal_metrics()
                    self._print(
                        '[MLE-GEN] epoch %d : pre_loss = %.4f, BLEU-3 = %.4f, gen_NLL = %.4f,\n' % (
                            epoch, pre_loss, bleu3_score, gen_nll))

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self._print('\n>>> Stop by pre signal, skip to adversarial training...')
                break
        if cfg.if_save and not cfg.if_test:
            self._save('MLE', epoch)

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.oracle_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step):
        total_loss = 0
        for step in range(d_step):
            real_samples = self.oracle_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()

        return total_loss / d_step if d_step != 0 else 0

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
