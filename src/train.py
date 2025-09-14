import torch
import torch.nn.functional as F
from torch import nn
import sys
import csv
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


# GAN生成器（加入门控机制）
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, seq_len=None):
        super(Generator, self).__init__()
        self.seq_len = seq_len  # 目标生成序列长度
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 输入维度 = 原始输入 + 历史生成结果（拼接后输入）
        self.fc = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 门控层：动态调整生成特征
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),  # 融合中间特征与原始输入
            nn.Sigmoid()
        )

        self.proj_out = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x, phase='train', eval_start=False, prev_fake=None):
        # 确保输入x是3D张量 (batch_size, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if phase == 'test':
            # 处理历史生成结果：确保与x的batch_size、seq_len完全匹配
            if prev_fake is None or eval_start:
                batch_size, seq_len = x.size(0), x.size(1)
                # 初始生成：创建非空零向量（避免后续拼接空张量）
                prev_fake = torch.zeros(batch_size, seq_len, self.output_dim, device=x.device)
            else:
                # 补全维度并调整序列长度
                if prev_fake.dim() == 2:
                    prev_fake = prev_fake.unsqueeze(1)
                # 线性插值调整seq_len，确保与x一致
                prev_fake = F.interpolate(
                    prev_fake.transpose(1, 2),  # (batch, dim, seq_len) 便于插值
                    size=x.size(1),
                    mode='linear'
                ).transpose(1, 2)  # 恢复为 (batch, seq_len, dim)
                # 强制batch_size一致
                if prev_fake.size(0) != x.size(0):
                    prev_fake = prev_fake[:x.size(0)]

            # 拼接输入与历史结果
            combined = torch.cat([x, prev_fake], dim=-1)
            hidden = self.fc(combined)

            # 门控机制：结合原始输入调整特征
            gate_input = torch.cat([hidden, x], dim=-1)  # 融合中间特征与原始输入
            gate = self.gate(gate_input)
            hidden = gate * hidden  # 过滤无效特征

            # 返回1个时间步结果（用于自回归生成）
            return self.proj_out(hidden)[:, :1, :]
        else:
            # 训练阶段：用与x同维度的零向量占位
            batch_size, seq_len = x.size(0), x.size(1)
            prev_fake = torch.zeros(batch_size, seq_len, self.output_dim, device=x.device)
            combined = torch.cat([x, prev_fake], dim=-1)
            hidden = self.fc(combined)

            # 门控机制
            gate_input = torch.cat([hidden, x], dim=-1)
            gate = self.gate(gate_input)
            hidden = gate * hidden

            return self.proj_out(hidden)


# GAN判别器（加入门控机制）
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        # 门控层：突出有效特征差异
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1概率
        )

    def forward(self, x):
        # 门控过滤无效特征
        gate = self.gate(x)  # [batch_size, seq_len, input_dim]
        x = x * gate  # 只保留重要特征

        # 对序列数据取时间步平均（适配任意序列长度）
        if x.dim() == 3:
            x = x.mean(dim=1)  # (batch_size, input_dim)
        return self.fc(x)


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    # 根据模态组合初始化生成器和判别器
    generators = []
    discriminators = []
    gen_optimizers = []
    dis_optimizers = []
    # 门控参数（多模态融合时使用）
    gate_params = []

    if hyp_params.modalities != 'LAV':
        # 计算输入输出维度（根据具体模态组合）
        if hyp_params.modalities == 'L':
            # 输入：文本维度，输出：音频和视频维度
            in_dim = hyp_params.orig_d_l
            out_dim_a = hyp_params.orig_d_a
            out_dim_v = hyp_params.orig_d_v
            generators.append(Generator(in_dim, out_dim_a, seq_len=hyp_params.a_len))
            generators.append(Generator(in_dim, out_dim_v, seq_len=hyp_params.v_len))
            discriminators.append(Discriminator(out_dim_a))
            discriminators.append(Discriminator(out_dim_v))
            # 初始化多模态融合门控
            gate_a = nn.Sequential(nn.Linear(out_dim_a, 1), nn.Sigmoid())
            gate_v = nn.Sequential(nn.Linear(out_dim_v, 1), nn.Sigmoid())
            gate_params.extend([gate_a, gate_v])

        elif hyp_params.modalities == 'A':
            # 输入：音频维度，输出：文本和视频维度
            in_dim = hyp_params.orig_d_a
            out_dim_l = hyp_params.orig_d_l
            out_dim_v = hyp_params.orig_d_v
            generators.append(Generator(in_dim, out_dim_l, seq_len=hyp_params.l_len))
            generators.append(Generator(in_dim, out_dim_v, seq_len=hyp_params.v_len))
            discriminators.append(Discriminator(out_dim_l))
            discriminators.append(Discriminator(out_dim_v))
            # 初始化多模态融合门控
            gate_l = nn.Sequential(nn.Linear(out_dim_l, 1), nn.Sigmoid())
            gate_v = nn.Sequential(nn.Linear(out_dim_v, 1), nn.Sigmoid())
            gate_params.extend([gate_l, gate_v])

        elif hyp_params.modalities == 'V':
            # 输入：视频维度，输出：文本和音频维度
            in_dim = hyp_params.orig_d_v
            out_dim_l = hyp_params.orig_d_l
            out_dim_a = hyp_params.orig_d_a
            generators.append(Generator(in_dim, out_dim_l, seq_len=hyp_params.l_len))
            generators.append(Generator(in_dim, out_dim_a, seq_len=hyp_params.a_len))
            discriminators.append(Discriminator(out_dim_l))
            discriminators.append(Discriminator(out_dim_a))
            # 初始化多模态融合门控
            gate_l = nn.Sequential(nn.Linear(out_dim_l, 1), nn.Sigmoid())
            gate_a = nn.Sequential(nn.Linear(out_dim_a, 1), nn.Sigmoid())
            gate_params.extend([gate_l, gate_a])

        elif hyp_params.modalities == 'LA':
            # 输入：文本+音频维度，输出：视频维度
            in_dim = hyp_params.orig_d_l + hyp_params.orig_d_a
            out_dim_v = hyp_params.orig_d_v
            generators.append(Generator(in_dim, out_dim_v, seq_len=hyp_params.v_len))
            discriminators.append(Discriminator(out_dim_v))
            # 初始化多模态融合门控
            gate_v = nn.Sequential(nn.Linear(out_dim_v, 1), nn.Sigmoid())
            gate_params.append(gate_v)

        elif hyp_params.modalities == 'LV':
            # 输入：文本+视频维度，输出：音频维度
            in_dim = hyp_params.orig_d_l + hyp_params.orig_d_v
            out_dim_a = hyp_params.orig_d_a
            generators.append(Generator(in_dim, out_dim_a, seq_len=hyp_params.a_len))
            discriminators.append(Discriminator(out_dim_a))
            # 初始化多模态融合门控
            gate_a = nn.Sequential(nn.Linear(out_dim_a, 1), nn.Sigmoid())
            gate_params.append(gate_a)

        elif hyp_params.modalities == 'AV':
            # 输入：音频+视频维度，输出：文本维度
            in_dim = hyp_params.orig_d_a + hyp_params.orig_d_v
            out_dim_l = hyp_params.orig_d_l
            generators.append(Generator(in_dim, out_dim_l, seq_len=hyp_params.l_len))
            discriminators.append(Discriminator(out_dim_l))
            # 初始化多模态融合门控
            gate_l = nn.Sequential(nn.Linear(out_dim_l, 1), nn.Sigmoid())
            gate_params.append(gate_l)

        else:
            raise ValueError('Unknown modalities type')

        # 移动到GPU并初始化优化器
        for i in range(len(generators)):
            if hyp_params.use_cuda:
                generators[i] = generators[i].cuda()
                discriminators[i] = discriminators[i].cuda()
            gen_optimizers.append(optim.Adam(generators[i].parameters(), lr=hyp_params.gen_lr))
            dis_optimizers.append(optim.Adam(discriminators[i].parameters(), lr=hyp_params.dis_lr))

        # 门控参数优化器
        if hyp_params.use_cuda:
            for gate in gate_params:
                gate = gate.cuda()
        gate_optimizer = optim.Adam([p for gate in gate_params for p in gate.parameters()], lr=hyp_params.gen_lr)

        gan_criterion = nn.BCELoss()

    # 初始化主模型
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    # 主模型优化器
    if hyp_params.use_bert:
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.text_model.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
            {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
    else:
        optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

    # 组装设置字典
    settings = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'gate_params': gate_params,
        'gate_optimizer': gate_optimizer if hyp_params.modalities != 'LAV' else None
    }

    if hyp_params.modalities != 'LAV':
        settings.update({
            'generators': generators,
            'discriminators': discriminators,
            'gen_optimizers': gen_optimizers,
            'dis_optimizers': dis_optimizers,
            'gan_criterion': gan_criterion
        })

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    gate_params = settings['gate_params']
    gate_optimizer = settings['gate_optimizer']

    generators = settings.get('generators', [])
    discriminators = settings.get('discriminators', [])
    gen_optimizers = settings.get('gen_optimizers', [])
    dis_optimizers = settings.get('dis_optimizers', [])
    gan_criterion = settings.get('gan_criterion', None)

    def train():
        model.train()
        for g in generators:
            g.train()
        for d in discriminators:
            d.train()
        for gate in gate_params:
            gate.train()

        epoch_loss = 0
        start_time = time.time()

        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)

            # 清零梯度
            model.zero_grad()
            for opt in gen_optimizers:
                opt.zero_grad()
            for opt in dis_optimizers:
                opt.zero_grad()
            if gate_optimizer:
                gate_optimizer.zero_grad()

            # 设备分配
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)
            net = nn.DataParallel(model) if hyp_params.distribute else model
            fake_data = []
            real_data = []

            # 1. 训练判别器
            if hyp_params.modalities == 'L':
                # 生成音频和视频
                gen_a, gen_v = generators
                dis_a, dis_v = discriminators

                # 真实数据
                real_a = audio
                real_v = vision
                real_data = [real_a, real_v]

                # 生成数据
                fake_a = gen_a(text)
                fake_v = gen_v(text)
                fake_data = [fake_a, fake_v]

                # 判别器损失
                real_label = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(batch_size, 1)
                fake_label = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(batch_size, 1)

                # 音频判别器
                dis_loss_a = (gan_criterion(dis_a(real_a), real_label) +
                              gan_criterion(dis_a(fake_a.detach()), fake_label)) / 2
                dis_loss_a.backward(retain_graph=True)
                dis_optimizers[0].step()

                # 视频判别器
                dis_loss_v = (gan_criterion(dis_v(real_v), real_label) +
                              gan_criterion(dis_v(fake_v.detach()), fake_label)) / 2
                dis_loss_v.backward(retain_graph=True)
                dis_optimizers[1].step()

            elif hyp_params.modalities == 'A':
                # 生成文本和视频
                gen_l, gen_v = generators
                dis_l, dis_v = discriminators

                real_l = text
                real_v = vision
                real_data = [real_l, real_v]

                fake_l = gen_l(audio)
                fake_v = gen_v(audio)
                fake_data = [fake_l, fake_v]

                real_label = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(batch_size, 1)
                fake_label = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(batch_size, 1)

                # 文本判别器
                dis_loss_l = (gan_criterion(dis_l(real_l), real_label) +
                              gan_criterion(dis_l(fake_l.detach()), fake_label)) / 2
                dis_loss_l.backward(retain_graph=True)
                dis_optimizers[0].step()

                # 视频判别器
                dis_loss_v = (gan_criterion(dis_v(real_v), real_label) +
                              gan_criterion(dis_v(fake_v.detach()), fake_label)) / 2
                dis_loss_v.backward(retain_graph=True)
                dis_optimizers[1].step()

            elif hyp_params.modalities == 'V':
                # 生成文本和音频
                gen_l, gen_a = generators
                dis_l, dis_a = discriminators

                real_l = text
                real_a = audio
                real_data = [real_l, real_a]

                fake_l = gen_l(vision)
                fake_a = gen_a(vision)
                fake_data = [fake_l, fake_a]

                real_label = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(batch_size, 1)
                fake_label = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(batch_size, 1)

                # 文本判别器
                dis_loss_l = (gan_criterion(dis_l(real_l), real_label) +
                              gan_criterion(dis_l(fake_l.detach()), fake_label)) / 2
                dis_loss_l.backward(retain_graph=True)
                dis_optimizers[0].step()

                # 音频判别器
                dis_loss_a = (gan_criterion(dis_a(real_a), real_label) +
                              gan_criterion(dis_a(fake_a.detach()), fake_label)) / 2
                dis_loss_a.backward(retain_graph=True)
                dis_optimizers[1].step()

            elif hyp_params.modalities == 'LA':
                # 生成视频
                gen_v = generators[0]
                dis_v = discriminators[0]

                real_v = vision
                real_data = [real_v]

                # 拼接文本和音频作为输入
                la_input = torch.cat([text, audio], dim=-1)
                fake_v = gen_v(la_input)
                fake_data = [fake_v]

                real_label = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(batch_size, 1)
                fake_label = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(batch_size, 1)

                # 视频判别器
                dis_loss_v = (gan_criterion(dis_v(real_v), real_label) +
                              gan_criterion(dis_v(fake_v.detach()), fake_label)) / 2
                dis_loss_v.backward(retain_graph=True)
                dis_optimizers[0].step()

            elif hyp_params.modalities == 'LV':
                # 生成音频
                gen_a = generators[0]
                dis_a = discriminators[0]

                real_a = audio
                real_data = [real_a]

                # 拼接文本和视频作为输入
                lv_input = torch.cat([text, vision], dim=-1)
                fake_a = gen_a(lv_input)
                fake_data = [fake_a]

                real_label = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(batch_size, 1)
                fake_label = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(batch_size, 1)

                # 音频判别器
                dis_loss_a = (gan_criterion(dis_a(real_a), real_label) +
                              gan_criterion(dis_a(fake_a.detach()), fake_label)) / 2
                dis_loss_a.backward(retain_graph=True)
                dis_optimizers[0].step()

            elif hyp_params.modalities == 'AV':
                # 生成文本
                gen_l = generators[0]
                dis_l = discriminators[0]

                real_l = text
                real_data = [real_l]

                # 拼接音频和视频作为输入
                av_input = torch.cat([audio, vision], dim=-1)
                fake_l = gen_l(av_input)
                fake_data = [fake_l]

                real_label = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(batch_size, 1)
                fake_label = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(batch_size, 1)

                # 文本判别器
                dis_loss_l = (gan_criterion(dis_l(real_l), real_label) +
                              gan_criterion(dis_l(fake_l.detach()), fake_label)) / 2
                dis_loss_l.backward(retain_graph=True)
                dis_optimizers[0].step()

            # 2. 训练生成器、门控和主模型
            gen_loss = 0
            if hyp_params.modalities == 'L':
                fake_a, fake_v = fake_data
                # 生成器损失（对抗损失+重建损失）
                gen_loss_a = gan_criterion(discriminators[0](fake_a), real_label) + \
                             F.mse_loss(fake_a, real_data[0])
                gen_loss_v = gan_criterion(discriminators[1](fake_v), real_label) + \
                             F.mse_loss(fake_v, real_data[1])
                gen_loss = gen_loss_a + gen_loss_v

                # 门控融合：真实特征与生成特征
                gate_a, gate_v = gate_params
                weight_a = gate_a(real_data[0])  # 门控系数 (batch, seq_len, 1)
                weight_v = gate_v(real_data[1])
                weighted_a = weight_a * real_data[0] + (1 - weight_a) * fake_a
                weighted_v = weight_v * real_data[1] + (1 - weight_v) * fake_v

                # 主模型预测
                preds, _ = net(text, weighted_a, weighted_v)

            elif hyp_params.modalities == 'A':
                fake_l, fake_v = fake_data
                gen_loss_l = gan_criterion(discriminators[0](fake_l), real_label) + \
                             F.mse_loss(fake_l, real_data[0])
                gen_loss_v = gan_criterion(discriminators[1](fake_v), real_label) + \
                             F.mse_loss(fake_v, real_data[1])
                gen_loss = gen_loss_l + gen_loss_v

                # 门控融合
                gate_l, gate_v = gate_params
                weight_l = gate_l(real_data[0])
                weight_v = gate_v(real_data[1])
                weighted_l = weight_l * real_data[0] + (1 - weight_l) * fake_l
                weighted_v = weight_v * real_data[1] + (1 - weight_v) * fake_v

                preds, _ = net(weighted_l, audio, weighted_v)

            elif hyp_params.modalities == 'V':
                fake_l, fake_a = fake_data
                gen_loss_l = gan_criterion(discriminators[0](fake_l), real_label) + \
                             F.mse_loss(fake_l, real_data[0])
                gen_loss_a = gan_criterion(discriminators[1](fake_a), real_label) + \
                             F.mse_loss(fake_a, real_data[1])
                gen_loss = gen_loss_l + gen_loss_a

                # 门控融合
                gate_l, gate_a = gate_params
                weight_l = gate_l(real_data[0])
                weight_a = gate_a(real_data[1])
                weighted_l = weight_l * real_data[0] + (1 - weight_l) * fake_l
                weighted_a = weight_a * real_data[1] + (1 - weight_a) * fake_a

                preds, _ = net(weighted_l, weighted_a, vision)

            elif hyp_params.modalities == 'LA':
                fake_v = fake_data[0]
                gen_loss_v = gan_criterion(discriminators[0](fake_v), real_label) + \
                             F.mse_loss(fake_v, real_data[0])
                gen_loss = gen_loss_v

                # 门控融合
                gate_v = gate_params[0]
                weight_v = gate_v(real_data[0])
                weighted_v = weight_v * real_data[0] + (1 - weight_v) * fake_v

                preds, _ = net(text, audio, weighted_v)

            elif hyp_params.modalities == 'LV':
                fake_a = fake_data[0]
                gen_loss_a = gan_criterion(discriminators[0](fake_a), real_label) + \
                             F.mse_loss(fake_a, real_data[0])
                gen_loss = gen_loss_a

                # 门控融合
                gate_a = gate_params[0]
                weight_a = gate_a(real_data[0])
                weighted_a = weight_a * real_data[0] + (1 - weight_a) * fake_a

                preds, _ = net(text, weighted_a, vision)

            elif hyp_params.modalities == 'AV':
                fake_l = fake_data[0]
                gen_loss_l = gan_criterion(discriminators[0](fake_l), real_label) + \
                             F.mse_loss(fake_l, real_data[0])
                gen_loss = gen_loss_l

                # 门控融合
                gate_l = gate_params[0]
                weight_l = gate_l(real_data[0])
                weighted_l = weight_l * real_data[0] + (1 - weight_l) * fake_l

                preds, _ = net(weighted_l, audio, vision)

            elif hyp_params.modalities == 'LAV':
                preds, _ = net(text, audio, vision)
                gen_loss = 0

            else:
                raise ValueError('Unknown modalities type')

            # 计算主任务损失
            if hyp_params.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
            task_loss = criterion(preds, eval_attr)

            # 总损失（主任务损失 + GAN损失）
            total_loss = task_loss + hyp_params.gan_weight * gen_loss
            total_loss.backward()

            # 更新生成器、门控和主模型参数
            for opt in gen_optimizers:
                torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], hyp_params.clip)
                opt.step()

            if gate_optimizer:
                torch.nn.utils.clip_grad_norm_([p for gate in gate_params for p in gate.parameters()], hyp_params.clip)
                gate_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            epoch_loss += total_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(test=False):
        model.eval()
        for g in generators:
            g.eval()
        for gate in gate_params:
            gate.eval()

        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()

                batch_size = text.size(0)
                net = nn.DataParallel(model) if hyp_params.distribute else model
                fake_data = []

                # 生成缺失模态
                if hyp_params.modalities == 'L':
                    if not test:
                        fake_a = generators[0](text)
                        fake_v = generators[1](text)
                    else:
                        # 初始化非空张量（与batch_size匹配）
                        fake_a = torch.zeros(batch_size, 0, generators[0].output_dim, device=text.device)
                        fake_v = torch.zeros(batch_size, 0, generators[1].output_dim, device=text.device)
                        for i in range(hyp_params.a_len):
                            if i == 0:
                                token = generators[0](text, phase='test', eval_start=True)
                            else:
                                token = generators[0](text, phase='test', prev_fake=fake_a)
                            fake_a = torch.cat((fake_a, token), dim=1)
                        for i in range(hyp_params.v_len):
                            if i == 0:
                                token = generators[1](text, phase='test', eval_start=True)
                            else:
                                token = generators[1](text, phase='test', prev_fake=fake_v)
                            fake_v = torch.cat((fake_v, token), dim=1)
                    fake_data = [fake_a, fake_v]

                    # 门控融合
                    gate_a, gate_v = gate_params
                    weight_a = gate_a(audio)
                    weight_v = gate_v(vision)
                    weighted_a = weight_a * audio + (1 - weight_a) * fake_a
                    weighted_v = weight_v * vision + (1 - weight_v) * fake_v

                    preds, _ = net(text, weighted_a, weighted_v)

                elif hyp_params.modalities == 'A':
                    if not test:
                        fake_l = generators[0](audio)
                        fake_v = generators[1](audio)
                    else:
                        # 初始化非空张量（与batch_size匹配）
                        fake_l = torch.zeros(batch_size, 0, generators[0].output_dim, device=audio.device)
                        fake_v = torch.zeros(batch_size, 0, generators[1].output_dim, device=audio.device)

                        for i in range(hyp_params.l_len):
                            if i == 0:
                                token = generators[0](audio, phase='test', eval_start=True)
                            else:
                                token = generators[0](audio, phase='test', prev_fake=fake_l)
                            fake_l = torch.cat((fake_l, token), dim=1)

                        for i in range(hyp_params.v_len):
                            if i == 0:
                                token = generators[1](audio, phase='test', eval_start=True)
                            else:
                                token = generators[1](audio, phase='test', prev_fake=fake_v)
                            fake_v = torch.cat((fake_v, token), dim=1)
                    fake_data = [fake_l, fake_v]

                    # 门控融合
                    gate_l, gate_v = gate_params
                    weight_l = gate_l(text)
                    weight_v = gate_v(vision)
                    weighted_l = weight_l * text + (1 - weight_l) * fake_l
                    weighted_v = weight_v * vision + (1 - weight_v) * fake_v

                    preds, _ = net(weighted_l, audio, weighted_v)

                elif hyp_params.modalities == 'V':
                    if not test:
                        fake_l = generators[0](vision)
                        fake_a = generators[1](vision)
                    else:
                        # 初始化非空张量（与batch_size匹配）
                        fake_l = torch.zeros(batch_size, 0, generators[0].output_dim, device=vision.device)
                        fake_a = torch.zeros(batch_size, 0, generators[1].output_dim, device=vision.device)
                        for i in range(hyp_params.l_len):
                            if i == 0:
                                token = generators[0](vision, phase='test', eval_start=True)
                            else:
                                token = generators[0](vision, phase='test', prev_fake=fake_l)
                            fake_l = torch.cat((fake_l, token), dim=1)
                        for i in range(hyp_params.a_len):
                            if i == 0:
                                token = generators[1](vision, phase='test', eval_start=True)
                            else:
                                token = generators[1](vision, phase='test', prev_fake=fake_a)
                            fake_a = torch.cat((fake_a, token), dim=1)
                    fake_data = [fake_l, fake_a]

                    # 门控融合
                    gate_l, gate_a = gate_params
                    weight_l = gate_l(text)
                    weight_a = gate_a(audio)
                    weighted_l = weight_l * text + (1 - weight_l) * fake_l
                    weighted_a = weight_a * audio + (1 - weight_a) * fake_a

                    preds, _ = net(weighted_l, weighted_a, vision)

                elif hyp_params.modalities == 'LA':
                    if not test:
                        la_input = torch.cat([text, audio], dim=-1)
                        fake_v = generators[0](la_input)
                    else:
                        # 初始化非空张量（与batch_size匹配）
                        fake_v = torch.zeros(batch_size, 0, generators[0].output_dim, device=text.device)
                        la_input = torch.cat([text, audio], dim=-1)
                        for i in range(hyp_params.v_len):
                            if i == 0:
                                token = generators[0](la_input, phase='test', eval_start=True)
                            else:
                                token = generators[0](la_input, phase='test', prev_fake=fake_v)
                            fake_v = torch.cat((fake_v, token), dim=1)
                    fake_data = [fake_v]

                    # 门控融合
                    gate_v = gate_params[0]
                    weight_v = gate_v(vision)
                    weighted_v = weight_v * vision + (1 - weight_v) * fake_v

                    preds, _ = net(text, audio, weighted_v)

                elif hyp_params.modalities == 'LV':
                    if not test:
                        lv_input = torch.cat([text, vision], dim=-1)
                        fake_a = generators[0](lv_input)
                    else:
                        # 初始化非空张量（与batch_size匹配）
                        fake_a = torch.zeros(batch_size, 0, generators[0].output_dim, device=text.device)
                        lv_input = torch.cat([text, vision], dim=-1)
                        for i in range(hyp_params.a_len):
                            if i == 0:
                                token = generators[0](lv_input, phase='test', eval_start=True)
                            else:
                                token = generators[0](lv_input, phase='test', prev_fake=fake_a)
                            fake_a = torch.cat((fake_a, token), dim=1)
                    fake_data = [fake_a]

                    # 门控融合
                    gate_a = gate_params[0]
                    weight_a = gate_a(audio)
                    weighted_a = weight_a * audio + (1 - weight_a) * fake_a

                    preds, _ = net(text, weighted_a, vision)

                elif hyp_params.modalities == 'AV':
                    if not test:
                        av_input = torch.cat([audio, vision], dim=-1)
                        fake_l = generators[0](av_input)
                    else:
                        # 初始化非空张量（与batch_size匹配）
                        fake_l = torch.zeros(batch_size, 0, generators[0].output_dim, device=audio.device)
                        av_input = torch.cat([audio, vision], dim=-1)
                        for i in range(hyp_params.l_len):
                            if i == 0:
                                token = generators[0](av_input, phase='test', eval_start=True)
                            else:
                                token = generators[0](av_input, phase='test', prev_fake=fake_l)
                            fake_l = torch.cat((fake_l, token), dim=1)
                    fake_data = [fake_l]

                    # 门控融合
                    gate_l = gate_params[0]
                    weight_l = gate_l(text)
                    weighted_l = weight_l * text + (1 - weight_l) * fake_l

                    preds, _ = net(weighted_l, audio, vision)

                elif hyp_params.modalities == 'LAV':
                    preds, _ = net(text, audio, vision)

                else:
                    raise ValueError('Unknown modalities type')

                # 计算损失
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                raw_loss = criterion(preds, eval_attr)
                total_loss += raw_loss.item() * batch_size

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    # 打印模型参数数量
    if hyp_params.modalities != 'LAV':
        gen_params = sum([sum(p.numel() for p in g.parameters()) for g in generators])
        dis_params = sum([sum(p.numel() for p in d.parameters()) for d in discriminators])
        gate_params_count = sum([sum(p.numel() for p in gate.parameters()) for gate in gate_params])
        print(f'Trainable Parameters for Generators: {gen_params}')
        print(f'Trainable Parameters for Discriminators: {dis_params}')
        print(f'Trainable Parameters for Gates: {gate_params_count}')

    mum_parameter = sum([param.nelement() for param in model.parameters()])
    print(f'Trainable Parameters for Multimodal Understanding Model (MUM): {mum_parameter}...')

    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()

        # 训练
        train()
        # 验证
        val_loss, _, _ = evaluate(test=False)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_valid:
            for i, g in enumerate(generators):
                save_model(hyp_params, g, name=f'GENERATOR_{i}')
            for i, d in enumerate(discriminators):
                save_model(hyp_params, d, name=f'DISCRIMINATOR_{i}')
            for i, gate in enumerate(gate_params):
                save_model(hyp_params, gate, name=f'GATE_{i}')
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    # 加载最佳模型进行测试
    for i, g in enumerate(generators):
        generators[i] = load_model(hyp_params, name=f'GENERATOR_{i}')
    for i, d in enumerate(discriminators):
        discriminators[i] = load_model(hyp_params, name=f'DISCRIMINATOR_{i}')
    for i, gate in enumerate(gate_params):
        gate_params[i] = load_model(hyp_params, name=f'GATE_{i}')
    model = load_model(hyp_params, name=hyp_params.name)

    _, results, truths = evaluate(test=True)

    # 评估指标计算
    if hyp_params.dataset == "mosei_senti" or hyp_params.dataset == 'mosei-bert':
        acc = eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi' or hyp_params.dataset == 'mosi-bert':
        acc = eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        acc = eval_iemocap(results, truths)
    elif hyp_params.dataset == 'sims':
        acc = eval_sims(results, truths)

    return acc