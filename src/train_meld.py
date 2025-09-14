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


####################################################################
#
# GAN Modules for Missing Modality Generation with Gating Mechanism
#
####################################################################

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, seq_len=33):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 特征提取网络
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(seq_len),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(seq_len),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

        # 门控机制：融合原始输入与生成特征（使用固定投影层避免动态定义）
        self.proj = nn.Linear(input_dim, output_dim)  # 固定投影层
        self.gate = nn.Sequential(
            nn.Linear(output_dim + output_dim, output_dim),  # 输入为投影后特征+生成特征
            nn.Sigmoid()  # 门控系数(0~1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        gen_feature = self.layers(x)  # 生成特征 (batch_size, seq_len, output_dim)

        # 门控融合：先将原始输入投影到输出维度（固定层避免动态计算）
        x_proj = self.proj(x)  # (batch_size, seq_len, output_dim)

        # 拼接生成特征和投影后的原始特征，计算门控系数
        gate = self.gate(torch.cat([gen_feature, x_proj], dim=-1))  # (batch_size, seq_len, output_dim)
        output = gate * gen_feature + (1 - gate) * x_proj  # 门控加权融合

        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, seq_len=33, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        # 门控层：突出重要特征
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        # 判别器网络
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # 门控过滤无效特征
        gate = self.gate(x)  # (batch_size, seq_len, input_dim)
        x = x * gate  # 只保留重要特征

        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)  # 展平序列维度
        x = self.layers(x)
        return x.view(batch_size, seq_len, 1)  # 恢复序列维度


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    # 初始化GAN组件
    if hyp_params.modalities != 'LA':
        if hyp_params.modalities == 'L':
            # 文本模态输入，生成音频模态
            input_dim = hyp_params.orig_d_l  # 文本特征维度（600）
            output_dim = hyp_params.orig_d_a  # 音频特征维度（300）
            generator = Generator(input_dim, output_dim, seq_len=hyp_params.l_len)
            discriminator = Discriminator(output_dim, seq_len=hyp_params.a_len)
            # 初始化特征融合门控（可训练参数）
            gate_audio = nn.Sequential(
                nn.Linear(output_dim, 1),  # 输入为音频特征维度（300）
                nn.Sigmoid()
            ).cuda() if hyp_params.use_cuda else nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())
        elif hyp_params.modalities == 'A':
            # 音频模态输入，生成文本模态
            input_dim = hyp_params.orig_d_a  # 音频特征维度（300）
            output_dim = hyp_params.orig_d_l  # 文本特征维度（600）
            generator = Generator(input_dim, output_dim, seq_len=hyp_params.a_len)
            discriminator = Discriminator(output_dim, seq_len=hyp_params.l_len)
            # 初始化特征融合门控（可训练参数）
            gate_text = nn.Sequential(
                nn.Linear(output_dim, 1),  # 输入为文本特征维度（600）
                nn.Sigmoid()
            ).cuda() if hyp_params.use_cuda else nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        # 优化器设置（包含门控参数优化）
        gen_params = list(generator.parameters())
        if hyp_params.modalities == 'L':
            gen_params.extend(gate_audio.parameters())
        else:
            gen_params.extend(gate_text.parameters())

        gen_optimizer = optim.Adam(gen_params, lr=hyp_params.gen_lr, betas=(0.5, 0.999))
        dis_optimizer = optim.Adam(discriminator.parameters(), lr=hyp_params.dis_lr, betas=(0.5, 0.999))
        gan_criterion = nn.BCELoss()  # GAN损失函数
        recon_criterion = nn.MSELoss()  # 重构损失函数

    # 主任务模型
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()
        if hyp_params.modalities != 'LA':
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            if hyp_params.modalities == 'L':
                gate_audio = gate_audio.cuda()
            else:
                gate_text = gate_text.cuda()

    # 主模型优化器
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

    # 配置训练参数
    if hyp_params.modalities != 'LA':
        settings = {
            'model': model,
            'generator': generator,
            'discriminator': discriminator,
            'gen_optimizer': gen_optimizer,
            'dis_optimizer': dis_optimizer,
            'gan_criterion': gan_criterion,
            'recon_criterion': recon_criterion,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler,
            'gate_audio': gate_audio if hyp_params.modalities == 'L' else None,
            'gate_text': gate_text if hyp_params.modalities == 'A' else None
        }
    elif hyp_params.modalities == 'LA':
        settings = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler
        }
    else:
        raise ValueError('Unknown modalities type')

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

    # GAN组件
    if hyp_params.modalities != 'LA':
        generator = settings['generator']
        discriminator = settings['discriminator']
        gen_optimizer = settings['gen_optimizer']
        dis_optimizer = settings['dis_optimizer']
        gan_criterion = settings['gan_criterion']
        recon_criterion = settings['recon_criterion']
        gate_audio = settings['gate_audio']
        gate_text = settings['gate_text']
    else:
        generator = None
        discriminator = None
        gate_audio = None
        gate_text = None

    def train():
        model.train()
        epoch_loss = 0
        if hyp_params.modalities != 'LA':
            generator.train()
            discriminator.train()
            if gate_audio:
                gate_audio.train()
            if gate_text:
                gate_text.train()

        for i_batch, (audio, text, masks, labels) in enumerate(train_loader):
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

            # 应用掩码
            masks_text = masks.unsqueeze(-1).expand(-1, hyp_params.l_len, hyp_params.orig_d_l)
            masks_audio = masks.unsqueeze(-1).expand(-1, hyp_params.a_len, hyp_params.orig_d_a)
            text = text * masks_text
            audio = audio * masks_audio
            batch_size = text.size(0)

            # 主模型梯度清零
            model.zero_grad()

            if hyp_params.modalities != 'LA':
                # GAN梯度清零
                generator.zero_grad()
                discriminator.zero_grad()
                if gate_audio:
                    gate_audio.zero_grad()
                if gate_text:
                    gate_text.zero_grad()

                # 生成缺失模态
                if hyp_params.modalities == 'L':
                    # 文本生成音频 (输入600维 -> 输出300维)
                    fake_audio = generator(text)
                    real_data = audio
                    gen_data = fake_audio

                    # 门控融合真实音频与生成音频
                    gate = gate_audio(real_data)  # 计算门控系数 (batch, seq_len, 1)
                    weighted_audio = gate * real_data + (1 - gate) * fake_audio  # 动态平衡
                elif hyp_params.modalities == 'A':
                    # 音频生成文本 (输入300维 -> 输出600维)
                    fake_text = generator(audio)
                    real_data = text
                    gen_data = fake_text

                    # 门控融合真实文本与生成文本
                    gate = gate_text(real_data)  # 计算门控系数 (batch, seq_len, 1)
                    weighted_text = gate * real_data + (1 - gate) * fake_text  # 动态平衡

                # 1. 训练判别器
                # 真实数据标签和预测
                real_label = torch.ones_like(real_data[..., :1])  # (batch, seq_len, 1)
                fake_label = torch.zeros_like(real_data[..., :1])

                real_pred = discriminator(real_data)
                dis_loss_real = gan_criterion(real_pred, real_label)

                # 生成数据预测（detach避免更新生成器）
                fake_pred = discriminator(gen_data.detach())
                dis_loss_fake = gan_criterion(fake_pred, fake_label)

                dis_loss = (dis_loss_real + dis_loss_fake) * 0.5
                dis_loss.backward(retain_graph=True)
                dis_optimizer.step()

                # 2. 训练生成器
                # GAN损失（希望生成数据被判别为真实）
                gen_pred = discriminator(gen_data)
                gen_loss_gan = gan_criterion(gen_pred, real_label)

                # 重构损失（生成数据与真实数据的相似度）
                gen_loss_recon = recon_criterion(gen_data, real_data)

                # 联合损失
                gen_loss = gen_loss_gan * hyp_params.gan_weight + gen_loss_recon

            # 3. 主任务训练
            if hyp_params.modalities != 'LA':
                if hyp_params.modalities == 'L':
                    # 使用门控融合后的音频特征
                    preds, _ = model(text, weighted_audio)
                elif hyp_params.modalities == 'A':
                    # 使用门控融合后的文本特征
                    preds, _ = model(weighted_text, audio)
            else:
                preds, _ = model(text, audio)

            # 主任务损失
            task_loss = criterion(preds.transpose(1, 2), labels)

            # 总损失计算
            if hyp_params.modalities != 'LA':
                total_loss = task_loss + gen_loss
            else:
                total_loss = task_loss

            total_loss.backward()

            # 参数更新
            if hyp_params.modalities != 'LA':
                gen_optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            epoch_loss += total_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(test=False):
        model.eval()
        if hyp_params.modalities != 'LA':
            generator.eval()
            if gate_audio:
                gate_audio.eval()
            if gate_text:
                gate_text.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []
        mask = []

        with torch.no_grad():
            for i_batch, (audio, text, masks, labels) in enumerate(loader):
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

                # 应用掩码
                masks_text = masks.unsqueeze(-1).expand(-1, hyp_params.l_len, hyp_params.orig_d_l)
                masks_audio = masks.unsqueeze(-1).expand(-1, hyp_params.a_len, hyp_params.orig_d_a)
                text = text * masks_text
                audio = audio * masks_audio
                batch_size = text.size(0)

                # 生成缺失模态（评估阶段）
                if hyp_params.modalities != 'LA':
                    if hyp_params.modalities == 'L':
                        fake_audio = generator(text)
                        # 门控融合
                        gate = gate_audio(audio)
                        weighted_audio = gate * audio + (1 - gate) * fake_audio
                        preds, _ = model(text, weighted_audio)
                    elif hyp_params.modalities == 'A':
                        fake_text = generator(audio)
                        # 门控融合
                        gate = gate_text(text)
                        weighted_text = gate * text + (1 - gate) * fake_text
                        preds, _ = model(weighted_text, audio)
                else:
                    preds, _ = model(text, audio)

                # 计算损失
                loss = criterion(preds.transpose(1, 2), labels)
                total_loss += loss.item() * batch_size

                # 收集结果
                results.append(preds)
                truths.append(labels)
                mask.append(masks)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        mask = torch.cat(mask)
        return avg_loss, results, truths, mask

    # 模型参数统计
    if hyp_params.modalities != 'LA':
        gen_params = sum([param.nelement() for param in generator.parameters()])
        dis_params = sum([param.nelement() for param in discriminator.parameters()])
        gate_params = sum([param.nelement() for param in gate_audio.parameters()]) if gate_audio else \
            sum([param.nelement() for param in gate_text.parameters()])
        print(
            f'Generator Parameters: {gen_params}, Discriminator Parameters: {dis_params}, Gating Parameters: {gate_params}...')
    mum_params = sum([param.nelement() for param in model.parameters()])
    print(f'Multimodal Understanding Model Parameters: {mum_params}...')

    # 训练主循环
    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()

        # 训练
        train_loss = train()

        # 验证
        val_loss, _, _, _ = evaluate(test=False)
        end = time.time()
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_valid:
            best_valid = val_loss
            save_model(hyp_params, model, name=hyp_params.name)
            if hyp_params.modalities != 'LA':
                save_model(hyp_params, generator, name='GENERATOR')
                save_model(hyp_params, discriminator, name='DISCRIMINATOR')
                if gate_audio:
                    save_model(hyp_params, gate_audio, name='GATE_AUDIO')
                if gate_text:
                    save_model(hyp_params, gate_text, name='GATE_TEXT')

    # 加载最佳模型并测试
    model = load_model(hyp_params, name=hyp_params.name)
    if hyp_params.modalities != 'LA':
        generator = load_model(hyp_params, name='GENERATOR')
        if gate_audio:
            gate_audio = load_model(hyp_params, name='GATE_AUDIO')
        if gate_text:
            gate_text = load_model(hyp_params, name='GATE_TEXT')
    _, results, truths, mask = evaluate(test=True)
    acc = eval_meld(results, truths, mask)

    return acc