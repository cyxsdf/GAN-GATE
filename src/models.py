import torch
from torch import nn
import torch.nn.functional as F

from modules.unimf import MultimodalTransformerEncoder
from modules.transformer import TransformerEncoder
from transformers import BertTokenizer, BertModel


# GAN生成器：加入门控机制控制生成特征有效性
class Generator(nn.Module):
    def __init__(self, hyp_params, missing):
        super(Generator, self).__init__()
        self.hyp_params = hyp_params
        self.missing = missing
        self.embed_dim = hyp_params.embed_dim

        # 上下文编码器
        self.context_encoder = nn.LSTM(
            input_size=hyp_params.orig_d_l,
            hidden_size=hyp_params.hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 根据缺失模态确定输入输出维度
        if missing == 'L':
            self.input_dim = hyp_params.orig_d_a + (
                hyp_params.orig_d_v if hyp_params.dataset not in ['meld_senti', 'meld_emo'] else 0)
            self.output_dim = hyp_params.orig_d_l
        elif missing == 'A':
            self.input_dim = hyp_params.orig_d_l + (
                hyp_params.orig_d_v if hyp_params.dataset not in ['meld_senti', 'meld_emo'] else 0)
            self.output_dim = hyp_params.orig_d_a
        elif missing == 'V':
            self.input_dim = hyp_params.orig_d_l + hyp_params.orig_d_a
            self.output_dim = hyp_params.orig_d_v
        else:
            raise ValueError('Unknown missing modality type')

        # 生成器网络主体
        self.proj_in = nn.Linear(self.input_dim, self.embed_dim)
        self.transformer = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=hyp_params.num_heads,
            layers=hyp_params.trans_layers,
            attn_dropout=hyp_params.attn_dropout,
            relu_dropout=hyp_params.relu_dropout,
            res_dropout=hyp_params.res_dropout
        )

        # 修复门控层维度：输入为生成特征+原始投影特征（均为embed_dim）
        self.gate = nn.Sequential(
            nn.Linear(self.embed_dim + self.embed_dim, 1),  # 输出维度改为1（门控系数）
            nn.Sigmoid()  # 门控系数(0~1)
        )

        self.proj_out = nn.Linear(self.embed_dim, self.output_dim)

        # 位置和模态类型嵌入
        self.position_embeddings = nn.Embedding(100, self.embed_dim)  # 最大序列长度100
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

    def forward(self, src, phase='train', eval_start=False):
        # src为可用模态特征拼接 [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = src.shape

        # 原始输入投影到嵌入维度
        src_proj = F.dropout(F.relu(self.proj_in(src)), p=0.1, training=self.training)

        # Transformer特征提取
        x = src_proj.transpose(0, 1)  # [seq_len, batch_size, embed_dim]

        # 添加位置和模态嵌入
        pos_ids = torch.arange(seq_len, device=src.device).unsqueeze(1).expand(-1, batch_size)
        pos_embeds = self.position_embeddings(pos_ids)
        modal_type = 0 if self.missing != 'L' else 1  # 区分输入模态类型
        modal_embeds = self.modal_type_embeddings(torch.full_like(pos_ids, modal_type))
        x = x + pos_embeds + modal_embeds

        # Transformer编码
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]

        # 门控机制：动态平衡生成特征与原始输入特征
        # 拼接后维度：[batch_size, seq_len, embed_dim + embed_dim]
        gate = self.gate(torch.cat([x, src_proj], dim=-1))  # 门控系数维度：[batch_size, seq_len, 1]
        x = gate * x + (1 - gate) * src_proj  # 门控融合（广播机制生效）

        # 最终输出投影
        output = self.proj_out(x)
        return output


# GAN判别器：加入门控机制增强特征区分能力
class Discriminator(nn.Module):
    def __init__(self, hyp_params, missing):
        super(Discriminator, self).__init__()
        self.hyp_params = hyp_params

        # 根据缺失模态确定输入维度
        if missing == 'L':
            self.input_dim = hyp_params.orig_d_l
        elif missing == 'A':
            self.input_dim = hyp_params.orig_d_a
        elif missing == 'V':
            self.input_dim = hyp_params.orig_d_v
        else:
            raise ValueError('Unknown missing modality type')

        # 新增门控层：过滤无效特征
        self.feature_gate = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.Sigmoid()
        )

        # 判别器主体网络
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x为模态特征 [batch_size, seq_len, input_dim]

        # 门控过滤无效特征
        gate = self.feature_gate(x)  # [batch_size, seq_len, input_dim]
        x = x * gate  # 只保留重要特征

        # 序列维度平均池化
        x = x.mean(dim=1)  # [batch_size, input_dim]
        return self.model(x)


class TRANSLATEModel(nn.Module):
    def __init__(self, hyp_params, missing=None):
        """原始Translate模型（保持不变）"""
        super(TRANSLATEModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.l_len, self.a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
            self.v_len, self.orig_d_v = 0, 0
        else:
            self.l_len, self.a_len, self.v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.trans_layers = hyp_params.trans_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.trans_dropout = hyp_params.trans_dropout
        self.modalities = hyp_params.modalities  # 输入模态
        self.missing = missing  # 缺失模态

        self.position_embeddings = nn.Embedding(max(self.l_len, self.a_len, self.v_len), self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        self.multi = nn.Parameter(torch.Tensor(1, self.embed_dim))
        nn.init.xavier_uniform_(self.multi)

        # 转换模块
        self.translator = TransformerEncoder(embed_dim=self.embed_dim,
                                             num_heads=self.num_heads,
                                             lens=(self.l_len, self.a_len, self.v_len),
                                             layers=self.trans_layers,
                                             modalities=self.modalities,
                                             missing=self.missing,
                                             attn_dropout=self.attn_dropout,
                                             relu_dropout=self.relu_dropout,
                                             res_dropout=self.res_dropout)

        # 投影模块
        if 'L' in self.modalities or self.missing == 'L':
            self.proj_l = nn.Linear(self.orig_d_l, self.embed_dim)
        if 'A' in self.modalities or self.missing == 'A':
            self.proj_a = nn.Linear(self.orig_d_a, self.embed_dim)
        if 'V' in self.modalities or self.missing == 'V':
            self.proj_v = nn.Linear(self.orig_d_v, self.embed_dim)

        if self.missing == 'L':
            self.out = nn.Linear(self.embed_dim, self.orig_d_l)
        elif self.missing == 'A':
            self.out = nn.Linear(self.embed_dim, self.orig_d_a)
        elif self.missing == 'V':
            self.out = nn.Linear(self.embed_dim, self.orig_d_v)
        else:
            raise ValueError('Unknown missing modality type')

    def forward(self, src, tgt, phase='train', eval_start=False):
        """原始前向传播逻辑（保持不变）"""
        if self.modalities == 'L':
            if self.missing == 'A':
                x_l, x_a = src, tgt
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)  # (seq, batch, embed_dim)
                x_a = x_a.transpose(0, 1)
            elif self.missing == 'V':
                x_l, x_v = src, tgt
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)
                x_v = x_v.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            if self.missing == 'L':
                x_a, x_l = src, tgt
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_l = x_l.transpose(0, 1)
            elif self.missing == 'V':
                x_a, x_v = src, tgt
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_v = x_v.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            if self.missing == 'L':
                x_v, x_l = src, tgt
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_l = x_l.transpose(0, 1)
            elif self.missing == 'A':
                x_v, x_a = src, tgt
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_a = x_a.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            (x_l, x_a), x_v = src, tgt
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
        elif self.modalities == 'LV':
            (x_l, x_v), x_a = src, tgt
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
        elif self.modalities == 'AV':
            (x_a, x_v), x_l = src, tgt
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_l = x_l.transpose(0, 1)
        else:
            raise ValueError('Unknown modalities type')

        # 模态类型嵌入
        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2

        # 准备[Uni]或[Bi] token
        batch_size = tgt.shape[0]
        multi = self.multi.unsqueeze(1).repeat(1, batch_size, 1)

        if phase != 'test':
            if self.missing == 'L':
                x_l = torch.cat((multi, x_l[:-1]), dim=0)
            elif self.missing == 'A':
                x_a = torch.cat((multi, x_a[:-1]), dim=0)
            elif self.missing == 'V':
                x_v = torch.cat((multi, x_v[:-1]), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        else:
            if eval_start:
                if self.missing == 'L':
                    x_l = multi  # 使用[Uni]作为生成起始
                elif self.missing == 'A':
                    x_a = multi
                elif self.missing == 'V':
                    x_v = multi
                else:
                    raise ValueError('Unknown missing modality type')
            else:
                if self.missing == 'L':
                    x_l = torch.cat((multi, x_l), dim=0)
                elif self.missing == 'A':
                    x_a = torch.cat((multi, x_a), dim=0)
                elif self.missing == 'V':
                    x_v = torch.cat((multi, x_v), dim=0)
                else:
                    raise ValueError('Unknown missing modality type')

        # 位置嵌入和模态类型嵌入
        if 'L' in self.modalities or self.missing == 'L':
            x_l_pos_ids = torch.arange(x_l.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            l_pos_embeds = self.position_embeddings(x_l_pos_ids)
            l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_l_pos_ids, L_MODAL_TYPE_IDX))
            l_embeds = l_pos_embeds + l_modal_type_embeds
            x_l = x_l + l_embeds
            x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)
        if 'A' in self.modalities or self.missing == 'A':
            x_a_pos_ids = torch.arange(x_a.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            a_pos_embeds = self.position_embeddings(x_a_pos_ids)
            a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_a_pos_ids, A_MODAL_TYPE_IDX))
            a_embeds = a_pos_embeds + a_modal_type_embeds
            x_a = x_a + a_embeds
            x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)
        if 'V' in self.modalities or self.missing == 'V':
            x_v_pos_ids = torch.arange(x_v.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            v_pos_embeds = self.position_embeddings(x_v_pos_ids)
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_v_pos_ids, V_MODAL_TYPE_IDX))
            v_embeds = v_pos_embeds + v_modal_type_embeds
            x_v = x_v + v_embeds
            x_v = F.dropout(x_v, p=self.embed_dropout, training=self.training)

        # 拼接输入并进行转换
        if self.modalities == 'L':
            if self.missing == 'A':
                x = torch.cat((x_l, x_a), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_l, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            if self.missing == 'L':
                x = torch.cat((x_a, x_l), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_a, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            if self.missing == 'L':
                x = torch.cat((x_v, x_l), dim=0)
            elif self.missing == 'A':
                x = torch.cat((x_v, x_a), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            x = torch.cat((x_l, x_a, x_v), dim=0)
        elif self.modalities == 'LV':
            x = torch.cat((x_l, x_v, x_a), dim=0)
        elif self.modalities == 'AV':
            x = torch.cat((x_a, x_v, x_l), dim=0)
        else:
            raise ValueError('Unknown modalities type')

        output = self.translator(x)

        # 提取输出
        if self.modalities == 'L':
            output = output[self.l_len:].transpose(0, 1)  # (batch, seq, embed_dim)
        elif self.modalities == 'A':
            output = output[self.a_len:].transpose(0, 1)
        elif self.modalities == 'V':
            output = output[self.v_len:].transpose(0, 1)
        elif self.modalities == 'LA':
            output = output[self.l_len + self.a_len:].transpose(0, 1)
        elif self.modalities == 'LV':
            output = output[self.l_len + self.v_len:].transpose(0, 1)
        elif self.modalities == 'AV':
            output = output[self.a_len + self.v_len:].transpose(0, 1)
        else:
            raise ValueError('Unknown modalities type')

        output = self.out(output)
        return output


class UNIMFModel(nn.Module):
    def __init__(self, hyp_params):
        """原始UniMF模型（保持不变）"""
        super(UNIMFModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.orig_l_len, self.orig_a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
        else:
            self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.l_kernel_size = hyp_params.l_kernel_size
        self.a_kernel_size = hyp_params.a_kernel_size
        if hyp_params.dataset != 'meld_senti' and hyp_params.dataset != 'meld_emo':
            self.v_kernel_size = hyp_params.v_kernel_size
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.multimodal_layers = hyp_params.multimodal_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.modalities = hyp_params.modalities
        self.dataset = hyp_params.dataset
        self.language = hyp_params.language
        self.use_bert = hyp_params.use_bert

        self.distribute = hyp_params.distribute

        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.cls_len = 33
        else:
            self.cls_len = 1
        self.cls = nn.Parameter(torch.Tensor(self.cls_len, self.embed_dim))
        nn.init.xavier_uniform_(self.cls)

        # 计算卷积后的序列长度
        self.l_len = self.orig_l_len - self.l_kernel_size + 1
        self.a_len = self.orig_a_len - self.a_kernel_size + 1
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.v_len = self.orig_v_len - self.v_kernel_size + 1

        output_dim = hyp_params.output_dim

        # BERT模型
        if self.use_bert:
            self.text_model = BertTextEncoder(language=hyp_params.language, use_finetune=True)

        # 1. 时间卷积块
        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=self.l_kernel_size)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=self.a_kernel_size)
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=self.v_kernel_size)
        if 'meld' in self.dataset:
            self.proj_cls = nn.Conv1d(self.orig_d_l + self.orig_d_a, self.embed_dim, kernel_size=1)

        # 2. GRU编码器
        self.t = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        self.a = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.v = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)

        # 3. 多模态融合块
        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.position_embeddings = nn.Embedding(max(self.cls_len, self.l_len, self.a_len), self.embed_dim)
        else:
            self.position_embeddings = nn.Embedding(max(self.l_len, self.a_len, self.v_len), self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        # 3.2. UniMF
        self.unimf = MultimodalTransformerEncoder(embed_dim=self.embed_dim,
                                                  num_heads=self.num_heads,
                                                  layers=self.multimodal_layers,
                                                  lens=(self.cls_len, self.l_len, self.a_len),
                                                  modalities=self.modalities,
                                                  attn_dropout=self.attn_dropout,
                                                  relu_dropout=self.relu_dropout,
                                                  res_dropout=self.res_dropout)

        # 4. 投影层
        combined_dim = self.embed_dim
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def forward(self, x_l, x_a, x_v=None):
        """原始前向传播逻辑（保持不变）"""
        if self.distribute:
            self.t.flatten_parameters()
            self.a.flatten_parameters()
            if x_v is not None:
                self.v.flatten_parameters()

        # 模态类型嵌入
        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2
        MULTI_MODAL_TYPE_IDX = 3

        # 准备[CLS] token
        batch_size = x_l.shape[0]
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            cls = self.cls.unsqueeze(1).repeat(1, batch_size, 1)
        else:
            cls = self.proj_cls(torch.cat((x_l, x_a), dim=-1).transpose(1, 2)).permute(2, 0, 1)

        # 位置嵌入和模态类型嵌入
        cls_pos_ids = torch.arange(self.cls_len, device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_l_pos_ids = torch.arange(self.l_len, device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_a_pos_ids = torch.arange(self.a_len, device=x_a.device).unsqueeze(1).expand(-1, batch_size)
        if x_v is not None:
            h_v_pos_ids = torch.arange(self.v_len, device=x_v.device).unsqueeze(1).expand(-1, batch_size)

        cls_pos_embeds = self.position_embeddings(cls_pos_ids)
        h_l_pos_embeds = self.position_embeddings(h_l_pos_ids)
        h_a_pos_embeds = self.position_embeddings(h_a_pos_ids)
        if x_v is not None:
            h_v_pos_embeds = self.position_embeddings(h_v_pos_ids)

        cls_modal_type_embeds = self.modal_type_embeddings(torch.full_like(cls_pos_ids, MULTI_MODAL_TYPE_IDX))
        l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_l_pos_ids, L_MODAL_TYPE_IDX))
        a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_a_pos_ids, A_MODAL_TYPE_IDX))
        if x_v is not None:
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_v_pos_ids, V_MODAL_TYPE_IDX))

        # 投影文本/视觉/音频特征并压缩序列长度
        if self.use_bert:
            x_l = self.text_model(x_l)

        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        if x_v is not None:
            x_v = x_v.transpose(1, 2)

        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        if x_v is not None:
            proj_x_v = self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        if x_v is not None:
            proj_x_v = proj_x_v.permute(2, 0, 1)

        # 使用GRU编码
        h_l, _ = self.t(proj_x_l)
        h_a, _ = self.a(proj_x_a)
        if x_v is not None:
            h_v, _ = self.v(proj_x_v)

        # 添加位置和模态类型嵌入
        cls_embeds = cls_pos_embeds + cls_modal_type_embeds
        l_embeds = h_l_pos_embeds + l_modal_type_embeds
        a_embeds = h_a_pos_embeds + a_modal_type_embeds
        if x_v is not None:
            v_embeds = h_v_pos_embeds + v_modal_type_embeds
        cls = cls + cls_embeds
        h_l = h_l + l_embeds
        h_a = h_a + a_embeds
        if x_v is not None:
            h_v = h_v + v_embeds
        h_l = F.dropout(h_l, p=self.embed_dropout, training=self.training)
        h_a = F.dropout(h_a, p=self.embed_dropout, training=self.training)
        if x_v is not None:
            h_v = F.dropout(h_v, p=self.embed_dropout, training=self.training)

        # 多模态融合
        if x_v is not None:
            x = torch.cat((cls, h_l, h_a, h_v), dim=0)
        else:
            x = torch.cat((cls, h_l, h_a), dim=0)
        x = self.unimf(x)

        if x_v is not None:
            last_hs = x[0]  # 获取[CLS] token用于预测
        else:
            last_hs = x[:self.cls_len]  # 获取[CLS] tokens用于预测

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        if x_v is None:
            output = output.transpose(0, 1)
        return output, last_hs


class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        """BERT文本编码器（保持不变）"""
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('pretrained_bert/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_cn')
            self.model = model_class.from_pretrained('pretrained_bert/bert_cn')

        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        return last_hidden_states.squeeze()

    def forward(self, text):
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]
        return last_hidden_states