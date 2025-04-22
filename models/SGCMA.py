import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录到Python路径

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from math import sqrt
import logging
import os
import time
# from peft import LoraConfig, TaskType, get_peft_model


from layers.StandardNorm import Normalize



class WikiHMM(nn.Module):
    # def __init__(self, hidden_state_num, colsum_threshold, diag_max=0.7, token_num=50003):
    def __init__(self, hidden_state_num, diag_max=0.7, token_num=50003):
        super(WikiHMM, self).__init__()
        self.hidden_state_num = hidden_state_num
        # self.colsum_threshold = colsum_threshold  
        self.diag_max = diag_max

        self.init_logits = nn.Parameter(torch.zeros(hidden_state_num))  # 均匀初始化
        self.transition_logits = nn.Parameter(torch.randn(hidden_state_num, hidden_state_num)*0.1)
        self.emission_logits = nn.Parameter(torch.randn(hidden_state_num, token_num)*0.1)

    def forward(self, text_input):
        batch_size = text_input.shape[0]
        seq_length = text_input.shape[1]

        # === 1. 将logits转换成概率 ===
        # text_pi = F.softmax(self.init_logits, dim=0)
        # text_A = F.softmax(self.transition_logits, dim=1)
        # text_B = F.softmax(self.emission_logits, dim=1)
        
        # logger.debug(f"text_pi:{text_pi}")
        # logger.debug(f"text_A: {text_A}")
        # logger.debug(f"text_B: {text_B}")
        # 计算状态转移矩阵、发射矩阵信息熵

        # === 2. 对数前向算法 （在数值稳定的对数空间进行，因为概率相乘数值会越来越小） ===
        log_pi = self.init_logits - torch.logsumexp(self.init_logits, dim=0, keepdim=True)
        log_A = self.transition_logits - torch.logsumexp(self.transition_logits, dim=1, keepdim=True)
        log_B = self.emission_logits - torch.logsumexp(self.emission_logits, dim=1, keepdim=True)

        log_likelihoods = torch.full((batch_size, seq_length), -float('inf'), device = text_input.device)

        # 对于第一个观测计算初始对数概率和似然
        log_alpha = log_pi + log_B[:, text_input[:,0]].T  # [batch_size, hidden_state_num]
        log_likelihoods[:, 0] = torch.logsumexp(log_alpha, dim=1)

        for t in range(1, seq_length):     # log_alpha:之前所有token和当前状态的概率

            # 先计算状态转移项，log_alpha_prev + log_A 
            transition_term = log_alpha.unsqueeze(2) + log_A.unsqueeze(0)  # [batch_size, hidden_state_num, hidden_state_num]   
            
            log_alpha = torch.logsumexp(transition_term, dim=1) + log_B[:, text_input[:, t]].T # [batch_size, hidden_state_num]
            
            log_likelihoods[:, t] = torch.logsumexp(log_alpha, dim=1)   # [batch_size,]

    
        # === 3. 计算批量负对数似然损失 ===
        batch_indices = torch.arange(batch_size, device=text_input.device) # 为每个batch生成索引序号(0,1,2...batch_size-1)
        log_likelihood = log_likelihoods[batch_indices, seq_length-1]
        likelihood_loss = -log_likelihood.mean()

        info_mat = - torch.exp(log_A) * log_A # [hidden_state_num, hidden_state_num]

        # === 4. Connectiveity Constraint ===

        mean_prob_trans2cls = torch.mean(torch.exp(log_A), dim=0) # [hidden_state_num]
        cnct_const = - torch.sum(mean_prob_trans2cls * torch.log(mean_prob_trans2cls + 1e-10)) # 要让这个熵尽量大

        # === 5. 正则化项 ===
        # (1) 计算状态转移矩阵每行的熵
        transition_entropy = torch.sum(info_mat, dim=1)  # [hidden_state_num]
        transition_entropy_loss = transition_entropy.mean()  # 熵的平均值

        # (2) maintain a large self-transition probability
        self_trans_const = 100 / torch.norm(torch.diag(torch.exp(log_A)), p=2)
        diag_reg = torch.sum(F.relu(torch.diag(torch.exp(log_A)) - self.diag_max).pow(2))


        return torch.exp(log_pi), torch.exp(log_A), torch.exp(log_B), likelihood_loss, transition_entropy_loss, cnct_const, diag_reg, self_trans_const

# FlattenHead：展平并输出预测结果
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.1, linear_layer=3):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        if linear_layer == 1:
            self.linear = nn.Linear(nf, target_window)

        else:
            ls = [nn.Linear(nf, 512), nn.GELU(), nn.Dropout(head_dropout)]
            for _ in range(linear_layer - 2):
                ls.append(nn.Linear(512, 512))
                ls.append(nn.GELU())
                ls.append(nn.Dropout(head_dropout))
            ls.append(nn.Linear(512, target_window))
            self.linear = nn.Sequential(*ls)

    def forward(self, x):  # x: [bs, nvars, d_ff, patch_num]
        x = self.flatten(x)  # x: [bs, nvars, d_ff * patch_num]
        x = self.linear(x)   # x: [bs, nvars, target_window]
        return x             # x: [bs, nvars, target_window]

class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output

"""划分patch, 通道独立"""
class Patching(nn.Module):
    """
    输入: 就是一个原始样本时间窗口 [bs, n_vars, L]
    输出: 经过分割patch, 通道独立的 [bs*n_vars, patch_num, patch_len], 这样处理之后可以送到TSCluster中定义的bkb(Transformer Encoder)中进行特征提取
    """
    def __init__(self, patch_len, stride):
        super(Patching, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))      # self.padding_patch_layer: 使用 ReplicationPad1d 在 patch_num 即时间维度上填充数据，填充长度为 (0, stride)，即只在右侧填充 stride 个时间步，确保分割时数据长度足够

    def forward(self, x):
        """
        :param x: (bs, n_vars, L)
        :return: (bs*n_vars, (seq_len + stride - patch_len) // stride + 1, patch_len)
        """
        # do patching
        n_vars = x.shape[1]
        # first pad "stride" length after the sequence  
        x = self.padding_patch_layer(x)   # x: [bs, n_vars, L+stride]
        # then unfold the sequence into patches
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # x: [bs, n_vars, (seq_len + stride - patch_len) // stride + 1, patch_len]
        
        # joint the batch and variable dimensions together for channel independent
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # x: [bs * n_vars, patch_num, patch_len]

        return x, n_vars

"""提供基于 patch 自身特征(embedding)的聚类概率，反映时序数据的内在结构。"""
class ClusterCenters(nn.Module):
    """
    输入: 一个批次的 patch embedding, 形状为 [bs*n_vars, patch_num, embedding_dim]
    输出: 每个 patch 属于各个聚类中心的概率分布, 形状为 [bs*n_vars, patch_num, cluster_num]
    """
    def __init__(self, embedding_dim, cluster_num, temperature=10):
        super(ClusterCenters, self).__init__()
        self.embedding_dim = embedding_dim
        self.cluster_num = cluster_num
        self.temperature = temperature
        self.mus = nn.Parameter(torch.randn(cluster_num, embedding_dim))  

    def forward(self, x):  # x: [bs*n_vars, patch_num, embedding_dim]
        x_expanded = x.unsqueeze(2)   # x_expanded: [B, patch_num, 1, embedding_dim]
        mus = F.tanh(self.mus)  
        mus_expanded = mus.unsqueeze(0).unsqueeze(0)  # [1, 1, cluster_num, embedding_dim]
        # 计算每个 patch 到每个聚类中心的平方距离
        mu_diff = (x_expanded - mus_expanded) ** 2   # [B, patch_num, cluster_num, embedding_dim]，这是在embedding上逐个维度的差
        mu_diff = mu_diff.sum(dim=-1)
        mu_diff_exp = torch.exp(-mu_diff / self.temperature)  
        z_p = mu_diff_exp / mu_diff_exp.sum(dim=-1, keepdim=True) # 距离越小概率越大
        return z_p   # [B, patch_num, cluster_num]

class TSCluster(nn.Module):
    """
    输入: 一个批次的patch, 形状为[bs*n_vars, patch_num, patch_len]
    输出: 一个批次每个patch的聚类概率, 形状为[bs*n_vars, patch_num, cluster_num]
          熵损失 entropy_loss
          中间输出特征提取结果emb_all:[bs*n_vars, patch_num, hidden_dim], 以便后续进行crossattention使用  
    """
    def __init__(self, bkb_kargs, hidden_dim, cluster_num, patch_num, train_trans=False):
        super(TSCluster, self).__init__()
        self.cluster_num = cluster_num
        self.patch_num = patch_num
        self.train_trans = train_trans

        """实例化一个 Transformer Encoder: self.bkb"""
        self.bkb = nn.TransformerEncoder(**bkb_kargs)

        """实例化 ClusterCenters: self.cluster_centers"""
        self.cluster_centers = ClusterCenters(hidden_dim, cluster_num)

        # 可训练HMM状态转移矩阵、初始状态分布 
        self.ts_init_logits = nn.Parameter(torch.randn(cluster_num))
        self.ts_transition_logits = nn.Parameter(torch.randn(cluster_num, cluster_num))
    
    def _init_trans_pi(self, init_logits, transition_logits):
        self.ts_init_logits.data = init_logits
        self.ts_transition_logits.data = transition_logits

    def forward(self, input_series, init_logits=None, transition_logits=None):   # input_series: [bs*n_vars, patch_num, patch_len]
        if self.train_trans:
            prob_pi = torch.softmax(self.ts_init_logits, dim=0)
            prob_A = torch.softmax(self.ts_transition_logits, dim=1)
        else:
            prob_pi = torch.softmax(init_logits, dim=0)
            prob_A = torch.softmax(transition_logits, dim=1)
        emb_all = self.bkb(input_series)  # 提取特征 [bs*n_vars, patch_num, hidden_dim]
        logits_all = self.cluster_centers(emb_all)  # 聚类概率分布 [bs*n_vars, patch_num, cluster_num] 
        logit_t = torch.tile(prob_pi, (logits_all.shape[0], 1))  # self.ts_pi 复制到批次大小（B） [bs*n_vars, cluster_num]
        all_logit_result = []
        # 初始时间步0：直接用初始状态和第一个patch的概率分布
        logit_t = torch.log(logit_t + 1e-10) + torch.log(logits_all[:, 0] + 1e-10)  # 第 2 维（cluster_num）未指定，PyTorch 默认取整个维度，不用写logits_all[:, 0, :]
        logit_t = torch.softmax(logit_t, dim=1) # 归一化
        all_logit_result.append(logit_t)
        # 后续时间步
        for i in range(1, self.patch_num):
            logit_t = torch.matmul(logit_t, prob_A)  # 计算下一个时间步的隐藏状态分布 (bs*n_vars, cluster_num)
            logit_t = torch.log(logit_t + 1e-10) + torch.log(logits_all[:, i] + 1e-10) 
            logit_t = torch.softmax(logit_t, dim=1)  # 归一化
            all_logit_result.append(logit_t)
        cluster_probs = torch.stack(all_logit_result, dim=1)   # [bs*n_vars, patch_num, cluster_num]
        # 计算熵损失，鼓励概率分布集中

        # mean_prob_each_cluster = torch.mean(cluster_probs, dim=-2)
        # entropy_loss_each_cluster = torch.sum(mean_prob_each_cluster * torch.log(mean_prob_each_cluster + 1e-10), dim=-1).mean()
        entropy_loss = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-10), dim=-1).mean()
        return cluster_probs, emb_all, prob_pi, prob_A, entropy_loss


"""将时间序列 patch 与文本 token embeddings 进行 cross-attention 并加权融合, 100个类使用了100个CrossAttentionLayer实例"""
class TSAligner(nn.Module):
    """
    输入: TSCluster返回的cluster_probs, emb_all。cluster_probs: 聚类软标签概率, 形状[B, patch_num, cluster_num]  ts_emb: 时间序列patch嵌入, 形状[B, patch_num, d_model]  
    输出: fused_emb: [B, patch_num, d_llm] - 融合后的嵌入
    """
    def __init__(self, cluster_num, d_model, n_heads, d_llm, attention_dropout=0.1, temperature=None):
        super(TSAligner, self).__init__()
        self.cluster_num = cluster_num
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_llm = d_llm

        # 为每个聚类创建一个 CrossAttentionLayer 实例
        # self.crossattention_layers = nn.ModuleList([
        #     CrossAttentionLayer(d_model, n_heads, d_llm=d_llm, attention_dropout=attention_dropout)
        #     for _ in range(cluster_num)
        # ])
        self.crossattention_layers = CrossAttentionLayer(d_model, n_heads, cluster_num, d_llm=d_llm, attention_dropout=attention_dropout, temperature=temperature)

    def forward(self, topk_token_embeddings, ts_emb, cluster_probs):   
        # B, P, _ = ts_emb.shape  # [B, patch_num, d_model]
        # aligned_embs = []
        # 对每个聚类进行注意力对齐
        # for c in range(self.cluster_num):
        #     # 获取当前聚类的 top-k token embeddings
        #     text_emb = topk_token_embeddings[c]  # 第c个cluster的topk_token_embedding  [topk, d_llm]
        #     # 使用 CrossAttentionLayer 计算对齐嵌入
        #     aligned = self.crossattention_layers[c](ts_emb, text_emb, text_emb)  # 一段时间序列整个和一个聚类里的topk_token进行注意力计算, scores:[B, H, patch_num, topk]，得到融合文本特征的 [B, patch_num, d_llm]
        #     # 根据聚类概率加权
        #     weight = cluster_probs[:, :, [c]]  # [B, patch_num, 1]
        #     aligned_embs.append(aligned * weight)  # [B, patch_num, d_llm]

        aligned = self.crossattention_layers(ts_emb, topk_token_embeddings, topk_token_embeddings) # [B, L, c, d_llm]

        # 融合所有聚类的加权嵌入
        fused_emb = torch.sum(aligned * cluster_probs[:, :, :, None], dim=-2)  # [B, patch_num, d_llm]
        return fused_emb
    
class CrossAttentionLayer(nn.Module):
    """
    input:
        TSCluster返回的target_embedding: (B, patch_num, d_model)
        source_embedding: (S, d_llm)
        value_embedding: (S, d_llm)
    output:
        融合了文本特征且映射到d_llm的时序patch (B, patch_num, d_llm)
    """
    def __init__(self, d_model, n_heads, n_cls=100, d_keys=None, d_llm=None, attention_dropout=0.1, temperature=None):
        super(CrossAttentionLayer, self).__init__()

        self.n_cls = n_cls

        d_keys = d_keys or (d_model // n_heads)

        self.d_keys = d_keys

        self.query_projection = nn.Parameter(torch.randn(n_cls, n_heads * d_keys, d_model + 1)) # +1 用于bias
        self.key_projection = nn.Parameter(torch.randn(n_cls, n_heads * d_keys, d_llm + 1)) # +1 用于bias
        self.value_projection = nn.Parameter(torch.randn(n_cls, n_heads * d_keys, d_llm + 1)) # +1 用于bias

        # self.out_projection = nn.Linear(d_keys * n_heads, d_llm)  # 将融合了文本原型后的时序patch，映射到LLM的维度，d_model -> d_llm
        self.out_projection_w = nn.Parameter(torch.randn(n_cls, n_heads, d_keys, d_llm))
        self.out_projection_b = nn.Parameter(torch.randn(n_cls, d_llm))

        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        self.temperature = temperature or 1. / sqrt(self.d_keys)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        target_embedding: (B, L, d_model)
        source_embedding: (n_cluster, k, d_llm)
        value_embedding: (n_cluster, k, d_llm) the same as source_embedding
        """
        B, L, d_model = target_embedding.shape
        c, k, d_llm = source_embedding.shape
        H = self.n_heads
        d_keys = self.d_keys

        target_embedding = torch.einsum("bld,cgd->blcg", target_embedding, self.query_projection[...,:-1]) + self.query_projection[..., -1] # [B, L, c, H * d_keys]
        target_embedding = target_embedding.view(B, L, c, H, d_keys)
        source_embedding = torch.einsum("ckd,cgd->kcg", source_embedding, self.key_projection[...,:-1]) + self.key_projection[..., -1] # [k, c, H * d_keys]
        source_embedding = source_embedding.view(k, c, H, d_keys)
        value_embedding = torch.einsum("ckd,cgd->kcg", value_embedding, self.value_projection[...,:-1]) + self.value_projection[..., -1] # [k, c, H * d_keys]
        value_embedding = value_embedding.view(k, c, H, d_keys)

        att_score = torch.einsum("blchd,kchd->bchlk", target_embedding, source_embedding) # [B, c, H, L, k]
        att_score = self.dropout(torch.softmax(self.temperature * att_score, dim=-1))
        out_value = torch.einsum("bchlk,kchd->bclhd", att_score, value_embedding) # [B, c, L, H, d_keys]

        # out = out.reshape(B, L, -1)    # [B, L, H*d_keys]
        aligned_ts = torch.einsum("bclhd,chde->blce", out_value, self.out_projection_w) + self.out_projection_b # [B, L, n_cls, d_llm]

        return aligned_ts  # [B, L, c, d_llm]
    

class CrossAttentionLayer_old(nn.Module):
    """
    input:
        TSCluster返回的target_embedding: (B, patch_num, d_model)
        source_embedding: (S, d_llm)
        value_embedding: (S, d_llm)
    output:
        融合了文本特征且映射到d_llm的时序patch (B, patch_num, d_llm)
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(CrossAttentionLayer_old, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)

        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)  # 将融合了文本原型后的时序patch，映射到LLM的维度，d_model -> d_llm
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        B, L, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)  # [B, L, H, d_keys]
        source_embedding = self.key_projection(source_embedding).view(B, L, H, -1)       # [B, L, H, d_keys]
        value_embedding = self.value_projection(value_embedding).view(B, L, H, -1)       # [B, L, H, d_keys]

        out = self.crossattention(target_embedding, source_embedding, value_embedding) # [B, L, H, d_keys]

        out = out.reshape(B, L, -1)    # [B, L, H*d_keys]
        aligned_ts = self.out_projection(out)
        return aligned_ts  # [B, L, d_llm]
    
    def crossattention(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        # this is the attention scores, (B, L, H, S), using the Einstein summation convention
        scores = torch.einsum("blhe,bshe->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # apply the attention score to value embedding and output the reprogramming embedding
        crossattention_embedding = torch.einsum("bhls,bshe->blhe", A, value_embedding)

        return crossattention_embedding  # (B, L, H, E)
    
# 主模型类
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff  
        self.d_llm = configs.llm_dim
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.cluster_num = configs.cluster_num
        self.topk = configs.topk
        self.topkmode = configs.topkmode
        self.n_heads = configs.n_heads
        self.epoch_init_logits = None
        self.epoch_transition_logits = None
        self.epoch_emission_logits = None  # 为vali阶段保存需要的发射矩阵
        self.train_trans = configs.fully_trainable_trans
        self.topk_token_embeddings_arch = None

        self.ems_trainable_while_joint = True

        if self.train_trans or 'll' not in configs.loss_mode:
            self.ems_trainable_while_joint = False
        

        # 计算 patch_num，考虑填充
        self.patch_num = (self.seq_len + self.stride - self.patch_len) // self.stride + 1
        if self.patch_num <= 0:
            raise ValueError(f"Invalid patch configuration: seq_len={self.seq_len}, patch_len={self.patch_len}, stride={self.stride}")

        self.device = torch.device(configs.device if torch.cuda.is_available() else "cpu")

        
        self.gpt2_config = GPT2Config.from_pretrained(r'gpt2')
        self.gpt2_config.num_hidden_layers = configs.llm_layers
        self.gpt2_config.output_attentions = False
        self.gpt2_config.output_hidden_states = True
    
        self.llm_model = GPT2Model.from_pretrained(
                    r'gpt2',
                    local_files_only=True,
                    config=self.gpt2_config,
                )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
                    r'gpt2',
                    local_files_only=True
                )
        
        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,  # 指定任务类型为因果语言建模，适合序列生成任务
        #     inference_mode=False,    # 表示模型处于训练模式
        #     r=configs.r,   # LoRA的秩，控制低秩矩阵的规模  =8
        #     lora_alpha=configs.lora_alpha,   # LoRA的缩放因子，调整适配强度  =32
        #     lora_dropout=configs.lora_dropout,  # Dropout概率，防止过拟合  =0.1
        #     target_modules=["c_attn"]   # 指定LoRA适配的模块为GPT-2的注意力层（c_attn）
        # )
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        
        self.llm_model = self.llm_model.to(self.device)

        # 冻结 LLM 参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        # 获取 LLM 的全部词嵌入
        self.word_embeddings = self.llm_model.get_input_embeddings().weight  # [50256, d_llm]

        # 加载gpt2_to_local_id字典
        gpt2_to_local_id = torch.load('hmm/gpt2_to_local_id.pt', weights_only=False)
        # 创建local_to_gpt2映射tensor
        max_local_id = max(gpt2_to_local_id.values())  # 本地token_id的最大值
        self.local_to_gpt2 = torch.zeros(max_local_id + 1, dtype=torch.long, device=self.device)
        for local_id, gpt2_id in enumerate([gpt2_to_local_id[k] for k in sorted(gpt2_to_local_id.keys())]):
            self.local_to_gpt2[local_id] = gpt2_id
        
        self.local_word_embeddings = self.word_embeddings[self.local_to_gpt2]

        """实例化 WikiHMM 模块: self.wiki_hmm"""
        self.wiki_hmm = WikiHMM(self.cluster_num, configs.diag_max)

        """实例化 Normalize 模块: self.normalize_layers"""
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        if self.topkmode == 'all':
            self.topk_projection = nn.Linear(max_local_id + 1, self.topk)
            self.embedding_bias = nn.Parameter(torch.randn(self.cluster_num, self.d_llm))

        """实例化 Patching 模块: self.patching"""
        self.patching = Patching(self.patch_len, self.stride)

        """在输入TransformerEncoder前, 需要先将它投影到d_model"""
        self.patch_projection = nn.Linear(self.patch_len, self.d_model)

        """实例化 TSCluster 模块: self.ts_cluster"""
        bkb_kargs = {'encoder_layer': nn.TransformerEncoderLayer(d_model=configs.d_model, nhead=configs.n_heads),
                     'num_layers': configs.e_layers}
        self.ts_cluster = TSCluster(bkb_kargs, configs.d_model, self.cluster_num, self.patch_num, train_trans=configs.fully_trainable_trans)

        """实例化 TSAligner 模块: self.ts_aligner"""
        self.ts_aligner = TSAligner(self.cluster_num, configs.d_model, self.n_heads, self.d_llm, attention_dropout=configs.dropout, temperature=configs.temperature)

        # self.ts_llm_fusion = CrossAttentionLayer_old(configs.d_model, self.n_heads, d_llm=self.d_llm)

        """用在GPT2最后一个隐藏层输出, 先进行一步降维到d_ff=32"""
        self.projection = nn.Linear(self.d_llm, self.d_ff)
        # self.projection = nn.Linear(self.d_llm + self.d_model, self.d_ff)

        """实例化 FlattenHead 模块: self.output_projection"""
        # self.head_nf = self.d_ff * self.patch_num
        self.valid_patch = (self.pred_len + self.stride - self.patch_len) // self.stride + 1
        self.head_nf = self.d_ff * self.valid_patch
        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout, linear_layer=configs.linear_layer)
        
    def _init_trans_pi(self):
        self.ts_cluster._init_trans_pi(self.wiki_hmm.init_logits.data, self.wiki_hmm.transition_logits.data)

    def _cal_archive_topk(self):
        prob_B = torch.softmax(self.wiki_hmm.emission_logits, dim=1)
        if self.topkmode == 'select':
            _, topk_local_indices = torch.topk(prob_B, k = self.topk, dim=1) # [n_cluster, topk]
            # 直接用tensor索引映射到gpt2_id
            topk_gpt2_indices = self.local_to_gpt2[topk_local_indices] # [n_cluster, topk]
            self.topk_token_embeddings_arch = self.word_embeddings[topk_gpt2_indices] # [dictionary_size, d_llm] -> [n_cluster, topk, d_llm]
        elif self.topkmode == 'all':
            self.topk_token_embeddings_arch = self.topk_projection(self.local_word_embeddings.T).T + self.embedding_bias[:,None,:] # [n_cluster, topk, d_llm]

    def forward(self, x_enc=None, text_input=None, is_pretrain=False):   
        if is_pretrain:
            _, _, _, likelihood_loss, transition_entropy_loss, cnct_const, diag_reg, self_trans_const = self.wiki_hmm(text_input)
            return likelihood_loss, transition_entropy_loss, cnct_const, diag_reg, self_trans_const
        else:
            dec_out, likelihood_loss, transition_entropy_loss, cnct_const, entropy_loss = self.forecast(x_enc, text_input)
            return dec_out[:, -self.pred_len:, :], likelihood_loss, transition_entropy_loss, cnct_const, entropy_loss

    def forecast(self, x_enc, text_input=None):
        # 训练时
        if text_input is not None:
            """调用 wiki_hmm 模块: self.wiki_hmm"""
            if not self.train_trans:
                prob_pi, prob_A, prob_B, likelihood_loss, transition_entropy_loss, cnct_const, _, _ = self.wiki_hmm(text_input)
            else:
                prob_pi, prob_A, prob_B, likelihood_loss, transition_entropy_loss, cnct_const, _, _ = None, None, self.wiki_hmm.emission_logits, 0, 0, 0, None, None

            """原始输入 x_enc: [bs, seq_len, n_vars]"""

            """调用 Normalize 模块: self.normalize_layers"""
            x_enc = self.normalize_layers(x_enc, 'norm')  # [bs, seq_len, n_vars]

            x_enc = x_enc.permute(0, 2, 1).contiguous()  # [bs, n_vars, seq_len]

            """调用 Patching 模块: self.patching"""
            x_patched, n_vars = self.patching(x_enc)  # [bs*n_vars, patch_num, patch_len]
            
            """调用 self.patch_projection"""
            enc_out = self.patch_projection(x_patched) # [bs*n_vars, patch_num, d_model]

            """调用 TSCluster 模块: self.ts_cluster"""
            cluster_probs, ts_embedding, _, _, entropy_loss = self.ts_cluster(enc_out, prob_pi, prob_A)  # [bs*n_vars, patch_num, cluster_num]

            """获取topk token embedding for all clusters"""
            if self.ems_trainable_while_joint:
                if self.topkmode == 'select':
                    _, topk_local_indices = torch.topk(prob_B, k = self.topk, dim=1) # [n_cluster, topk]
                    # 直接用tensor索引映射到gpt2_id
                    topk_gpt2_indices = self.local_to_gpt2[topk_local_indices] # [n_cluster, topk]
                    topk_token_embeddings = self.word_embeddings[topk_gpt2_indices] # [dictionary_size, d_llm] -> [n_cluster, topk, d_llm]
                elif self.topkmode == 'all':
                    topk_token_embeddings = self.topk_projection(self.local_word_embeddings.T).T + self.embedding_bias[:,None,:] # [n_cluster, topk, d_llm]
            else:
                topk_token_embeddings = self.topk_token_embeddings_arch
            """调用 TSAligner 模块: self.ts_aligner"""
            fused_emb = self.ts_aligner(topk_token_embeddings, ts_embedding, cluster_probs)  # [bs*n_vars, patch_num, d_llm]
            
            """ 输入 LLM """
            dec_out = self.llm_model(inputs_embeds=fused_emb).last_hidden_state[:, -self.valid_patch:, :]  # [bs*n_vars, patch_num, d_llm]

            """调用 self.ts_llm_fusion"""
            # dec_fusion = self.ts_llm_fusion(ts_embedding, dec_out, dec_out)  # [bs*n_vars, patch_num, d_llm]

            """调用 self.projection, 将GPT2隐藏层输出先进行一次降维"""
            dec_out = self.projection(dec_out)  # [bs*n_vars, patch_num, d_ff]
            # dec_out = self.projection(torch.cat([dec_out, ts_embedding], dim=-1))  # [bs*n_vars, patch_num, d_ff]

            # 重塑形状
            dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
            dec_out = dec_out.permute(0, 1, 3, 2).contiguous()  # [bs, n_vars, d_ff, patch_num]

            """调用 FlattenHead 模块: self.output_projection"""
            dec_out = self.output_projection(dec_out)  # [bs, n_vars, pred_len]

            dec_out = dec_out.permute(0, 2, 1).contiguous()  # [bs, pred_len, n_vars]
            dec_out = self.normalize_layers(dec_out, 'denorm')
            return dec_out, likelihood_loss, transition_entropy_loss, cnct_const, entropy_loss
        else:
            logit_pi = self.wiki_hmm.init_logits.detach()
            logit_A = self.wiki_hmm.transition_logits.detach()
            logit_B = self.wiki_hmm.emission_logits.detach()

            # prob_pi = torch.softmax(logit_pi, dim=0)
            # prob_A = torch.softmax(logit_A, dim=1)
            # prob_B = torch.softmax(logit_B, dim=1)


            if self.ems_trainable_while_joint:
                _, topk_local_indices = torch.topk(logit_B, k = self.topk, dim=1)
                topk_gpt2_indices = self.local_to_gpt2[topk_local_indices]
                topk_token_embeddings = self.word_embeddings[topk_gpt2_indices] 
            else:
                topk_token_embeddings = self.topk_token_embeddings_arch

            x_enc = self.normalize_layers(x_enc, 'norm')  
            B, T, N = x_enc.size()
            x_enc = x_enc.permute(0, 2, 1).contiguous()  
            x_patched, n_vars = self.patching(x_enc) 
            enc_out = self.patch_projection(x_patched)
            cluster_probs, ts_embedding, _, _, entropy_loss = self.ts_cluster(enc_out, logit_pi, logit_A)
            fused_emb = self.ts_aligner(topk_token_embeddings, ts_embedding, cluster_probs) 
            dec_out = self.llm_model(inputs_embeds=fused_emb).last_hidden_state[:, -self.valid_patch:, :]  
            # dec_out = self.ts_llm_fusion(ts_embedding, dec_out, dec_out)
            dec_out = self.projection(dec_out)  
            # dec_out = self.projection(torch.cat([dec_out, ts_embedding], dim=-1))  
            dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
            dec_out = dec_out.permute(0, 1, 3, 2).contiguous() 
            dec_out = self.output_projection(dec_out)  
            dec_out = dec_out.permute(0, 2, 1).contiguous()  
            dec_out = self.normalize_layers(dec_out, 'denorm')
            return dec_out, 0, 0, 0, entropy_loss

# if __name__ == "__main__":
#     n_cls, d_model, d_keys, n_heads, d_llm = 100, 128, 64, 12, 768
#     topk = 10
#     batch_size = 32
#     patch_num = 11
#     cross_attention_layer = CrossAttentionLayer(d_model, n_heads, n_cls, d_keys, d_llm)
#     target_embedding = torch.randn(batch_size, patch_num, d_model)
#     source_embedding = torch.randn(n_cls, topk, d_llm)
#     value_embedding = torch.randn(n_cls, topk, d_llm)
#     cluster_probs = torch.randn(batch_size, patch_num, n_cls)
#     aligned_ts = cross_attention_layer(target_embedding, source_embedding, value_embedding)

