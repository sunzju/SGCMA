import torchvision
torchvision.disable_beta_transforms_warning()

import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import random
import numpy as np
import os


from models import SGCMA
from data_provider.data_factory import data_provider
from utils.tools import text_data_provider, remap_tokens_to_local_vocab, del_files, EarlyStopping, adjust_learning_rate, vali

# change something

"""环境变量"""
os.environ['CURL_CA_BUNDLE'] = ''  # 禁用 SSL 证书验证
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"  # 设置 PyTorch 内存分配策略


def parse_args():
    parser = argparse.ArgumentParser(description='SGCMA')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, default='Mymodel',
                        help='model name, options: [Autoformer, DLinear, Mymodel]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # wikihmm
    parser.add_argument('--sentences_path', type=str, default='wikidata/sentences_35.h5', help='wiki_sentences file')
    parser.add_argument('--unique_tokens_path', type=str, default='wikidata/tokens_35.pkl', help='unique tokens file')
    parser.add_argument('--pretrain_hmm_path', type=str, default='wikidata', help='save_pretrain_hmm_path')
    parser.add_argument('--hidden_state_num', type=int, default=100, help='hmm hidden_state_num')
    parser.add_argument('--colsum_threshold', type=float, default=0.2, help='hidden_state arrived lower bound logits')
    parser.add_argument('--text_batch_size', type=int, default=1024, help='batch size of wikipedia data')

    # ts data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='datasets type')
    parser.add_argument('--root_path', type=str, default='dataset/ETT-small', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')   # 每条样本长度是seq_len，再对样本分patch
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of TSCluster Transformer Encoder ')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=30, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')

    # optimization
    parser.add_argument('--num_workers', type=int, default=32, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=6, help='pretrain hmm epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_data, train_loader = data_provider(args, 'train')
    print(len(train_loader))
    text_dataset, text_dataloader = text_data_provider(args, ts_iter_count = len(train_loader))
    iter_train_loader = iter(train_loader)
    iter_text_dataloader = iter(text_dataloader)
    for i in range(10):
        batch_x, batch_y, _, _ = next(iter_train_loader)
        text_batch = next(iter_text_dataloader)
        print(batch_x.shape)
        print(text_batch.shape)
