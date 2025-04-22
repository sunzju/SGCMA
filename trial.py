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
import logging
from models import SGCMA
from data_provider.data_factory import data_provider
from utils.tools import text_data_provider, remap_tokens_to_local_vocab, del_files, EarlyStopping, adjust_learning_rate, vali, logging_vali_result
import seaborn as sns
import matplotlib.pyplot as plt
from models.SGCMA import WikiHMM
# change something

def makeup_logging(logging_file_nm):
    # # 配置日志，INFO输出到控制台和文件,DEBUG只输出到文件
    log_dir = "log"  
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, logging_file_nm)  
    # 自定义 Logger
    logger = logging.getLogger('SGCMA_Training')
    logger.setLevel(logging.DEBUG)  # 设置全局最低级别为 DEBUG
    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # 文件 Handler：输出 DEBUG 及以上到文件
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)  # 文件记录 DEBUG 及以上
    file_handler.setFormatter(formatter)
    # 控制台 Handler：仅输出 INFO 及以上
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

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


    # wikihmm
    parser.add_argument('--sentences_path', type=str, default='wikidata/sentences_35.h5', help='wiki_sentences file')
    parser.add_argument('--unique_tokens_path', type=str, default='wikidata/tokens_35.pkl', help='unique tokens file')
    parser.add_argument('--pretrain_hmm_path', type=str, default='wikidata', help='save_pretrain_hmm_path')
    parser.add_argument('--hidden_state_num', type=int, default=100, help='hmm hidden_state_num')
    parser.add_argument('--colsum_threshold', type=float, default=0.2, help='hidden_state arrived lower bound logits')
    parser.add_argument('--text_batch_size', type=int, default=1024, help='batch size of wikipedia data')
    parser.add_argument('--vocab_size', type=int, default=50256, help='vocab size')

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

    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', '-pl', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')

    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--d_model', '-dm', type=int, default=128, help='dimension of TSCluster Transformer Encoder ')
    parser.add_argument('--n_heads', '-nh', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', '-el', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', '-df', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--temperature', '-tp', type=float, default=0.25, help='temperature')
    parser.add_argument('--cluster_num', '-cn', type=int, default=100, help='cluster number')
    parser.add_argument('--topk', '-tk', type=int, default=1000, help='topk')
    parser.add_argument('--topkmode', '-tkm', type=str, default='select', help='select or all')
    parser.add_argument('--loss_mode', '-lm', type=str, default='mae', help='mse, mse+hmm, mse+entropy, mse+hmm+entropy')
    parser.add_argument('--hmm_reg', '-hr', type=float, default=0.1, help='hmm regularization')
    parser.add_argument('--entropy_reg', '-er', type=float, default=1, help='entropy regularization')
    parser.add_argument('--hmm_pretrained_flag', '-hpf', type=int, default=1, help='1:  hmm is pretrained, there is no need to pretrain the hmm, 0: need to pretrain the hmm')
    parser.add_argument('--hmm_pretrain_mode', '-hpm', type=str, default='ll', help='ll+diag+entropy')
    parser.add_argument('--load_hmm_flag', '-lh',  type=int, default=1, help='is load hmm')
    parser.add_argument('--linear_layer', '-ll', type=int, default=1, help='linear layer')
    parser.add_argument('--diag_max', '-dmx', type=float, default=0.7, help='diag max')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--eval_interval_iters', '-eii', type=int, default=-1, help='max epochs')
    parser.add_argument('--load_checkpoint', '-lc', type=int, default=1, help='load checkpoint')
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seq_len', '-sl', type=int, default=96, help='input sequence length') 
    parser.add_argument('--fully_trainable_trans', '-ftt', type=int, default=1, help='1: train the fully trainable transition matrix and initial prob in ts_aligner using the pre-trained hmm param as initial value')


    # optimization
    parser.add_argument('--num_workers', type=int, default=3, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--pretrain_epochs', '-pe', type=int, default=2, help='pretrain hmm epochs')
    parser.add_argument('--eval_batch_size', '-ebs', type=int, default=32, help='batch size of model evaluation')

    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)
    
    return parser.parse_args()


    # 设置随机种子
    
    

    


if __name__ == '__main__':
    
    args = parse_args()

    hmm_model = WikiHMM(100, diag_max=0.7).cuda()
    init_logits = torch.load('hmm/init_logits.pt')
    transition_logits = torch.load('hmm/transition_logits.pt')
    emission_logits = torch.load('hmm/emission_logits.pt')

    hmm_model.init_logits.data = init_logits.data
    hmm_model.transition_logits.data = transition_logits.data
    hmm_model.emission_logits.data = emission_logits.data

    torch.save(hmm_model.state_dict(), 'checkpoints/hmm_100_ll_checkpoint.pth')

    # print(hmm_model.init_logits.data)
    
    
