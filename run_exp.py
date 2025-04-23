import torchvision
torchvision.disable_beta_transforms_warning()

import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import random
import numpy as np
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from models import SGCMA
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali

"""环境变量"""
os.environ['CURL_CA_BUNDLE'] = ''  # 禁用 SSL 证书验证
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"  # 设置 PyTorch 内存分配策略

def makeup_logging(logging_file_nm):
    # # 配置日志，INFO输出到控制台和文件, DEBUG只输出到文件
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

def parse_args():
    parser = argparse.ArgumentParser(description='SGCMA')
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='SGCMA', help='model name')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--gpt2_path', type=str, default="gpt2", help='root path of gpt2')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--model_comment', type=str, default='1118', help='prefix when saving test results')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='datasets type')
    parser.add_argument('--root_path', type=str, default="dataset/ETT-small", help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')   # 每条样本长度是seq_len，再对样本分patch
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of TSCluster Transformer Encoder ')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads both TSCluster and TSAligner')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_interval_iters', type=int, default=400, help='max epochs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='PEMS', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)
    
    return parser.parse_args()

def main():
    # 设置随机种子
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 解析参数
    args = parse_args()

    # 确定设备（单卡）
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for ii in range(args.itr):
        # 设置实验记录
        setting = '{}_{}_ft{}_sl{}_pl{}_lr{}_lradj{}_{}_cmt{}'.format(
            args.task_name, args.data, args.features, args.seq_len, args.pred_len, args.learning_rate, args.lradj, ii, args.model_comment)
        print(setting)
        logger = makeup_logging(setting)

        # 加载数据
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        model = SGCMA.Model(args).float()
        model = model.to(device)

        # 检查点路径
        check_path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(check_path):
            os.makedirs(check_path)

        early_stopping = EarlyStopping(patience=args.patience)  

        # 优化器
        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        # 损失函数
        mse_metric = nn.MSELoss()
        mae_metric = nn.L1Loss()

        # 训练循环
        for epoch in range(args.train_epochs):
            iter_count = 0

            train_loss = 0.0
            train_mae_loss = 0.0
            train_entropy_loss = 0.0

            model.train()
            model_optim.zero_grad()
            epcoh_start_t = time.time()

            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                # 数据移动到设备
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                # 前向传播
                outputs, entropy_loss = model(batch_x)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                # 计算损失
                mae_loss = mae_metric(outputs, batch_y)  
                loss = mae_loss + entropy_loss  # 总损失



                # 反向传播和优化
                model_optim.zero_grad()
                mae_loss.backward()
                model_optim.step()

                train_loss += loss.cpu().detach().item()
                train_mae_loss += mae_loss.cpu().detach().item()
                train_entropy_loss += entropy_loss.cpu().detach().item()
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch + 1} [{batch_idx}/{len(train_loader)}, {time.time() - epcoh_start_t:.2f}s]| Train MAE Loss: {train_mae_loss / (batch_idx + 1):.7f} | "f"Train Entropy Loss: {train_entropy_loss / (batch_idx + 1):.7f}", end='\r')
                    
                    # torch.cuda.empty_cache()

            vali_mse_loss, vali_mae_loss = vali(args, model, vali_data, vali_loader, mse_metric, mae_metric, device)
            test_mse_loss, test_mae_loss = vali(args, model, test_data, test_loader, mse_metric, mae_metric, device)
            print('')
            adjust_learning_rate(model_optim, None, epoch + 1, args)

            logger.info(f"Epoch {epoch + 1} | Train MAE Loss: {train_mae_loss / len(train_loader):.7f} | "f"Vali MSE Loss: {vali_mse_loss:.7f} | Test MSE Loss:{test_mse_loss:.7f} | Test MAE Loss:{test_mae_loss:.7f}")

            early_stopping(vali_mse_loss, model, check_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

if __name__ == '__main__':
    main()