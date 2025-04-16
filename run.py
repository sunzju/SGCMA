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
from utils.tools import text_data_provider, remap_tokens_to_local_vocab, del_files, EarlyStopping, adjust_learning_rate, vali
import seaborn as sns
import matplotlib.pyplot as plt
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
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

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
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')   # 每条样本长度是seq_len，再对样本分patch
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
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

    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of TSCluster Transformer Encoder ')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--cluster_num', type=int, default=32, help='cluster number')
    parser.add_argument('--topk', type=int, default=512, help='topk')
    parser.add_argument('--topkmode', type=str, default='select', help='select or all')
    parser.add_argument('--loss_mode', type=str, default='mse+hmm', help='mse, mse+hmm, mse+entropy, mse+hmm+entropy')
    parser.add_argument('--hmm_reg', type=float, default=0.1, help='hmm regularization')
    parser.add_argument('--entropy_reg', type=float, default=1, help='entropy regularization')
    parser.add_argument('--hmm_pretrained_flag', type=int, default=1, help='is pretrain')
    parser.add_argument('--hmm_pretrain_mode', type=str, default='ll', help='ll+diag+entropy')
    parser.add_argument('--load_hmm_flag', type=int, default=1, help='is load hmm')
    parser.add_argument('--linear_layer', type=int, default=1, help='linear layer')
    parser.add_argument('--diag_max', type=float, default=0.7, help='diag max')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--eval_interval_iters', type=int, default=100, help='max epochs')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=2, help='pretrain hmm epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='batch size of model evaluation')

    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载映射字典
    gpt2_to_local_id = torch.load('utils/gpt2_to_local_id.pt', weights_only=False)
    gpt2_to_local = torch.full((args.vocab_size,), -1, dtype=torch.int32)
    for gpt2_id, local_id in gpt2_to_local_id.items():
        gpt2_to_local[gpt2_id] = local_id  # [50256,], 没出现的id是-1
    gpt2_to_local = gpt2_to_local.to(device)

    for ii in range(args.itr):
        # 设置实验记录
        setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_el{}_df{}_cn{}_tk{}_lm{}_tp{}_hr{}_er{}_lr{}_hpm{}_lln{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_ff, args.cluster_num, args.topk, args.loss_mode, args.temperature, args.hmm_reg, args.entropy_reg, args.learning_rate, args.hmm_pretrain_mode, args.linear_layer)
        print(setting)
        logger = makeup_logging(setting)

        # 加载 ts dataset 和 ts dataloader
        train_data, train_loader = data_provider(args, 'train') 
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        """ 初始化模型, 创建类的实例时, 会自动调用其__init__方法, __init__方法中的代码会按顺序执行。"""
        model = SGCMA.Model(args).float().to(device)

        # 检查点路径
        check_pth = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        if not os.path.exists(check_pth):
            os.makedirs(check_pth)

        early_stopping = EarlyStopping(patience=args.patience)

        # 优化器
        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        # 损失函数
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        checkpoint_path = os.path.join(check_pth, 'checkpoint')
        if os.path.exists(checkpoint_path):
            try:
                print(f"Found checkpoint at {checkpoint_path}. Resuming Phase 1 training.")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint, strict=False)  # 加载模型状态
                print(f"Loaded from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("No checkpoint found. Starting Phase 1 from scratch.")    
        else:
            print("No checkpoint found. Starting from scratch.")        

        _, text_dataloader = text_data_provider(args)

        if args.load_hmm_flag:
            pretrain_hmm_path = os.path.join(args.checkpoints, f'hmm_{args.cluster_num}_{args.hmm_pretrain_mode}_checkpoint.pth')
            if os.path.exists(pretrain_hmm_path):
                print(f"Pretrained HMM found at {pretrain_hmm_path}.")
                full_model_state = torch.load(pretrain_hmm_path, map_location=device)
                model.wiki_hmm.load_state_dict(full_model_state)

        if not args.hmm_pretrained_flag:
            # === Phase 1: Train text HMM for 5 epochs ===
            print(f"Starting Phase 1: Training text HMM for {args.pretrain_epochs} epochs")
            model.train()
            for epoch in range(args.pretrain_epochs):  
                epoch_time = time.time()
                epoch_loss = 0.0
                iter_text_dataloader = iter(text_dataloader)
                with tqdm(range(len(text_dataloader)), desc=f"Text Epoch {epoch + 1}") as pbar:   # pbar 是 tqdm 包装后的 dataloader，仍然是一个可迭代对象
                    for batch_idx in pbar:
                        sentences = next(iter_text_dataloader)
                        sentences = sentences.to(device)
                        sentences_mapped = remap_tokens_to_local_vocab(sentences, gpt2_to_local) 
                        model_optim.zero_grad()
                        likelihood_loss, transition_entropy_loss, cnct_const, diag_reg, self_trans_const = model(text_input=sentences_mapped, is_pretrain=True)  # is_pretrain=True，则hmm_loss= self.wiki_hmm(text_input)                    
                        hmm_loss = likelihood_loss
                        hmm_loss.backward()
                        model_optim.step()
                        epoch_loss += hmm_loss.item()
                        pbar.set_postfix({'loss': hmm_loss.item()})
                        if batch_idx % 1000 == 999:
                            with torch.no_grad():
                                transition_logits = model.wiki_hmm.transition_logits.detach().cpu()
                                transition_matrix = torch.softmax(transition_logits, dim=1).numpy()  # 转换为概率矩阵

                                # 绘制状态转移矩阵热力图
                                plt.figure(figsize=(12, 10))
                                sns.heatmap(transition_matrix, cmap='Reds', linewidths=0.1, linecolor='white',annot=False, fmt=".2f", 
                                            xticklabels=False, yticklabels=False, cbar=True)
                                plt.title("State Transition Matrix")
                                plt.xlabel("Next State")
                                plt.ylabel("Current State")
                                plt.savefig(os.path.join(check_pth, f'transition_matrix_heatmap_{epoch + 1}-{batch_idx}.png'), bbox_inches="tight")
                                plt.close()
                                hmm_checkpoint = model.wiki_hmm.state_dict()
                                torch.save(hmm_checkpoint, os.path.join(args.checkpoints, f'hmm_{args.cluster_num}_{args.hmm_pretrain_mode}_checkpoint.pth'))
                                print(f"Saved checkpoint for hmm pretraining")

                avg_train_loss = epoch_loss / len(text_dataloader)
                logger.info(f"Text Epoch {epoch + 1} cost time: {time.time() - epoch_time}")
                logger.info(f"Text Epoch {epoch + 1} Average Loss: {avg_train_loss:.7f}")

                # 保存当前epoch的模型和状态


        # === Phase 2: Joint training ===
        print("Starting Phase 2: Joint training")
        best_test_mse = float('inf')
        best_vali_mse = float('inf')
        for epoch in range(args.train_epochs):  
            if early_stopping.early_stop:
                break
            model.train()
            train_loss = 0.0         # 总损失
            train_hmm_loss = 0.0     # text的极大似然损失
            train_entropy_loss = 0.0 # ts聚类的熵损失            
            train_mseloss = 0.0      # ts的mse损失
            epoch_time = time.time()
            model_optim.zero_grad()
            # zip 将两个 DataLoader 的迭代器配对，同步迭代.  对于zip两个Dataloader同步迭代，使用tqdm包装，total总长度需要显式地写出来。单个DataLoader时不需要
            pbar = tqdm(range(len(train_loader)), desc=f"Train Epoch {epoch + 1}", total=len(train_loader))
            iter_train_loader = iter(train_loader)
            iter_text_dataloader = iter(text_dataloader)

            for batch_idx in pbar:

                ts_batch = next(iter_train_loader)
                text_batch = next(iter_text_dataloader)
                batch_x, batch_y, _, _ = ts_batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                text_batch = text_batch.to(device)
                
                text_mapped = remap_tokens_to_local_vocab(text_batch, gpt2_to_local)

                outputs, likelihood_loss, transition_entropy_loss, cnct_const, entropy_loss = model(
                    batch_x, text_input=text_mapped, is_pretrain=False
                )  # 但其实只用到了batch_x，其他都没用

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                mseloss = criterion(outputs, batch_y)

                # total_loss = mseloss + hmm_loss + entropy_loss
                loss = 0
                if 'mse' in args.loss_mode:
                    loss += mseloss
                if 'hmm' in args.loss_mode:
                    loss += args.hmm_reg * likelihood_loss
                if 'entropy' in args.loss_mode:
                    loss += args.entropy_reg * entropy_loss

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                train_loss += loss.cpu().detach().item()
                train_hmm_loss += likelihood_loss.cpu().detach().item()
                train_mseloss += mseloss.cpu().detach().item()
                train_entropy_loss += entropy_loss.cpu().detach().item()
                
                torch.cuda.empty_cache()

                pbar.set_postfix({
                    'mse': mseloss.cpu().detach().item(),  
                    'entropy_loss': entropy_loss.cpu().detach().item(),                     
                    'hmm_loss': likelihood_loss.cpu().detach().item()
                })

                if batch_idx % args.eval_interval_iters == args.eval_interval_iters - 1:
                    # torch.save(model.wiki_hmm.init_logits, os.path.join('pretrain_model/vali', 'epoch_init_logits.pt'))
                    # torch.save(model.wiki_hmm.transition_logits, os.path.join('pretrain_model/vali', 'epoch_transition_logits.pt'))
                    # torch.save(model.wiki_hmm.emission_logits, os.path.join('pretrain_model/vali', 'epoch_emission_logits.pt'))
                    vali_mseloss, vali_mae_loss = vali(args, model, vali_data, vali_loader, criterion, mae_metric)
                    test_mseloss, test_mae_loss = vali(args, model, test_data, test_loader, criterion, mae_metric)
                    print('')
                    if vali_mseloss < best_vali_mse:
                        best_vali_mse = vali_mseloss
                    if test_mseloss < best_test_mse:
                        best_test_mse = test_mseloss
                    logger.info(f"Epoch {epoch + 1}/{batch_idx + 1} | V MSE Loss: {vali_mseloss:.7f} | T MSE Loss: {test_mseloss:.7f} | "
                  f"T MAE Loss: {test_mae_loss:.7f} | Best V MSE: {best_vali_mse:.7f} | Best T MSE: {best_test_mse:.7f}")
                    early_stopping(vali_mseloss, model, check_pth)
                    if early_stopping.early_stop:
                        logger.info("Early stopping")
                        break

                # print(f"mse: {train_mseloss / (batch_idx + 1)}, entropy_loss: {train_entropy_loss / (batch_idx + 1)}, hmm_loss: {train_hmm_loss / (batch_idx + 1)}", end='\r')
           

            logger.info(f"Epoch {epoch + 1} | Train Total Loss: {train_loss / len(train_loader):.7f} | "
                  f"HMM Loss: {train_hmm_loss / len(train_loader):.7f} | "
                  f"MSE Loss: {train_mseloss / len(train_loader):.7f} | "
                  f"Entropy Loss: {train_entropy_loss / len(train_loader):.7f} ")



if __name__ == '__main__':
    
    main()