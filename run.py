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

# hjhjhjjh
from models import SGCMA
from data_provider.data_factory import data_provider
from utils.tools import text_data_provider, remap_tokens_to_local_vocab, del_files, EarlyStopping, adjust_learning_rate, vali

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
    parser.add_argument('--sentences_path', type=str, default='exp2\wikidata\sentences_35.h5', help='wiki_sentences file')
    parser.add_argument('--unique_tokens_path', type=str, default='exp2\wikidata\tokens_35.pkl', help='unique tokens file')
    parser.add_argument('--pretrain_hmm_path', type=str, default='exp2\pretrain_model', help='save_pretrain_hmm_path')
    parser.add_argument('--hidden_state_num', type=int, default=100, help='hmm hidden_state_num')
    parser.add_argument('--colsum_threshold', type=float, default=0.2, help='hidden_state arrived lower bound logits')
    parser.add_argument('--text_batch_size', type=int, default=1024, help='batch size of wikipedia data')

    # ts data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='datasets type')
    parser.add_argument('--root_path', type=str, default='exp2\datasets\ETT-small', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
    parser.add_argument('--checkpoints', type=str, default='exp2\checkpoints', help='location of model checkpoints')

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
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
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

def main():
    # 设置随机种子
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 解析参数
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载映射字典
    gpt2_to_local_id = torch.load(r'exp2\utils\gpt2_to_local_id.pt')
    vocab_size = 50256
    gpt2_to_local = torch.full((vocab_size,), -1, dtype=torch.int32)
    for gpt2_id, local_id in gpt2_to_local_id.items():
        gpt2_to_local[gpt2_id] = local_id  # [50256,], 没出现的id是-1
    gpt2_to_local = gpt2_to_local.to(device)

    for ii in range(args.itr):
        # 设置实验记录
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_ff, args.factor, args.embed, args.des, ii)

        # 加载 ts dataset 和 ts dataloader
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        """ 初始化模型, 创建类的实例时, 会自动调用其__init__方法, __init__方法中的代码会按顺序执行。"""
        model = SGCMA.Model(args).float().to(device)

        # 检查点路径
        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=args.patience)

        # 优化器
        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        # 损失函数
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        
        pretrain_hmm_path = os.path.join(args.pretrain_hmm_path, 'text_hmm.pth')
        if os.path.exists(pretrain_hmm_path):
            print(f"Pretrained HMM found at {pretrain_hmm_path}. Skipping Phase 1.")
            model.load_state_dict(torch.load(pretrain_hmm_path))

        else:
            checkpoint_path = os.path.join(args.pretrain_hmm_path, 'checkpoint.pth')
            start_epoch = 0
            if os.path.exists(checkpoint_path):
                print(f"Found checkpoint at {checkpoint_path}. Resuming Phase 1 training.")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型状态
                model_optim.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
                start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
                print(f"Resuming from epoch {start_epoch}")
            else:
                print("No checkpoint found. Starting Phase 1 from scratch.")        
            # === Phase 1: Train text HMM for 5 epochs ===
            print("Starting Phase 1: Training text HMM for 5 epochs")
            text_dataset, text_dataloader = text_data_provider(args)
            model.train()
            for epoch in range(start_epoch, args.pretrain_epochs):  
                epoch_time = time.time()
                epoch_loss = 0.0
                with tqdm(text_dataloader, desc=f"Text Epoch {epoch + 1}") as pbar:   # pbar 是 tqdm 包装后的 dataloader，仍然是一个可迭代对象
                    for batch_idx, sentences in enumerate(pbar):
                        sentences = sentences.to(device)
                        sentences_mapped = remap_tokens_to_local_vocab(sentences, gpt2_to_local) 
                        model_optim.zero_grad()
                        hmm_loss = model(text_input = sentences_mapped, is_pretrain=True)  # is_pretrain=True，则hmm_loss= self.wiki_hmm(text_input)                    
                        hmm_loss.backward()
                        model_optim.step()
                        epoch_loss += hmm_loss.item()
                        pbar.set_postfix({'loss': hmm_loss.item()})
                avg_train_loss = epoch_loss/len(text_dataloader)
                print(f"Text Epoch {epoch + 1} cost time: {time.time() - epoch_time}")
                print(f"Text Epoch {epoch + 1} Average Loss: {avg_train_loss:.7f}")

                # 保存当前epoch的模型和状态
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_optim.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint for epoch {epoch + 1} at {checkpoint_path}")
            # 保存预训练模型
            torch.save(model.state_dict(), os.path.join(args.pretrain_hmm_path, 'text_hmm.pth'))
            torch.save(model.wiki_hmm.init_logits, os.path.join(args.pretrain_hmm_path, 'init_logits.pt'))
            torch.save(model.wiki_hmm.transition_logits, os.path.join(args.pretrain_hmm_path, 'transition_logits.pt'))
            torch.save(model.wiki_hmm.emission_logits, os.path.join(args.pretrain_hmm_path, 'emission_logits.pt'))
            print("Saved text HMM model")

        # === Phase 2: Joint training ===
        print("Starting Phase 2: Joint training")
        for epoch in range(args.train_epochs):  
            model.train()
            train_loss = 0.0         # 总损失
            train_hmm_loss = 0.0     # text的极大似然损失
            train_entropy_loss = 0.0 # ts聚类的熵损失            
            train_mseloss = 0.0      # ts的mse损失

            epoch_time = time.time()
            model_optim.zero_grad()

            # 动态下采样 text dataloader，以保持和ts相同的迭代次数
            text_dataset, text_dataloader = text_data_provider(args, ts_iter_count = len(train_loader))
            # zip 将两个 DataLoader 的迭代器配对，同步迭代.  对于zip两个Dataloader同步迭代，使用tqdm包装，total总长度需要显式地写出来。单个DataLoader时不需要
            with tqdm(zip(train_loader, text_dataloader), desc=f"Train Epoch {epoch + 1}", total=len(train_loader)) as pbar:
                for batch_idx, (ts_batch, text_batch) in enumerate(pbar):
                    batch_x, batch_y, batch_x_mark, batch_y_mark = ts_batch
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                    text_batch = text_batch.to(device)
                    
                    text_mapped = remap_tokens_to_local_vocab(text_batch, gpt2_to_local)

                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                    outputs, hmm_loss, entropy_loss = model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, text_input=text_mapped, is_pretrain=False
                    )  # 但其实只用到了batch_x，其他都没用

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    mseloss = criterion(outputs, batch_y)

                    total_loss = mseloss + hmm_loss + entropy_loss

                    model_optim.zero_grad()
                    total_loss.backward()
                    model_optim.step()

                    train_loss += total_loss.item()
                    train_hmm_loss += hmm_loss.item()
                    train_mseloss += mseloss.item()
                    train_entropy_loss += entropy_loss.item()

                    pbar.set_postfix({
                        'mse': train_mseloss / (batch_idx + 1),  
                        'entropy_loss': train_entropy_loss / (batch_idx + 1),                     
                        'hmm_loss': train_hmm_loss / (batch_idx + 1)
                    })
            torch.save(model.wiki_hmm.init_logits, os.path.join(r'exp2\pretrain_model\vali', 'epoch_init_logits.pt'))
            torch.save(model.wiki_hmm.transition_logits, os.path.join(r'exp2\pretrain_model\vali', 'epoch_transition_logits.pt'))
            torch.save(model.wiki_hmm.emission_logits, os.path.join(r'exp2\pretrain_model\vali', 'epoch_emission_logits.pt'))
            print("Save epoch_logits")

            print(f"Epoch {epoch + 1} | Train Total Loss: {train_loss / len(train_loader):.7f} | "
                  f"HMM Loss: {train_hmm_loss / len(train_loader):.7f} | "
                  f"MSE Loss: {train_mseloss / len(train_loader):.7f} | "
                  f"Entropy Loss: {train_entropy_loss / len(train_loader):.7f} ")

            # 验证和早停
            vali_mseloss, vali_mae_loss = vali(args, model, vali_data, vali_loader, criterion, mae_metric)
            test_mseloss, test_mae_loss = vali(args, model, test_data, test_loader, criterion, mae_metric)
            print(f"Epoch {epoch + 1} | Vali MSE Loss: {vali_mseloss:.7f} | Test MSE Loss: {test_mseloss:.7f} | "
                  f"MAE Loss: {test_mae_loss:.7f}")

            early_stopping(vali_mseloss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        del_files(path)
        print('Deleted checkpoints successfully')

if __name__ == '__main__':
    main()