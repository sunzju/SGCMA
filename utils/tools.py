import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import matplotlib.pyplot as plt
import random
import shutil
from tqdm import tqdm
import h5py
import os

plt.switch_backend('agg')

"""文本HMM相关函数定义"""
class WikiDataset(Dataset):
    def __init__(self, sentences_path): 
        self.sentences_path = sentences_path
        self.f = h5py.File(self.sentences_path, 'r')
        self.len = len(self.f['sentences'])
    def __getitem__(self, idx):
        sentence = self.f['sentences'][idx]        # 一个样本
        return torch.tensor(sentence, dtype=torch.int32)    
    def __len__(self):
        return self.len

class EpochRandomSampler(Sampler):
    def __init__(self, total_len, max_samples):
        self.total_len = total_len
        self.max_samples = max_samples
    def __iter__(self):
        indices = torch.randperm(self.total_len)[:self.max_samples]
        return iter(indices.tolist())
    def __len__(self):
        return self.max_samples

def text_data_provider(args, ts_iter_count=None):
    max_samples = ts_iter_count * args.text_batch_size if ts_iter_count is not None else None
    text_dataset = WikiDataset(sentences_path=args.sentences_path)
    total_len = len(text_dataset)
    # text_sampler = EpochRandomSampler(total_len, max_samples)
    text_dataloader = DataLoader(
        text_dataset,
        batch_size = args.text_batch_size,
        # sampler = text_sampler,
        num_workers=4, 
        pin_memory=True
    )
    return text_dataset, text_dataloader
    

# 将非连续的token_id映射成连续的
def remap_tokens_to_local_vocab(sentences_tokens_batch, gpt2_to_local):  # gpt2_to_local_id 是一个字典，键是非连续id，值是连
    mapped_tokens = gpt2_to_local[sentences_tokens_batch]
    # 确保没有无效 token ID
    assert (mapped_tokens >= 0).all(), "发现无效的token ID映射为-1"
    return mapped_tokens




def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == 'type3':   #
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':   #
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':   # traffic \ electricity
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_mode=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def del_files(dir_path):
    shutil.rmtree(dir_path)

def vali(args, model, vali_data, vali_loader, criterion, mae_metric):
    device = next(model.parameters()).device
    total_mseloss = []
    total_mae_loss = []

    # 加载保存的 emission_logits
    init_logits_path = os.path.join('pretrain_model/vali', 'epoch_init_logits.pt')
    transition_logits_path = os.path.join('pretrain_model/vali', 'epoch_transition_logits.pt')
    emission_logits_path = os.path.join('pretrain_model/vali', 'epoch_emission_logits.pt')
    model.epoch_init_logits = torch.load(init_logits_path).to(device)
    model.epoch_transition_logits = torch.load(transition_logits_path).to(device)
    model.epoch_emission_logits = torch.load(emission_logits_path).to(device)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs, _ , _= model(batch_x)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            pred = outputs.detach()
            true = batch_y.detach()

            mseloss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            
            total_mseloss.append(mseloss.item())
            total_mae_loss.append(mae_loss.item())
    
    total_mseloss = np.average(total_mseloss)
    total_mae_loss = np.average(total_mae_loss)
    model.train()
    return total_mseloss, total_mae_loss

def test(args,  model, train_loader, vali_loader, criterion):
    device = next(model.parameters()).device
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(device)
        batch_y_mark = torch.ones(true.shape).to(device)
        loss = criterion(pred, true)

    model.train()
    return loss