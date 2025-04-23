import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {1: 1e-4, 2: 5e-5, 3: 5e-5, 4: 1e-5, 5: 5e-6, 6: 1e-6, 7: 5e-7}
    elif args.lradj == 'type3':   #
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** (epoch // 1))}
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

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def vali(args, model, vali_data, vali_loader, mse_metric, mae_metric, device):
    
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    total_entropy_loss = 0.0

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs, entropy_loss= model(batch_x)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            pred = outputs.detach()
            true = batch_y.detach()

            mse_loss = mse_metric(pred, true)
            mae_loss = mae_metric(pred, true)
            
            total_entropy_loss += entropy_loss.cpu().detach().item()
            total_mse_loss += mse_loss.cpu().detach().item()
            total_mae_loss += mae_loss.cpu().detach().item()
            print(f'[{i}/{len(vali_loader)}]', end='\r')
    
    total_mse_loss = total_mse_loss / len(vali_loader)
    total_mae_loss = total_mae_loss / len(vali_loader)
    total_entropy_loss = total_entropy_loss / len(vali_loader)

    model.train()
    return total_mse_loss, total_mae_loss

