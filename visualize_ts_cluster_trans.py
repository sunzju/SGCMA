import argparse
from matplotlib import pyplot as plt
import seaborn
import torch
from data_provider.data_factory import data_provider
from models.SGCMA import Model

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
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
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


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Model(args).to(device)
    model.load_state_dict(torch.load("checkpoints/long_term_forecast_ETTh1_ftM_sl96_pl96_lr0.0001_lradjPEMS_0_cmt1118/checkpoint", weights_only=True, map_location=device))
    model.eval()

    test_data, test_loader = data_provider(args, 'test')

    with torch.no_grad():

        for batch_idx, (batch_x, _, _, _) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            x_enc = model.normalize_layers(batch_x, 'norm')  # [bs, seq_len, n_vars]
            B, T, N = x_enc.size()
            x_enc = x_enc.permute(0, 2, 1).contiguous()  # [bs, n_vars, seq_len]
            x_patched, n_vars = model.patching(x_enc)  # [bs*n_vars, patch_num, patch_len]
            
            enc_out = model.patch_projection(x_patched) # [bs*n_vars, patch_num, d_model]

            cluster_probs, _, _ = model.ts_cluster(enc_out)
            #  cluster_probs: [B, patch_num, cluster_num]
            seaborn.heatmap(cluster_probs[0].cpu().detach().numpy(), cmap='YlGnBu')
            plt.show()



