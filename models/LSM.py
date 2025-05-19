import torch
from mamba_ssm import Mamba
from layers.RevIN import RevIN
# from RevIN import RevIN
import torch.nn as nn
from layers.Embed import PatchEmbedding


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(1728, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """

    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf * mask, dim=1)
        x_inv = x - x_var

        return x_var, x_inv

class Model(torch.nn.Module):
    def __init__(self,configs,patch_len=16,stride=8):
        super(Model, self).__init__()
        self.configs=configs
        if self.configs.revin==1:
            self.revin_layer = RevIN(self.configs.enc_in)
        self.lin0=torch.nn.Linear(self.configs.seq_len,self.configs.seq_len+self.configs.pred_len)
        self.lin1=torch.nn.Linear(self.configs.d_model,self.configs.n1)
        # self.lin1=torch.nn.Linear(self.configs.seq_len,self.configs.n1)
        self.dropout1=torch.nn.Dropout(self.configs.dropout)

        self.lin2=torch.nn.Linear(self.configs.n1,self.configs.n2)
        self.dropout2=torch.nn.Dropout(self.configs.dropout)

        self.lin3 = torch.nn.Linear(self.configs.n2, self.configs.n3)
        self.dropout2 = torch.nn.Dropout(self.configs.dropout)
        self.dim = configs.d_model

        # self.task_name = configs.task_name

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(192, 16)
        self.dropout = nn.Dropout(configs.head_dropout)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, configs.dropout)

        if self.configs.ch_ind==1:
            self.d_model_param1=1
            self.d_model_param2=1

        else:
            self.d_model_param1=self.configs.n2
            self.d_model_param2=self.configs.n1

        self.mamba1=Mamba(d_model=self.configs.n3,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)
        self.mamba2=Mamba(d_model=self.configs.n2,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)
        self.mamba3=Mamba(d_model=self.configs.n1,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)
        self.mamba4=Mamba(d_model=self.d_model_param2,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)

        self.lin4=torch.nn.Linear(self.configs.n3,self.configs.n2)
        self.lin5=torch.nn.Linear(2*self.configs.n2, self.configs.n1)
        # self.lin6=torch.nn.Linear(2*self.configs.n1,self.configs.pred_len)
        self.lin6=torch.nn.Linear(2*self.configs.n1,32)

        self.linear_de = torch.nn.Linear(384,32)

        self.Pnum = int((configs.pred_len + configs.seq_len - configs.patch_len) / configs.stride + 2)
        self.head = Flatten_Head(configs.enc_in, configs.d_model * self.Pnum, configs.pred_len,
                                 head_dropout=configs.dropout)
        # self.mask_spectrum = configs.mask_spectrum
        # self.disentanglement = FourierFilter(self.mask_spectrum)

    def forward(self, x):
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'norm')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # seasonal_init, trend_init = self.disentanglement(x)









        x = torch.permute(x, (0, 2, 1))
        x = self.lin0(x)
        # (16,7,192)
        x,n_vars = self.patch_embedding(x)
        # (112,24,32)
        # if self.configs.ch_ind == 1:
        #     x = torch.reshape(x, (x.shape[0] * x.shape[1], 1, x.shape[2]))
        # (1344,1,96)
        x = self.lin1(x)
        x_res1 = x
        x = self.dropout1(x)
        # x3 = self.attn(x)
        x3 = self.mamba3(x)
        # x3 = torch.permute(x3,(0,2,1))
        # # x3 =
        # x3 = self.mamba4(x3)
        # x3 = torch.permute(x3,(0,2,1))
        # x3 = x3_1+x3 # 512
        x3 = x_res1 + x3


        x = self.lin2(x)
        x_res2 = x
        x = self.dropout2(x)
        x2 = self.mamba2(x) # 128
        # x2 = self.attn(x)
        # x2 = torch.permute(x2,(0,2,1))
        # x2 = self.mamba4(x2)
        # x2 = torch.permute(x2,(0,2,1))
        # x2 = x2_1+x2
        x2 = x_res2+x2

        x = self.lin3(x)
        x_res3 = x
        x = self.dropout2(x)
        x1 = self.mamba1(x) # 32
        # x1 = self.attn(x)
        # x1 = torch.permute(x1,(0,2,1))
        # x1_ = self.mamba4(x1)
        # x1 = torch.permute(x1,(0,2,1))
        # x=x1_1+x1+x_res3
        x = x_res3+x1


        x = self.lin4(x) # x=tensor:(112,1,256)
        x_d1 = x
        # x = x+x_res2
        x = torch.cat([x, x2], dim=2)

        x = self.lin5(x)
        x_d2 = x
        # x = x+x_res1
        x = torch.cat([x, x3], dim=2)

        x = self.lin6(x)
        x_d3 = x
        # x = torch.cat([x_d1, x_d2], dim=2)
        # x = self.linear_de(x)

        x_input = torch.reshape(
            x, (-1, n_vars, x.shape[-2], x.shape[-1])) # (16,7,24,32)
        x_input = x_input.permute(0, 1, 3, 2) # (16,7,32,24)

        x_input = self.head(x_input)
        x = x_input.permute(0, 2, 1)

        #
        # if self.configs.ch_ind == 1:
        #     x = torch.reshape(x, (-1, self.configs.enc_in, self.configs.pred_len))



        # x = self.head(x)
        # x = self.flatten(x)
        # x = self.linear(x)
        # x = self.dropout(x)

        # x = torch.permute(x, (0, 2, 1))
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'denorm')
        else:
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))

        return x

# if __name__== '__main__':
#     from torchstat import stat
#
#     parser = argparse.ArgumentParser(description='Time Series Forecasting')
#
#     # RANDOM SEED
#     parser.add_argument('--random_seed', type=int, default=2024, help='random seed')
#
#     # BASIC CONFIG
#     parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
#     parser.add_argument('--model_id', type=str, required=False, default='ETTh1', help='model id')
#     parser.add_argument('--model', type=str, required=False, default='TimeMachine',
#                         help='model name, options: [TimeMachine]')
#     parser.add_argument('--model_id_name', type=str, required=False, default='ETTh1', help='model id name')
#
#     # DATALOADER
#     parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
#     parser.add_argument('--root_path', type=str, default='data/ETT-small/', help='root path of the data file')
#     parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
#     parser.add_argument('--features', type=str, default='M',
#                         help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#     parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
#     parser.add_argument('--freq', type=str, default='h',
#                         help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
#
#     # FORECASTING TASK
#     parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=48, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
#     parser.add_argument('--n1', type=int, default=512, help='First Embedded representation')
#     parser.add_argument('--n2', type=int, default=128, help='Second Embedded representation')
#     parser.add_argument('--n3', type=int, default=32, help='Third Embedded representation')
#
#     # METHOD
#     parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
#     parser.add_argument('--ch_ind', type=int, default=1, help='Channel Independence; True 1 False 0')
#     parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')
#     parser.add_argument('--d_state', type=int, default=256, help='d_state parameter of Mamba')
#     parser.add_argument('--dconv', type=int, default=2, help='d_conv parameter of Mamba')
#     parser.add_argument('--e_fact', type=int, default=1, help='expand factor parameter of Mamba')
#     parser.add_argument('--enc_in', type=int, default=7,
#                         help='encoder input size')  # Use this hyperparameter as the number of channels
#     parser.add_argument('--dropout', type=float, default=0.7, help='dropout')
#     parser.add_argument('--embed', type=str, default='timeF',
#                         help='time features encoding, options:[timeF, fixed, learned]')
#     parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
#
#     # OPTIMIZATION
#     parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=1, help='experiments times')
#     parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='test', help='exp description')
#     parser.add_argument('--loss', type=str, default='mse', help='loss function')
#     parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
#     parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
#     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
#
#     # GPU
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
#     parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
#
#     args = parser.parse_args()
#
#     model = Model(args).to(device='cuda:0')
#     batch_x = torch.randn(16,96,7, device='cuda:0')
#     flops, params = profile(model, inputs=(batch_x,))
#
#     # model = Model(args)
#     # stat(model, (batch_x))
#
#     print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
#     print("params=", str(params / 1e6) + '{}'.format("M"))