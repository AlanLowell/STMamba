import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import numpy as np

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        x = self.norm(x)
        return self.fn(x, **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1): # 64 ,8
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SToken(nn.Module):  #97.4 oa:  97.802 #  --> L
    def __init__(self, dim):
        super(SToken, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.bias = nn.Parameter(torch.zeros(1,1, dim))
        nn.init.xavier_normal_(self.bias)


    def forward(self, x):
        # 在序列维度上进行SE注意力计算
        squeeze_seq =  torch.mean(x, dim=1).unsqueeze(1)  # 在band维度上进行全局平均池化  [64, 1 ,16]

        squeeze_seq = self.fc1(squeeze_seq)

        out = squeeze_seq + self.bias.expand(x.shape[0], -1, -1)
        #out = torch.cat((out, x[:,1:,:]), dim=1)
        return out * x


class S6_noC(nn.Module):
    def __init__(self,seq_len, d_model):
        super(S6_noC, self).__init__()

        self.state_dim = d_model

        self.LN_B = nn.Linear(d_model, d_model) #2*
        self.LN_C = nn.Linear(d_model, d_model)
        self.LN_delta = nn.Linear(d_model, d_model)  #

        self.delta = nn.Parameter(torch.zeros(1, seq_len, d_model)) #[L D]
        nn.init.xavier_normal_(self.delta) #yes

        self.A = nn.Parameter(torch.zeros(seq_len, d_model))  # [D,N]
        nn.init.xavier_normal_(self.A)#yes
        self.ST = SToken(d_model)

        self.Sigmoid = nn.Sigmoid()
        ''' '''

    def forward(self, x): #A,B:[B,L,N,N] C:[B,L,N] X:[B,L,D]    N是隐藏层维度N << S


        z = self.Sigmoid(x)

        B_0 = self.LN_B(x)
        C_ = self.LN_C(x)
        ''''''

        T_ = self.ST(x)  #[B L D]
        ''''''
        delta = self.LN_delta(x)  # [B L D]
        #delta = F.softplus(delta + self.delta)  # [B L D]  #
        delta = self.Sigmoid(delta + self.delta)

        A_ = torch.einsum('B L D,L D -> B L D', delta, self.A)#[B L D]
        #A_ = torch.exp(A_)
        B_ = torch.einsum('B L D,B L D->B L D ', delta, B_0)  #[B L D]

        output = []
        # 这里定义的 s 必须手动放到cuda上，因为后面会用到
        s = torch.zeros(x.shape[0], x.shape[2]).cuda()  # [b d]

        for t in range(x.shape[1]):
            s = torch.einsum('B D,B D-> B D', A_[:, t,], s) + torch.einsum('B D, B D->B D',
                                                                                      B_[:, t,],
                                                                                      x[:, t, :])  # [batch, D]

            y_pred = torch.einsum('B D,B D-> B D', C_[:, t], s) +T_[:,t,:]
            output.append(y_pred.unsqueeze(1))

        # 预测状态
        x_out = torch.cat(output, dim=1)
        x_out = x_out*z

        ''''''
        return  x_out

class Scan(nn.Module):
    def __init__(self,seq, dim): # 64, 1, 8 ,8
        super().__init__()


    def forward(self, x):# x:[64,64,11,11]


        cen = x.shape[2]//2  #5
        x = rearrange(x, 'b c h w -> b h w c')  # [批次64，空间25，光谱64]
        x_out = torch.zeros(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]).cuda()
        x_out[:, 0, :] = x[:, cen, cen, :]
        for i in range(cen): #层数 0~4

            if(i==0):
               #第1层 2*4=8个
               x_out[:,1:3,:]=x[:,cen-1,cen:cen+2,:]
               x_out[:, 3:5, :] =x[:,cen:cen+2,cen+1,:]
               x_out[:, 5:7, :] =x[:,cen+1,cen-1:cen+1,:]
               x_out[:, 7:9, :] =x[:,cen-1:cen+1,cen-1,:]

            if (i == 1):
               # 第2层 4*4=16个  4
               x_out[:, 9:13, :] = x[:, cen - 2, cen-1:cen + 3, :]
               x_out[:, 13:17, :] = x[:, cen-1:cen + 3, cen + 2, :]
               x_out[:, 17:21, :] = x[:, cen + 2, cen - 2:cen + 2, :]
               x_out[:, 21:25, :] = x[:, cen - 2:cen + 2, cen - 2, :]

            if (i == 2):
                # 第3层 6*4=24个  6
                x_out[:, 25:31, :] = x[:, cen - 3, cen - 2:cen + 4, :]
                x_out[:, 31:37, :] = x[:, cen - 2:cen + 4, cen + 3, :]
                x_out[:, 37:43, :] = x[:, cen + 3, cen - 3:cen + 3, :]
                x_out[:, 43:49, :] = x[:, cen - 3:cen + 3, cen - 3, :]
            if (i == 3):
               # 第4层 8*4=32个  8
               x_out[:, 49:57, :] = x[:, cen - 4, cen - 3:cen + 5, :]
               x_out[:, 57:65, :] = x[:, cen - 3:cen + 5, cen + 4, :]
               x_out[:, 65:73, :] = x[:, cen + 4, cen - 4:cen + 4, :]
               x_out[:, 73:81, :] = x[:, cen - 4:cen + 4, cen - 4, :]
            if (i == 4):
                # 第5层 10*4=40个  10
                x_out[:, 81:91, :] = x[:, cen - 5, cen - 4:cen + 6, :]
                x_out[:, 91:101, :] = x[:, cen - 4:cen + 6, cen + 5, :]
                x_out[:, 101:111, :] = x[:, cen + 5, cen - 5:cen + 5, :]
                x_out[:, 111:121, :] = x[:, cen - 5:cen + 5, cen - 5, :]


        return x_out

class STMambaBlock(nn.Module):
    def __init__(self,seq_len, dim, depth, mlp_dim, dropout): # 64, 1, 8 ,8
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, S6_noC(seq_len,dim))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):# x:[64,5,64]
        for S6_noC, mlp in self.layers:
            x = S6_noC(x)  # go to attention
            x = nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2), ).squeeze()  # [B D]
            x = mlp(x)  # go to MLP_Block
        return x

NUM_CLASS = 21 #9  16 21

class STMamba(nn.Module):
    def __init__(self, in_channels=1,patch = 15, dim=NUM_CLASS,   depth=1, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(STMamba, self).__init__()

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*28, out_channels=dim, kernel_size=(3, 3)),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )



        self.pos_embedding = nn.Parameter(torch.empty(1, ((patch-4 ) ** 2 + 1), dim))
        torch.nn.init.uniform_(self.pos_embedding)#, std=.02

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.scan = Scan(seq = (patch-4 ) ** 2 + 1,  dim=dim)
        self.STMambaBlock = STMambaBlock((patch-4 ) ** 2 + 1,dim, depth,  mlp_dim, dropout)



    def forward(self, x, mask=None): # x:[64, 1, 30, 9, 9]

        x = self.conv3d_features(x) #->x:[64,8,28,7,7 ]

        x = rearrange(x, 'b c h w y -> b (c h) w y') #8个通道合一，增强光谱空间特征 -> [64,8*28,7,7]
        x = self.conv2d_features(x) # ->[64,(8*28)64,5,5] #2D 卷积提取空间特征
        #x = rearrange(x,'b c h w -> b (h w) c') #[批次64，空间25，光谱64]
        x = self.scan(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        #print('cls_tokens: ', cls_tokens.shape,'x: ',x.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.STMambaBlock(x, mask)  #x_before [64, 5, 64]   x_after: [64, 5, 64]
        #print('x_after: ', x.shape)
        #x = self.to_cls_token(x[:, 0,:]) # -> x:[64,64] 没啥用 nn.identity
        if(x.shape[0]==16 or x.shape[0]==21):
            x = x.unsqueeze(0)  #最后一个批次只有一个数据
        #print('x: ', x.shape)

        return x


