import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

class signal2embedding(nn.Module):
    def __init__(self, in_channels, embed_dim,stride,dropout,d_ratio,kernel_size):
        super(signal2embedding,self).__init__()
        self.embedding_block = nn.Sequential(
                nn.Conv1d(in_channels,embed_dim,kernel_size=kernel_size,stride=stride,padding=2,bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.GELU()
                )
        self.dropout = dropout
        self.dropout_ = nn.Dropout(d_ratio)
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.embedding_block(x)
        if self.dropout:
            x = self.dropout_(x)
        return x

class DSConv_f2(nn.Module):
    def __init__(self, in_channel,out_channel,kernel_size=(3,3), strides=(2,2)):
        super(DSConv_f2, self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,strides,padding=(1,1),groups=in_channel,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.GELU()

        self.pointwise = nn.Conv2d(out_channel,out_channel,1,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.act(self.bn(self.pointwise(x)))
        return x

class DSConv(nn.Module):
    def __init__(self, in_channel,out_channel,kernel_size=(3,3), strides=(1,1)):
        super(DSConv, self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,strides,padding=(1,1),groups=in_channel,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.GELU()

        self.pointwise = nn.Conv2d(out_channel,out_channel,1,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.act(self.bn(self.pointwise(x)))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.25,r=4):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CT_block(nn.Module):
    def __init__(self, in_channel, embedding_channel, embed_dim, embed_k, embed_s,r=4,d_ratio=0.1):
        super(CT_block,self).__init__()
        self.root_n = int(np.sqrt(embed_dim))
        self.embedding = signal2embedding(in_channels=embedding_channel, embed_dim=in_channel, kernel_size=embed_k, stride=embed_s,dropout=True,d_ratio=0.25)

        self.norm_fact = (embed_dim//4) ** -0.5

        self.dsconv = DSConv(in_channel=in_channel,out_channel=in_channel)
        self.dsconv2 = DSConv_f2(in_channel=in_channel,out_channel=in_channel)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(d_ratio)
        self.bn = nn.BatchNorm1d(in_channel)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=embed_dim//4)
    def forward(self,x):
        x_ = self.embedding(x)
        b, c, n = x_.shape
        x = x_.reshape(b, c, self.root_n, self.root_n)
        b, h, w, c = x.shape
        q = self.dsconv(x)
        k = self.dsconv2(x)
        v = self.dsconv2(x)
        
        q = rearrange(q,'b c h w -> b c (h w)')
        k = rearrange(k,'b c h w -> b c (h w)')
        v = rearrange(v,'b c h w -> b c (h w)')
        
        attn = v @ k.transpose(-2,-1) * self.norm_fact
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        x = (attn @ q)
       
        x = x + x_

        x_ = self.mlp(x)
        x_ = self.dropout(x_)
        x = x + x_

        return x

class CLFormer(nn.Module):
    def __init__(self, output_shape,embed_k=[7, 8, 8],in_channels=[4, 8, 16], embed_dims=[1024, 256, 64],
                 embedding_channels=[1, 4, 8], embed_s=4, d_ratio=0,depth=3):
        super(CLFormer, self).__init__()
        self.CT_block = nn.ModuleList(
            [
                CT_block(in_channel=in_channels[i], embedding_channel=embedding_channels[i], embed_dim=embed_dims[i],embed_k=embed_k[i],embed_s=embed_s)
            for i in range(depth)
            ]
        )
        self.fc1 = nn.Linear(1024,64,bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, output_shape,bias=False)

    def forward(self, x):
        
        for block in self.CT_block:
            x = block(x)
        x = rearrange(x, 'b c n -> b (c n)')
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    model = CLFormer(3).eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 128
    timings=np.zeros((repetitions,1))


    B = 1
    C = 1
    N = 4096
    input_data = torch.randn((B, C, N)).float()

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            torch_out = model(input_data)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    print('torch 모델 평균 소요 시간 : ', np.mean(np.array(timings)))
    print('표준편차', np.std(np.array(timings)))