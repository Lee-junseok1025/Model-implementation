import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SMC(nn.Module):
    def __init__(self,in_c, out_c, s=2, k=[3,5,7,9],p=[1,2,3,4]):
        super(SMC, self).__init__()
        self.split_c = out_c // 4
        self.dconv = nn.Conv1d(in_c,out_c,1,groups=in_c)
        self.conv_block = nn.ModuleList([
            nn.Conv1d(self.split_c,self.split_c,k[i],2,p[i],groups=self.split_c)
            for i in range(4)
        ])
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(out_c)
    def forward(self,x):
        split_x = self.dconv(x).split(self.split_c,1)
        for i, (s_x, conv) in enumerate(zip(split_x, self.conv_block)):
            if i == 0:
                z = conv(s_x)
            else:
                z = torch.cat([z,conv(s_x)],1)
        z = self.gelu(self.bn(z))
        return z   
    
class BSA(nn.Module):
    def __init__(self,in_c):
        super(BSA, self).__init__()
        self.q = nn.Conv1d(in_c,in_c,1)
        self.k = nn.Conv1d(in_c,in_c,1)
        self.v = nn.Conv1d(in_c,in_c,1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(in_c,in_c,1)
    def forward(self,x):
        t, k, u = self.q(x), self.k(x), self.v(x)
        t = F.softmax(torch.mean(t,1,keepdim=True),dim=-1)
        u = F.relu(u)
        
        tk = torch.mean(((u * k) + u),-1,keepdim=True)
        attn = tk * u
        x = self.conv(attn)
        return x
        
class LFE_block(nn.Module):
    def __init__(self,in_c, out_c,embed_dim,r=4):
        super(LFE_block, self).__init__()
        self.smc = SMC(in_c=in_c,out_c=out_c)
        
        self.bsa = BSA(out_c)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//r),
            nn.ReLU(),
            nn.Linear(embed_dim//r,embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim,eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim,eps=1e-6)
    def forward(self,x):
        smc = self.smc(x)
        
        bsa = self.norm1(self.bsa(smc)) + smc
                
        y = self.norm2(self.ffn(bsa)) + bsa
        return y

class LiveFormer(nn.Module):
    def __init__(self,output_shape:int,in_cs:list = [1,32,64,128], out_cs:list=[32,64,128,256], embed_dims:list=[512,256,128]):
        super(LiveFormer, self).__init__()
        self.embedding = nn.Conv1d(in_cs[0],out_cs[0],15,2,7)
        self.pooling = nn.AvgPool1d(2, 2)
        
        self.fe_blocks = nn.ModuleList([
            LFE_block(in_cs[i+1],out_cs[i+1],embed_dims[i])
            for i in range(3)
        ])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, output_shape)
    def forward(self,x):
        x = self.pooling(x)
        x = self.embedding(x)
        
        for block in self.fe_blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    model = LiveFormer(3).eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 128
    timings=np.zeros((repetitions,1))
    
    B = 1
    C = 1
    N = 4096
    input_data1 = torch.randn((B, C, N)).float()

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            torch_out = model(input_data1)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    print('torch 모델 평균 소요 시간 : ', np.mean(np.array(timings[1:])))
    print('표준편차', np.std(np.array(timings[1:])))
    print('Max: ',np.max(np.array(timings[1:])))
    print('Min: ',np.min(np.array(timings[1:])))