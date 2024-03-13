import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, channels, embed_dim):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv1d(in_channels=channels, out_channels=embed_dim,
                              kernel_size=patch_size, stride= patch_size, padding=0,bias=False
                              )
        self.bn = nn.BatchNorm1d(embed_dim)
    def forward(self,x):
        x = self.bn(self.proj(x))
        return x
class DConv(nn.Module):
    def __init__(self, n_in, k):
        n_out = n_in
        p = 7
        super(DConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, padding=p, groups=n_in, stride=1,bias=False)
    def forward(self, x):
        y = self.depthwise(x)
        return y
class MSA(nn.Module):
    def __init__(self,in_c,k,f=4):
        super(MSA, self).__init__()
        self.dconv = DConv(in_c,k)
        self.bn = nn.BatchNorm1d(in_c)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_c,eps=1e-6),
            nn.Linear(in_c,in_c*f),
            nn.ReLU(),
            nn.Linear(in_c*f,in_c),
            nn.Dropout(0.2)
        )
        
    def forward(self,x):
        y = self.bn(x)
        y = self.dconv(y)
        x = x + y
        b, c, n = x.size()
        x = x.transpose(-2,-1)
        y = self.ffn(x)        
        x = x + y
        x = x.transpose(-2,-1)
        return x
    
class LiteFormer(nn.Module):
    def __init__(self,output_shape,depth=8):
        super(LiteFormer,self).__init__()
        self.patch = PatchEmbed(channels=1,embed_dim=64,patch_size=8)
        
        self.blocks = nn.ModuleList([
            MSA(in_c=64,k=15)
            for _ in range(depth)
        ])        
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64,output_shape)
        
    def forward(self,x):
        x = self.patch(x)
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    model = LiteFormer(3).eval()
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