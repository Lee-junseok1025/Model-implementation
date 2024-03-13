import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VFE(nn.Module):
    def __init__(self, n=8):
        super(VFE, self).__init__()
        self.conv_list = nn.ModuleList()
        for i in range(1,n+1):
            self.conv_list.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(2 * i) - 1, padding=i-1))
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(n)

    def forward(self, x):
        for i, block in enumerate(self.conv_list):
            if i == 0:
                y = block(x)
            else:
                y = torch.cat([y,block(x)],1)    
        y = self.gelu(y)
        y = self.bn(y)
        b, c, s = y.shape
        y = y.reshape(b, c, 64, 64)
        return y # (bs, C0, 32, 32)
class DSC(nn.Module):
    def __init__(self, n_in, n_out):
        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=3, padding=1, groups=n_in, stride=2)
        self.pconv = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1)
        self.act_fn = nn.Mish()
    def forward(self, x):
        y = self.depthwise(x)
        y = self.act_fn(self.pconv(x))
        return y

class DC(nn.Module):
    def __init__(self, n_in):
        super(DC, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=n_in, out_channels=n_in, kernel_size=3, padding=1, groups=n_in, stride=1)
        self.pconv = nn.Conv2d(in_channels=n_in, out_channels=n_in, kernel_size=1)
        self.act_fn = nn.Mish()
    def forward(self, x):
        y = self.depthwise(x)
        y = self.act_fn(self.pconv(x))
        return y
class MSA(nn.Module):
    def __init__(self,out_c):
        super(MSA, self).__init__()
        self.dsc_conv = DC(out_c)
        self.dsc_bn = nn.BatchNorm2d(out_c)
        
        self.pconv = nn.Conv2d(out_c,out_c,1)
        self.pconv_bn = nn.BatchNorm2d(out_c)
        
    def forward(self,x):
        x = self.dsc_bn(self.dsc_conv(x))
        _x = x
        a, b, v = F.softmax(self.pconv_bn(self.pconv(x))), F.softmax(self.pconv_bn(self.pconv(x)),dim=-1), self.pconv_bn(self.pconv(x))
        attn = (a @ v) * F.sigmoid(b @ v)
        x = _x + attn
        return x
    
class stage(nn.Module):
    def __init__(self,in_c,out_c,head,depth):
        super(stage,self).__init__()
        self.head = head
        split_head = out_c // self.head
        self.dsc_conv = DSC(n_in=in_c,n_out=out_c)
        self.dsc_bn = nn.BatchNorm2d(out_c)
        self.act_fn = nn.Mish()
        self.msa_block = nn.ModuleList([
            MSA(out_c//split_head)
            for i in range(depth)
        ])
    def forward(self,x):
        x = self.act_fn(self.dsc_bn(self.dsc_conv(x)))
        split_x = x.split(self.head,1)
        for i, s_x in enumerate(split_x):
            if i == 0:
                for msa in self.msa_block:
                    z = msa(s_x)
            else:
                for msa in self.msa_block:
                    y = msa(s_x)
                z = torch.cat([z,y],1)
        z = x + z
        return z                    
class SANet(nn.Module):
    def __init__(self,output_shape):
        super(SANet,self).__init__()
        self.vfe = VFE(8)
        
        self.stage1 = stage(8,8,8,1)
        self.stage2 = stage(8,16,8,2)
        self.stage3 = stage(16,32,8,1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            #nn.Linear(64,64),
            #nn.GELU(),
            nn.Linear(32,output_shape)
        )
    def forward(self,x):
        x = self.vfe(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    model = SANet(3).eval().cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 128
    timings=np.zeros((repetitions,1))
    
    B = 1
    C = 1
    N = 4096
    input_data1 = torch.randn((B, C, N)).float().cuda()

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