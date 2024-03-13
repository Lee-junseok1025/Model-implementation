import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AGM_1D(nn.Module):
    def __init__(self,in_c,out_c,k,dk,s,p,d=2):
        super(AGM_1D,self).__init__()
        self.aconv = nn.Conv1d(in_c,out_c,kernel_size=k,stride=s,padding=p,dilation=d)
        self.abn = nn.BatchNorm1d(out_c)
        self.act_fn = nn.ReLU()
        
        if dk > 3:
            dp = 2
        else:
            dp = 1
        self.dconv = nn.Conv1d(out_c,out_c,kernel_size=dk,stride=s,padding=dp,groups=out_c)
        self.dbn = nn.BatchNorm1d(out_c)
    def forward(self,x):
        x = self.act_fn(self.abn(self.aconv(x)))
        
        x_ = self.act_fn(self.dbn(self.dconv(x)))
        
        x = torch.concat([x, x_],1)
        return x
class AGM_2D(nn.Module):
    def __init__(self,in_c,out_c,k,dk,s,p,d=2):
        super(AGM_2D,self).__init__()
        self.aconv = nn.Conv2d(in_c,out_c,kernel_size=k,stride=s,padding=p,dilation=d)
        self.abn = nn.BatchNorm2d(out_c)
        self.act_fn = nn.ReLU()
        
        if dk > 3:
            dp = 2
        else:
            dp = 1
        self.dconv = nn.Conv2d(out_c,out_c,kernel_size=dk,stride=s,padding=dp,groups=out_c)
        self.dbn = nn.BatchNorm2d(out_c)
    def forward(self,x):
        x = self.act_fn(self.abn(self.aconv(x)))
        
        x_ = self.act_fn(self.dbn(self.dconv(x)))
        
        x = torch.concat([x, x_],1)
        return x
class TFG(nn.Module):
    def __init__(self):
        super(TFG,self).__init__()
        self.agm1 = AGM_2D(in_c=1,out_c=8,k=5,dk=3,s=1,p=0,d=2)
        self.agm2 = AGM_2D(in_c=16,out_c=16,k=5,dk=3,s=1,p=0,d=2)
        self.agm3 = AGM_2D(in_c=32,out_c=32,k=3,dk=3,s=1,p=0,d=2)
        self.agm4 = AGM_2D(in_c=64,out_c=64,k=3,dk=3,s=1,p=0,d=2)
        self.agm5 = AGM_2D(in_c=128,out_c=128,k=3,dk=3,s=1,p=0,d=2)
        self.pooling = nn.AvgPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        x = self.pooling(self.agm1(x))
        x = self.agm2(x)
        x = self.agm3(x)
        x = self.agm4(x)
        x = self.agm5(x)
        x = self.gap(x).squeeze(-1)
        return x
class FDS(nn.Module):
    def __init__(self):
        super(FDS,self).__init__()
        self.conv = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=15,stride=2)
        self.bn = nn.BatchNorm1d(16)
        self.act_fn = nn.ReLU()
        self.agm1 = AGM_1D(in_c=16,out_c=16,k=13,dk=3,s=1,p=0,d=2)
        self.agm2 = AGM_1D(in_c=32,out_c=32,k=7,dk=3,s=1,p=0,d=2)
        self.agm3 = AGM_1D(in_c=64,out_c=64,k=5,dk=3,s=1,p=0,d=2)
        self.agm4 = AGM_1D(in_c=128,out_c=128,k=3,dk=3,s=1,p=0,d=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
    def forward(self,x):
        x = self.act_fn(self.bn(self.conv(x)))
        x = self.agm1(x)
        x = self.agm2(x)
        x = self.agm3(x)
        x = self.agm4(x)
        x = self.gap(x)
        return x
class TDS(nn.Module):
    def __init__(self):
        super(TDS,self).__init__()
        self.agm1 = AGM_1D(in_c=1,out_c=8,k=15,dk=3,s=1,p=0,d=2)
        self.agm2 = AGM_1D(in_c=16,out_c=16,k=13,dk=3,s=1,p=0,d=2)
        self.agm3 = AGM_1D(in_c=32,out_c=32,k=7,dk=3,s=1,p=0,d=2)
        self.agm4 = AGM_1D(in_c=64,out_c=64,k=5,dk=3,s=1,p=0,d=2)
        self.agm5 = AGM_1D(in_c=128,out_c=128,k=3,dk=3,s=1,p=0,d=2)
        self.pooling = nn.AvgPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
    def forward(self,x):
        x = self.pooling(self.agm1(x))
        x = self.agm2(x)
        x = self.agm3(x)
        x = self.agm4(x)
        x = self.agm5(x)
        x = self.gap(x)
        return x
class TGA(nn.Module):
    def __init__(self):
        super(TGA,self).__init__()
        self.sigmoid_block = nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=256,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm1d(256),
            nn.Sigmoid()
        )
        
        self.relu_block = nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=256,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm1d(256),
            nn.Sigmoid()
        )
    def forward(self,tds, tfg, fds):
        relu_tds = self.relu_block(tds)
        sigmid_tds_1 = self.sigmoid_block(tds)
        sigmid_tds_2 = self.sigmoid_block(tds)
        
        relu_tfg = self.relu_block(tfg)
        sigmid_tfg_1 = self.sigmoid_block(tfg)
        sigmid_tfg_2 = self.sigmoid_block(tfg)
        
        relu_fds = self.relu_block(fds)
        sigmid_fds_1 = self.sigmoid_block(fds)
        sigmid_fds_2 = self.sigmoid_block(fds)
        
        relu_tds = relu_tds * sigmid_tfg_1 * sigmid_fds_1
        relu_tfg = relu_tfg * sigmid_tds_1 * sigmid_fds_2
        relu_fds = relu_fds * sigmid_tfg_2 * sigmid_tds_2
        y = torch.concat([relu_tds,relu_tfg,relu_fds],1)
        return y
        
class MIMTNet(nn.Module):
    def __init__(self, output_shape):
        super(MIMTNet, self).__init__()
        self.tfg = TFG()
        self.fds = FDS()
        self.tds = TDS()
        
        self.tga = TGA()

        self.conv1 = nn.Conv1d(in_channels=768,out_channels=512,kernel_size=1,stride=1,padding=0)
        # self.bn1 = nn.BatchNorm1d(512)
        self.act_fn = nn.ReLU()
        self.fc = nn.Linear(512,output_shape)
    
    def forward(self,tfg_x,tds_x,fds_x):
        tfg = self.tfg(tfg_x)
        tds = self.tds(tds_x)
        fds = self.fds(fds_x)
        tga = self.tga(tds, tfg, fds)
        x = self.act_fn(self.conv1(tga))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    model = MIMTNet(3).eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 128
    timings=np.zeros((repetitions,1))
    
    B = 1
    C = 1
    N = 4096
    input_data2 = torch.randn((B, C, N)).float()
    input_data1 = torch.randn((B, 1, 64, 64)).float()
    input_data3 = torch.randn((B, 1, 2048)).float()

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            torch_out = model(input_data1,input_data2,input_data3)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    print('torch 모델 평균 소요 시간 : ', np.mean(np.array(timings[1:])))
    print('표준편차', np.std(np.array(timings[1:])))
    print('Max: ',np.max(np.array(timings[1:])))
    print('Min: ',np.min(np.array(timings[1:])))