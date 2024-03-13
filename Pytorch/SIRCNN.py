import numpy as np
import torch
import torch.nn as nn

class DSConv(nn.Module):
    def __init__(self, in_channel,out_channel, s, k=3):
        super(DSConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channel,in_channel,kernel_size=(k,k),stride=(s,s),padding=(1,1),groups=in_channel,bias=False)
        self.pointwise_conv = nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class IRB(nn.Module):
    def __init__(self, in_channels, out_channels, t,s):
        super(IRB, self).__init__()
        expansion_filters = in_channels * t
        self.s = s
        self.expansion_cov = nn.Conv2d(in_channels=in_channels, out_channels=expansion_filters, kernel_size=1,bias=False)
        self.expansion_bn = nn.BatchNorm2d(expansion_filters)

        self.depthwise_conv = DSConv(in_channel=expansion_filters,out_channel=expansion_filters,s=s)
        self.depthwise_bn = nn.BatchNorm2d(expansion_filters)

        self.linear_conv = nn.Conv2d(in_channels=expansion_filters, out_channels=out_channels,kernel_size=1,bias=False)
        self.linear_bn = nn.BatchNorm2d(out_channels)

        self.act_fn = nn.ReLU6()
    def forward(self,x):
        y = self.act_fn(self.expansion_bn(self.expansion_cov(x)))
        
        y = self.act_fn(self.depthwise_bn(self.depthwise_conv(y)))

        y = self.linear_bn(self.linear_conv(y))

        if self.s == 1 and x.shape[1] == y.shape[1]:
            y = x + y
        
        return y
    
class SIRCNN(nn.Module):
    def __init__(self, output_shape):
        super(SIRCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,kernel_size=(3,3),stride=(1,1),bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.block1 = IRB(in_channels=32, out_channels=16 ,t=1,s=1)

        self.block2 = IRB(in_channels=16, out_channels=24,t=6,s=2)
        self.block3 = IRB(in_channels=24, out_channels=24,t=6,s=1)

        self.block4 = IRB(in_channels=24, out_channels=96,t=6,s=1)
        self.block5 = nn.ModuleList([
            IRB(in_channels=96,out_channels=96,t=6,s=1)
            for _ in range(2)
        ])

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(96,output_shape,bias=False)
        self.act_fn = nn.ReLU6()

    def forward(self, x):
        x = self.act_fn(self.bn1(self.conv1(x)))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        for block in self.block5:
            x = block(x)

        x = self.pooling(x)
        b, c, _, _ = x.shape
        x = x.reshape(b,c)

        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = SIRCNN(3).eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 128
    timings=np.zeros((repetitions,1))


    B = 1
    C = 1
    N = 4096
    input_data = torch.randn((B, C, 64, 64)).float()

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