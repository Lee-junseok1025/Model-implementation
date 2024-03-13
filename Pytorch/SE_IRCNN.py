import numpy as np
import torch
import torch.nn as nn

class DSConv(nn.Module):
    def __init__(self, in_channel,out_channel, s, k, p):
        super(DSConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channel,in_channel,kernel_size=(k,k),stride=(s,s),padding=p,groups=in_channel,bias=False)
        self.pointwise_conv = nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
class SE_block(nn.Module):
    def __init__(self,embed_dim, r=4):
        super(SE_block,self).__init__()
        self.embed_dim = embed_dim
        self.reduction_feature = embed_dim // r
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, self.reduction_feature,bias=False),
            nn.GELU(),
            nn.Linear(self.reduction_feature, embed_dim,bias=False),
            nn.GELU()
        )
        self.sigmoid = nn.Hardsigmoid()
    def forward(self,x):
        b, c, _, _ = x.shape
        x_ = self.gap(x)
        x_ = x_.reshape(b, c)
        x_ = self.mlp(x_)
        x_ = x_.reshape(b,self.embed_dim,1,1)
        x_ = self.sigmoid(x_)
        # x_ = nn.functional.log_softmax(x_,dim=1)
        x = x * x_
        return x

class IRB(nn.Module):
    def __init__(self, in_channels, expansion_filters, out_channels, k, s, p):
        super(IRB, self).__init__()
        self.s = s
        self.expansion_cov = nn.Conv2d(in_channels=in_channels, out_channels=expansion_filters, kernel_size=1,bias=False)
        self.expansion_bn = nn.BatchNorm2d(expansion_filters)

        self.depthwise_conv = DSConv(in_channel=expansion_filters,out_channel=expansion_filters, k=k,s=s, p=p)
        self.depthwise_bn = nn.BatchNorm2d(expansion_filters)

        self.se_block = SE_block(expansion_filters)

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

class SE_IRCNN(nn.Module):
    def __init__(self, output_shape):
        super(SE_IRCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.block1 = IRB(in_channels=16, expansion_filters=16,out_channels=16, k=3 ,s=2, p=1)

        self.block2 = IRB(in_channels=16, expansion_filters=72,out_channels=24, k=3, s=2, p=1)

        self.block3 = IRB(in_channels=24, expansion_filters=88,out_channels=24, k=3, s=1, p=1)

        self.block4 = IRB(in_channels=24, expansion_filters=96,out_channels=40, k=5, s=2, p=2)
        

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Conv2d(40,48,1,bias=False)

        self.conv3 = nn.Conv2d(48,output_shape,1,bias=False)

        self.act_fn = nn.ReLU6()

    def forward(self, x):
        x = self.act_fn(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pooling(x)

        x = self.act_fn(self.conv2(x))
        x = self.conv3(x)

        b, c, _, _ = x.shape
        x = x.reshape(b,c)

        return x



    
if __name__ == "__main__":
    model = SE_IRCNN(3).eval()
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
    print('총 소요 시간', np.std(np.array(timings)))