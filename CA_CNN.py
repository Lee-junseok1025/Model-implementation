import numpy as np
import torch
import torch.nn as nn

class DSConv(nn.Module):
    def __init__(self, in_channel,out_channel, s, p=1, k=3):
        super(DSConv, self).__init__()
        if k > 3:
            p = 2
        self.depthwise_conv = nn.Conv2d(in_channel,in_channel,kernel_size=(k,k),stride=(s,s),padding=p,groups=in_channel,bias=False)
        self.pointwise_conv = nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class CA(nn.Module):
    def __init__(self, in_channel,r=32):
        super(CA, self).__init__()
        reduction_channel = max(8, in_channel//r)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=reduction_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(reduction_channel)

        self.conv_h = nn.Conv2d(in_channels=reduction_channel,out_channels=in_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn_h = nn.BatchNorm2d(in_channel)

        self.conv_w = nn.Conv2d(in_channels=reduction_channel,out_channels=in_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn_w = nn.BatchNorm2d(in_channel)

        self.act_fn = nn.Hardswish()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        concat_tensor = torch.cat([x_h,x_w],2)

        y = self.act_fn(self.bn1(self.conv1(concat_tensor)))
        
        x_h, x_w = torch.split(y,[h, w],dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.sigmoid(self.conv_h(x_h))
        x_w = self.sigmoid(self.conv_w(x_w))

        y = x * x_h * x_w
        return y


class IRB(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_filters, k, s, attn=True, activation='HS'):
        super(IRB, self).__init__()
        self.s = s
        self.attn = attn
        self.expansion_cov = nn.Conv2d(in_channels=in_channels, out_channels=expansion_filters, kernel_size=(1,1),bias=False)
        self.expansion_bn = nn.BatchNorm2d(expansion_filters)

        self.depthwise_conv = DSConv(in_channel=expansion_filters,out_channel=expansion_filters,k=k,s=s)
        self.depthwise_bn = nn.BatchNorm2d(expansion_filters)

        self.coordinate_attn = CA(in_channel=expansion_filters)

        self.linear_conv = nn.Conv2d(in_channels=expansion_filters, out_channels=out_channels,kernel_size=(1,1),bias=False)
        self.linear_bn = nn.BatchNorm2d(out_channels)

        self.act_fn = nn.ReLU6() if activation == 'RE' else nn.Hardswish()
    def forward(self,x):
        y = self.act_fn(self.expansion_bn(self.expansion_cov(x)))
        
        y = self.act_fn(self.depthwise_bn(self.depthwise_conv(y)))

        if self.attn:
            y = self.coordinate_attn(y)

        y = self.linear_bn(self.linear_conv(y))

        if self.s == 1 and x.shape[1] == y.shape[1]:
            y = x + y
        
        return y

class CA_CNN(nn.Module):
    def __init__(self, output_shape):
        super(CA_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,kernel_size=(3,3),stride=(2,2),bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.h_swish = nn.Hardswish()

        self.block1 = IRB(in_channels=16, out_channels=16, expansion_filters=16, k=3, s=2, activation='RE')

        self.block2 = IRB(in_channels=16, out_channels=24, expansion_filters=36, k=3, s=2, attn=False, activation='RE')

        self.block3 = IRB(in_channels=24, out_channels=24, expansion_filters=44, k=3, s=1, attn=False, activation='RE')

        self.block4 = IRB(in_channels=24, out_channels=40, expansion_filters=48, k=5, s=2)
        self.block5 = IRB(in_channels=40, out_channels=40, expansion_filters=120, k=5, s=1)
        self.block6 = IRB(in_channels=40, out_channels=40, expansion_filters=120, k=5,s=1)

        self.block7 = IRB(in_channels=40, out_channels=48, expansion_filters=60, k=5, s=1)
        self.block8 = IRB(in_channels=48, out_channels=48, expansion_filters=72, k=5, s=2)

        self.block9 = IRB(in_channels=48, out_channels=96, expansion_filters=144, k=5, s=1)
        self.block10 = IRB(in_channels=96, out_channels=96, expansion_filters=288, k=5, s=1)
        self.block11 = IRB(in_channels=96, out_channels=96, expansion_filters=288, k=5, s=1)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=288, kernel_size=(1,1), bias=False)

        self.conv3 = nn.Conv2d(in_channels=288, out_channels=512,kernel_size=(1,1), bias=False)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=output_shape, kernel_size=(1,1), bias=False)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(1)
    def forward(self, x):
        x = self.h_swish(self.bn1(self.conv1(x)))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.h_swish(self.conv2(x))
        x = self.pooling(x)
        x = self.h_swish(self.conv3(x))
        x = self.conv4(x)
        b, c, _, _ = x.shape
        x = x.reshape(b,c)
        return x



if __name__ == "__main__":
    model = CA_CNN(3).eval()
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