import numpy as np
import torch
import torch.nn as nn

class signalembed(nn.Module):
    def __init__(self,in_channels, out_channels, s=2, last=False):
        super(signalembed, self).__init__()
        self.last = last
        if s > 1:
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=9, stride=s,padding=2,bias=False)
        else:
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7,padding=2,bias=False)
        self.act_fn = nn.ReLU() if last == False else nn.Tanh()
        self.pooling = nn.MaxPool1d(2,padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act_fn(x)
        x = self.pooling(x)
        return x

class signalembed_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = signalembed(in_channels=1,out_channels=8)
        self.block2 = signalembed(in_channels=8,out_channels=16)
        self.block3 = signalembed(in_channels=16,out_channels=32,s=1)
        self.block4 = signalembed(in_channels=32, out_channels=64, s=1, last=True)
        #self.block4 = signalembed(in_channels=32,out_channels=64)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.unsqueeze(1)

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,bias=False)
    
    def forward(self,x):
        max_x, _ = torch.max(x,1)
        avg_x = torch.mean(x,1).unsqueeze(1)

        concat_tensor = torch.cat([max_x.unsqueeze(1),avg_x],1)

        y = self.conv(concat_tensor).sigmoid()

        y = x * y
        return y

class CAM(nn.Module):
    def __init__(self,in_channels, split):
        super(CAM, self).__init__()
        N = int(in_channels / split)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Linear(in_features=in_channels, out_features=N, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()

        x = self.pooling(x)
        x = x.reshape(b, c)
        x = self.mlp(x).softmax(1)
        return x.unsqueeze(2).unsqueeze(3)

class split_cnn(nn.Module):
    def __init__(self,in_channels, out_channels, split, use_split=True):
        super().__init__()
        self.use_split = use_split
        if use_split:
            N = in_channels // split
            self.N = N
            self.block = nn.ModuleList([
                nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1,bias=False)
            for _ in range(N*split)
            ])
            self.cam = CAM(in_channels=in_channels, split=split)
        else:
            self.block = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)

            self.cam = CAM(in_channels=in_channels, split=split)
        self.sam = SAM()

        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_fn = nn.ReLU()
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        c_attn = self.cam(x)
        if self.use_split:
            x_split_tensor = torch.split(x, self.N, 1)
            x_hat = torch.zeros_like(x_split_tensor[0])

            for x_, block in zip(x_split_tensor, self.block):
                x = block(x_)
                x = x * c_attn
                x_hat = torch.cat([x_hat, x], 1)
        
            x = self.sam(x_hat[:,self.N:])
        else:
            x = self.block(x)
            x = self.sam(x)
        x = self.act_fn(self.bn(self.conv(x)))
        x = self.pooling(x)
        return x

class LEFE_Net(nn.Module):
    def __init__(self,output_shape):
        super(LEFE_Net, self).__init__()
        self.signal2embed = signalembed_block()

        self.split_cnn1 = split_cnn(in_channels=1, out_channels=16,split=1, use_split=False)
        self.split_cnn2 = split_cnn(in_channels=16, out_channels=32, split=8)
        self.split_cnn3 = split_cnn(in_channels=32, out_channels=64, split=8)
        self.split_cnn4 = split_cnn(in_channels=64, out_channels=128, split=8)

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128, 256, bias=False)
        self.fc2 = nn.Linear(256,output_shape, bias=False)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.signal2embed(x)
        x = self.split_cnn1(x)
        x = self.split_cnn2(x)
        x = self.split_cnn3(x)
        x = self.split_cnn4(x)

        x = self.pooling(x)
        b, c, _, _ = x.size()
        x = x.reshape(b, c)

        x = self.act_fn(self.fc1(x))
        x = self.fc2(x)
        return x

        
    
if __name__ == "__main__":
    model = LEFE_Net(3).eval()
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
    print('총 소요 시간', np.std(np.array(timings)))