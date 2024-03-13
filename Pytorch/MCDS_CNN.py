import numpy as np
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, channels, embed_dim):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patchs = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels=channels, out_channels= embed_dim,
                              kernel_size=patch_size, stride= patch_size, padding=0,bias=False
                              )
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(embed_dim)
    def forward(self,x):
        x = self.bn(self.gelu(self.proj(x)))
        return x
    
class DSConv(nn.Module):
    def __init__(self, in_channel, k, s):
        super(DSConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channel,in_channel,kernel_size=k,stride=1, groups=in_channel,padding=1,bias=False)
        self.pointwise_conv = nn.Conv2d(in_channel,in_channel,kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class Light_unit(nn.Module):
    def __init__(self, in_channels, out_channels, s=0):
        super(Light_unit, self).__init__()
        self.s = s

        self.depthwise_conv = DSConv(in_channel=in_channels, k=3, s=1)
        self.depthwise_bn = nn.BatchNorm2d(in_channels)

        self.linear_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1,bias=False)
        self.linear_bn = nn.BatchNorm2d(out_channels)

        self.act_fn = nn.GELU()
    def forward(self,x):
        
        y = self.act_fn(self.depthwise_bn(self.depthwise_conv(x)))

        y = x + y

        y = self.act_fn(self.linear_bn(self.linear_conv(y)))
        return y
    
    
class MCDS_CNN(nn.Module):
    def __init__(self, output_shape):
        super(MCDS_CNN, self).__init__()
        self.patch_embed = PatchEmbed(img_size=96,patch_size=16,channels=1,embed_dim=128)
        self.pooling = nn.AdaptiveAvgPool2d(1)


        self.block = nn.ModuleList([
            Light_unit(in_channels=128,out_channels=128,s=0)
        for _ in range(2)
        ])

        self.linear = nn.Linear(128,output_shape,bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.patch_embed(x)

        for block in self.block:
            x = block(x)

        x = self.pooling(x)
        b, c, _, _ = x.shape
        x = x.reshape(b,c)

        x = self.linear(x)
        return x

    
    
if __name__ == "__main__":
    model = MCDS_CNN(3).eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 128
    timings=np.zeros((repetitions,1))


    B = 1
    C = 1
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
    print('Max: ',np.max(np.array(timings)))
    print('Min: ',np.min(np.array(timings)))