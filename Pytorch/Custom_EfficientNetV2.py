import torch
import torch.nn as nn
import torch.nn.functional as F


### CBAM

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)
class CAM(nn.Module):
    def __init__(self,filters,r=16):
        super(CAM, self).__init__()
        self.filters = filters
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(filters, filters // r,bias=False),
            nn.Hardswish(True),
            nn.Linear(filters // r, filters,bias=False),
            nn.Hardsigmoid(True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
    
    def forward(self, x):
        x_avg_pool = self.avg_pool(x)
        x_avg_pool = self.mlp(self.avg_pool(x))
        x_max_pool = self.mlp(self.max_pool(x))

        channel_att = x_avg_pool + x_max_pool
        channel_att = channel_att.unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * channel_att

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,stride=1,padding=3,bias=False),
            nn.BatchNorm2d(1),
            nn.Hardsigmoid(True)
        )
        
        self.h_swish = nn.Hardswish()
        self.h_sigmoid = nn.Hardsigmoid()
    
    def forward(self,x):
        max, _ = torch.max(x,dim=1,keepdim=True)
        avg = torch.mean(x,dim=1,keepdim=True)
        concat = torch.cat((max,avg),dim=1)
        spatial_att = self.conv(concat)
        return x * spatial_att

class CBAM(nn.Module):
    def __init__(self, filters ,r=16):
        super(CBAM, self).__init__()
        self.chaanel_att = CAM(filters=filters)
        self.spatial_att = SAM()
    
    def forward(self, x):
        x = self.chaanel_att(x)
        x = self.spatial_att(x)
        return x




### EfficientNet


class Fused_MBConv(nn.Module):
    def __init__(self, in_channel, out_channel, stride,use_CBAM=True):
        super(Fused_MBConv, self).__init__()
        self.stride = stride
        self.use_CBAM = use_CBAM
        self.fused_conv = nn.Conv2d(in_channels=in_channel, out_channels=(out_channel),
                               kernel_size=3,stride=stride,padding=1,bias=False)
        self.fused_bn = nn.BatchNorm2d(out_channel)
        self.h_swish = nn.Hardswish(True)
        self.cbam = CBAM(filters=out_channel)

        self.point_wise_conv = nn.Conv2d(in_channels=(out_channel), out_channels=in_channel,
                                         kernel_size=3,stride=stride,padding=1,bias=False)
        self.point_wise_bn = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        y = self.h_swish(self.fused_bn(self.fused_conv(x)))
        if self.use_CBAM ==True:
            y = self.cbam(y)
        y = self.point_wise_bn(self.point_wise_conv(y))
        if x.shape[-1] == y.shape[-1] and self.stride == 1:
            y = x + y
        return y

class Depthwise_conv(nn.Module):
    def __init__(self, in_channel, out_channel,k,stride=None):
        super(Depthwise_conv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                                        kernel_size=k,stride=stride,groups=in_channel,bias=False)
    def forward(self, x):
        output = self.depthwise_conv(x)
        return output


class MBConv(nn.Module):
    def __init__(self, in_channel, out_channel, stride, k, use_CBAM=True):
        super(MBConv, self).__init__()
        self.use_CBAM = CBAM
        self.stride = stride
        self.expansion_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                          kernel_size=1,stride=1,bias=False)
        self.expansion_bn = nn.BatchNorm2d(out_channel)

        self.depth_wise_conv = Depthwise_conv(in_channel=(out_channel),out_channel=out_channel,
                                              k=k, stride=stride)
        self.depth_wise_bn = nn.BatchNorm2d(out_channel)

        self.cbam = CBAM(out_channel)

        self.point_wise_conv = nn.Conv2d(in_channels=(out_channel),out_channels=out_channel,
                                          kernel_size=1,stride=1,bias=False)
        self.point_wise_bn = nn.BatchNorm2d(in_channel)
        self.h_swish = nn.Hardswish(True)

    def forward(self, x):
        y = self.h_swish(self.expansion_bn(self.expansion_conv(x)))
        y = self.h_swish(self.depth_wise_bn(self.depth_wise_conv(y)))
        if self.use_CBAM == True:
            y = self.cbam(y)
        y = self.point_wise_bn(self.point_wise_conv(y))
        if self.stride == 1 and x.shape[-1] == y.shape[-1]:
            y = x + y
        return y 



class EfficientNet(nn.Module):
    def __init__(self, in_channel, filters, classes, k, stride, repeat, use_CBAM=True) -> None:
        super(EfficientNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=filters[0], kernel_size=3,
                              stride=stride[0],padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.h_swish = nn.Hardswish(True)

        self.Fused_MBConv1_list = nn.ModuleList([
            Fused_MBConv(in_channel=filters[0],out_channel=filters[1],stride=stride[1])
            for _ in range(repeat[0])
        ])

        self.Fused_MBConv2_list = nn.ModuleList([
            Fused_MBConv(in_channel=filters[1],out_channel=filters[2], stride=[2])
            for _ in range(repeat[1])
        ])
        self.Fused_MBConv3_list = nn.ModuleList([
            Fused_MBConv(in_channel=filters[2],out_channel=filters[3], stride=[3])
            for _ in range(repeat[2])
        ])

        self.MBConv1_list = nn.ModuleList([
            MBConv(in_channel=filters[3],out_channel=filters[4], stride=stride[4],k=k)
            for _ in range(repeat[3])
        ])
        self.MBConv2_list = nn.ModuleList([
            MBConv(in_channel=filters[4],out_channel=filters[5],stride=stride[5],k=k)
            for _ in range(repeat[4])
        ])
        self.MBConv3_list = nn.ModuleList([
            MBConv(in_channel=filters[5],out_channel=filters[6],stride=stride[6],k=k)
            for _ in range(repeat[5])
        ])

        self.conv2 = nn.Conv2d(in_channels=filters[6],out_channels=filters[6],kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(filters[6])
        self.globalaveragepooling = nn.AdaptiveAvgPool2d((1,1))
        self.output = nn.Linear(filters[6], classes,bias=False)

    def forward(self, x):
        y = self.h_swish(self.bn1(self.conv1(x['image'])))

        for block in self.Fused_MBConv1_list:
            y = block(y)
        
        for block in self.Fused_MBConv2_list:
            y = block(y)

        for block in self.Fused_MBConv3_list:
            y = block(y)
        
        for block in self.MBConv1_list:
            y = block(y)

        for block in self.MBConv2_list:
            y = block(y)
        
        for block in self.MBConv3_list:
            y = block(y)
        
        y = self.bn2(self.conv2(y))
        y = self.globalaveragepooling(y)
        print(y.shape)
        output = self.output(y)
        return output





    



