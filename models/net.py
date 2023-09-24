import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

# Adapted from https://github.com/biubug6/Pytorch_Retinaface
def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

# Adapted from https://github.com/biubug6/Pytorch_Retinaface
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

# Adapted from https://github.com/biubug6/Pytorch_Retinaface
class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn(in_channels_list[0], out_channels, leaky = leaky)
        self.output2 = conv_bn(in_channels_list[1], out_channels, leaky = leaky)
        self.output3 = conv_bn(in_channels_list[2], out_channels, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)
        
    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



def conv_bn1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride=stride, padding=1, groups=2, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bnr(inp, oup, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def ydsb(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class inception01(nn.Module):
    def __init__(self, in_channel=32):
        super(inception01,self).__init__()
        self.conv1x1_1 = conv_bnr(in_channel, in_channel, 1, 0)
        self.conv1x1_2 = conv_bnr(in_channel, in_channel//4, 1, 0)
        
        self.conv3x3 = conv_bnr(in_channel, in_channel//4, 3, 1)
        self.conv5x5 = conv_bnr(in_channel, in_channel//4, 5, 2)
        self.conv7x7 = conv_bnr(in_channel, in_channel//4, 7, 3)
        
        self.conv1x1_3 = conv_bnr(in_channel, in_channel, 3, 1)
        
    def forward(self, input):
        out1x1_1 = self.conv1x1_1(input)
        out1x1_2 = self.conv1x1_2(input)
        
        out3x3 = self.conv3x3(out1x1_1)
        out5x5 = self.conv5x5(out1x1_1)
        out7x7 = self.conv7x7(out1x1_1)
        out = torch.cat([out1x1_2, out3x3, out5x5, out7x7], dim=1)
        out = self.conv1x1_3(out)
        out = out + input
        
        return out
        


class LiteNet(nn.Module):
    def __init__(self):
        super(LiteNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)#320X320
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)#320X320
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)#160X160
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)#160X160
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool1 = ydsb(128, 128)#80X80
        self.stage1_1 = inception01(128)
        self.stage1_2 = inception01(128)
        self.stage1 = inception01(128)
        self.pool2 = ydsb(128, 256)#40X40
        self.stage2_1 = inception01(256)
        self.stage2 = inception01(256)
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_bnr(256, 512, 3, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)
        x = self.pool1(x)
        x = self.stage1_1(x)
        x = self.stage1_2(x)
        x = self.stage1(x)
        x = self.pool2(x)
        x = self.stage2_1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x
