import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, group=1, activation=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()
                    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Bottleneck : 2 stacks of Conv with shortcut of (True/False)
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, shortcut=True)
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.shortcut = shortcut
    
    # if the shortcut is True then it will Conv + itself else it will use only Conv
    def forward(self, x)
        x_in = x # for residual connection
        self.conv1(x)
        self.conv2(x)
        if self.shortcut:
            x = x_in + x
        return x



