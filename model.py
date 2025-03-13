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




