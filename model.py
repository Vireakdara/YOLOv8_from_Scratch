'''
Yolov8 architecture
The overall architecture is BackBone + Neck + Head
'''

import torch
import torch.nn as nn

# 1. Conv: Conv2d + BatchNorm2d + SiLU
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()
                    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# 2. C2f (cross-stage partial bottleneck with 2 convolutions): Conv + Bottlenecks + Conv
# Combine high-level features with contextual information to improve detection accuracy.

# 2.1 Bottleneck : 2 stacks of Conv with shortcut of (True/False)
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.shortcut = shortcut
    
    # if the shortcut is True then it will Conv + itself else it will use only Conv
    def forward(self, x):
        x_in = x # for residual connection
        self.conv1(x)
        self.conv2(x)
        if self.shortcut:
            x = x_in + x
        return x

# 2.2 C2f : Conv + Bottleneck*N + Conv
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcut=True):
        super().__init__()
        self.num_bottlenecks = num_bottlenecks
        self.mid_channels = out_channels//2
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # sequence of bottleneck layers
        self.sequence_bottlenecks = nn.ModuleList([Bottleneck(self.mid_channels,self.mid_channels) for _ in range(num_bottlenecks)])
        
        # H x W x 0.5(n+2)c_out
        self.conv2 = Conv((num_bottlenecks+2)*self.mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # put the conv1 to x 
        x = self.conv1(x)

        # split x along channel dimension
        # x = [batch_size, out_channels, w, h]
        # :x.shape[1] // 2	Select from the start to halfway (first half)
        # x.shape[1] // 2:	Select from halfway to the end (second half)
        x1,x2=x[:,:x.shape[1]//2,:,:], x[:,x.shape[1]//2:,:,:]

        # list of outputs
        outputs = [x1, x2]  # x1 is fed through the bottlenecks
        
        for i in range(self.num_bottlenecks):
            x1 = self.sequence_bottlenecks[i](x1) 
            outputs.insert(0,x1)

        outputs = torch.cat(outputs,dim=1) # [bs, 0.5out_channels(num_bottlenecks+2),w,h]
        out = self.conv2(outputs)

        return out

# Testing
# c2f=C2f(in_channels=64,out_channels=128,num_bottlenecks=2)
# print(f"{sum(p.numel() for p in c2f.parameters())/1e6} million parameters")

# dummy_input=torch.rand((1,64,244,244))
# dummy_input=c2f(dummy_input)
# print("Output shape: ", dummy_input.shape)

# SPPF (spatial pyramid pooling fast): Conv + Maxpool2d + Conv
# Process features at various scales and pool them into a fixed-size feature map.
class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF)
    
    Args:
        in_channels  (int):  Number of input channels
        out_channels (int):  Number of output channels
        pool_kernel  (int):  Kernel size for MaxPool2d (default 5)
        
    Typically, SPPF is implemented as:
      1) A 1x1 convolution (reduce/increase channels),
      2) Three sequential MaxPool2d (with stride=1 and padding such that output size stays the same),
      3) Concat the original + each pooled feature map,
      4) A final 1x1 convolution to fuse back into out_channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels//2
        # Initial 1×1 conv
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)

        # MaxPool2d with stride=1 and appropriate padding so size remains constant
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1, ceil_mode=False)

        # Final 1×1 conv after concatenation (4x channels from x + 3 pooled outputs)
        # Example:
        # Assuming:

        # in_channels = 64
        # After the 1x1 Conv, you get 32 channels.
        # After pooling and concatenating:
        # 1 original feature map with 32 channels.
        # 3 pooled feature maps with 32 channels each.
        self.conv2 = Conv(hidden_channels*4, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # 1. Initial 1×1 conv
        x = self.conv1(x)

        # 2. Sequential pooling steps
        y1 = self.max_pool(x)
        y2 = self.max_pool(y1)
        y3 = self.max_pool(y2)

        # 3. Concat the original + each pooled output
        concat = torch.cat([x,y1,y2,y3], dim=1) # channel dimension is dim=1

        # 4. Final 1×1 conv
        outputs = self.conv2(concat)

        return outputs

# Testing
# sppf=SPPF(in_channels=128,out_channels=512)
# print(f"{sum(p.numel() for p in sppf.parameters())/1e6} million parameters")
# 0.140416 million parameters
# dummy_input=sppf(dummy_input)
# print("Output shape: ", dummy_input.shape)
# Output shape:  torch.Size([1, 512, 244, 244])


### ======= ======== ======= ###
### ======= BackBone ======= ###
### ======= ======== ======= ###

# backbone = darknet53

# return d,w,r based on version
def yolo_params(version):
    if version=='n':
        return 1/3,1/4,2.0
    elif version=='s':
        return 1/3,1/2,2.0
    elif version=='m':
        return 2/3,3/4,1.5
    elif version=='l':
        return 1.0,1.0,1.0
    elif version=='x':
        return 1.0,1.25,1.0


class Backbone(nn.Module):
    def __init__(self, version, in_channels=3, shortcut = True):
        super().__init__()
        d,w,r = yolo_params(version)

        # conv layers
        self.conv_0 = Conv(in_channels, int(64*w), kernel_size=3, stride=2, padding=1)
        self.conv_1 = Conv(int(64*w), int(128*w), kernel_size=3, stride=2, padding=1)
        self.conv_3 = Conv(int(128*w), int(256*w), kernel_size=3, stride=2, padding=1)
        self.conv_5 = Conv(int(256*w), int(512*w), kernel_size=3, stride=2, padding=1)
        self.conv_7 = Conv(int(512*w), int(512*w*r), kernel_size=3, stride=2, padding=1)

        # c2f layers
        self.c2f_2 = C2f(int(128*w), int(128*w), num_bottlenecks=int(3*d), shortcut=True)
        self.c2f_4 = C2f(int(256*w), int(256*w), num_bottlenecks=int(6*d), shortcut=True)
        self.c2f_6 = C2f(int(512*w), int(512*w), num_bottlenecks=int(6*d), shortcut=True)
        self.c2f_8 = C2f(int(512*w*r), int(512*w*r), num_bottlenecks=int(3*d), shortcut=True)

        #sppf
        self.sppf = SPPF(int(512*w*r),int(512*w*r))

    def forward(self,x):
        x=self.conv_0(x)
        x=self.conv_1(x)
        x=self.c2f_2(x)
        x=self.conv_3(x)
        out1 = self.c2f_4(x) # Keep it for output
        x=self.conv_5(out1)
        out2 = self.c2f_6(x) # Keep it for output
        x=self.conv_7(out2)
        x=self.c2f_8(x) 
        out3 = self.sppf(x) # Keep it for output

        return out1, out2, out3

print("----Nano model -----")
backbone_n=Backbone(version='n')
print(f"{sum(p.numel() for p in backbone_n.parameters())/1e6} million parameters")

print("----Small model -----")
backbone_s=Backbone(version='s')
print(f"{sum(p.numel() for p in backbone_s.parameters())/1e6} million parameters")

# Testing

x=torch.rand((1,3,640,640))
out1,out2,out3=backbone_n(x)
print(out1.shape)
print(out2.shape)
print(out3.shape)

# ----Nano model -----
# 1.272656 million parameters
# ----Small model -----
# 5.079712 million parameters
# torch.Size([1, 64, 80, 80])
# torch.Size([1, 128, 40, 40])
# torch.Size([1, 256, 20, 20])


### ======= ======== ======= ###
### =======   Neck   ======= ###
### ======= ======== ======= ###

# upsample = nearest-neighbor interpolation with scale_factor = 2
#            doesn't have trainable parameters

class Upsample(nn.Module):
    def __init__(self, scale_factor = 2 , mode = 'nearest'):
        super().__intit()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        return nn.function.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

