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

# x=torch.rand((1,3,640,640))
# out1,out2,out3=backbone_n(x)
# print(out1.shape)
# print(out2.shape)
# print(out3.shape)

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
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class Neck(nn.Module):
    def __init__(self, version):
        super().__init__()
        d,w,r = yolo_params(version)

        self.up=Upsample() # No trainable parameters
        self.c2f_1=C2f(in_channels=int(512*w*(1+r)), out_channels=int(512*w),num_bottlenecks=int(3*d),shortcut=False)
        self.c2f_2=C2f(in_channels=int(768*w), out_channels=int(256*w),num_bottlenecks=int(3*d),shortcut=False)
        self.c2f_3=C2f(in_channels=int(768*w), out_channels=int(512*w),num_bottlenecks=int(3*d),shortcut=False)
        self.c2f_4=C2f(in_channels=int(512*w*(1+r)), out_channels=int(512*w*r),num_bottlenecks=int(3*d),shortcut=False)

        self.cv_1=Conv(in_channels=int(256*w),out_channels=int(256*w),kernel_size=3,stride=2, padding=1)
        self.cv_2=Conv(in_channels=int(512*w),out_channels=int(512*w),kernel_size=3,stride=2, padding=1)

    def forward(self, x_res_1, x_res_2, x): # x_res_1, x_res_2, x = ouptut of the backbone 
        res_1 = x               # for residual connection
        x = self.up(x)
        x = torch.cat([x, x_res_2], dim=1)

        res_2 = self.c2f_1(x)   # for residual connection

        x = self.up(res_2)
        x = torch.cat([x, x_res_1], dim=1)

        out_1 = self.c2f_2(x)

        x = self.cv_1(out_1)

        x = torch.cat([x, res_2], dim=1)
        out_2 = self.c2f_3(x)

        x = self.cv_2(out2)

        x = torch.cat([x, res_1], dim=1)
        out_3 = self.c2f_4(x)

        return out_1,out_2,out_3

# Testing
neck=Neck(version='n')
print(f"{sum(p.numel() for p in neck.parameters())/1e6} million parameters")

x=torch.rand((1,3,640,640))
out1,out2,out3=Backbone(version='n')(x)
out_1,out_2,out_3=neck(out1,out2,out3)
print(out_1.shape)
print(out_2.shape)
print(out_3.shape)

### ======= ======== ======= ###
### =======   Head   ======= ###
### ======= ======== ======= ###

# Consist of 3 modules: (1) bbox coordinates, (2) classification scores, (3) distribution focal loss (DFL).

# DFL considers the predicted bbox coordinates as a 
# probability distribution. At inference time, it samples from the distribution to get refined coordinates 

# DFL
class DFL(nn.Module):
    def __init__(self,ch=16):
        super().__init__()
        
        self.ch=ch
        
        self.conv=nn.Conv2d(in_channels=ch,out_channels=1,kernel_size=1,bias=False).requires_grad_(False)
        
        # initialize conv with [0,...,ch-1]
        x=torch.arange(ch,dtype=torch.float).view(1,ch,1,1)
        self.conv.weight.data[:]=torch.nn.Parameter(x) # DFL only has ch parameters

    def forward(self,x):
        # x must have num_channels = 4*ch: x=[bs,4*ch,c]
        b,c,a=x.shape                           # c=4*ch
        x=x.view(b,4,self.ch,a).transpose(1,2)  # [bs,ch,4,a]

        # take softmax on channel dimension to get distribution probabilities
        x=x.softmax(1)                          # [b,ch,4,a]
        x=self.conv(x)                          # [b,1,4,a]
        return x.view(b,4,a)                    # [b,4,a]

# Testing
# dummy_input=torch.rand((1,64,128))
# dfl=DFL()
# print(f"{sum(p.numel() for p in dfl.parameters())} parameters")

# dummy_output=dfl(dummy_input)
# print(dummy_output.shape)

# print(dfl)

class Head(nn.Module):
    def __init__(self,version,ch=16,num_classes=80):

        super().__init__()
        self.ch=ch                          # dfl channels
        self.coordinates=self.ch*4          # number of bounding box coordinates 
        self.nc=num_classes                 # 80 for COCO
        self.no=self.coordinates+self.nc    # number of outputs per anchor box

        self.stride=torch.zeros(3)          # strides computed during build
        
        d,w,r=yolo_params(version=version)
        
        # for bounding boxes
        self.box=nn.ModuleList([
            nn.Sequential(Conv(int(256*w),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w*r),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1))
        ])

        # for classification
        self.cls=nn.ModuleList([
            nn.Sequential(Conv(int(256*w),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w*r),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1))
        ])

        # dfl
        self.dfl=DFL()

    def forward(self,x):
        # x = output of Neck = list of 3 tensors with different resolution and different channel dim
        #     x[0]=[bs, ch0, w0, h0], x[1]=[bs, ch1, w1, h1], x[2]=[bs,ch2, w2, h2] 

        for i in range(len(self.box)):       # detection head i
            box=self.box[i](x[i])            # [bs,num_coordinates,w,h]
            cls=self.cls[i](x[i])            # [bs,num_classes,w,h]
            x[i]=torch.cat((box,cls),dim=1)  # [bs,num_coordinates+num_classes,w,h]

        # in training, no dfl output
        if self.training:
            return x                         # [3,bs,num_coordinates+num_classes,w,h]
        
        # in inference time, dfl produces refined bounding box coordinates
        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))

        # concatenate predictions from all detection layers
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2) #[bs, 4*self.ch + self.nc, sum_i(h[i]w[i])]
        
        # split out predictions for box and cls
        #           box=[bs,4×self.ch,sum_i(h[i]w[i])]
        #           cls=[bs,self.nc,sum_i(h[i]w[i])]
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)


        a, b = self.dfl(box).chunk(2, 1)  # a=b=[bs,2×self.ch,sum_i(h[i]w[i])]
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        
        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)


    def make_anchors(self, x, strides, offset=0.5):
        # x= list of feature maps: x=[x[0],...,x[N-1]], in our case N= num_detection_heads=3
        #                          each having shape [bs,ch,w,h]
        #    each feature map x[i] gives output[i] = w*h anchor coordinates + w*h stride values
        
        # strides = list of stride values indicating how much 
        #           the spatial resolution of the feature map is reduced compared to the original image

        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # x coordinates of anchor centers
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # y coordinates of anchor centers
            sy, sx = torch.meshgrid(sy, sx)                                # all anchor centers 
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)


detect=Head(version='n')
print(f"{sum(p.numel() for p in detect.parameters())/1e6} million parameters")

# out_1,out_2,out_3 are output of the neck
output=detect([out_1,out_2,out_3])
print(output[0].shape)
print(output[1].shape)
print(output[2].shape)

print(detect)
