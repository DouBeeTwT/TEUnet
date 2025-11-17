from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TypeVar, Union, Tuple
T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

class CBR(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 1,
                 dilation: _size_2_t =1):
        super().__init__()
        self.CBR = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.CBR(x)

class DoubleConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 1,
                 p: float=0.05):
        super().__init__()
        self.double_conv2d = nn.Sequential(
            CBR(in_channels, out_channels, kernel_size, stride, padding),
            nn.Dropout2d(p),
            CBR(out_channels, out_channels, kernel_size, stride=1, padding=padding),
        )

    def forward(self, x:Tensor) -> Tensor:
        x1 = self.double_conv2d(x)
        return x1

class DoubleConv2dResConnect(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 1,
                 p: float=0.05):
        super().__init__()
        self.double_conv2d = nn.Sequential(
            CBR(in_channels, out_channels, kernel_size, stride, padding),
            nn.Dropout2d(p),
            CBR(out_channels, out_channels, kernel_size, stride=1, padding=padding),
        )
        if out_channels != in_channels:
            self.connect =nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.connect = None

    def forward(self, x:Tensor) -> Tensor:
        x1 = self.double_conv2d(x)
        if self.connect is None:
            return x1 + x
        else:
            return x1 + self.connect(x)

class UpSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 1,
                 scale_factor:int =2,
                 p:float=0.05):
        super().__init__()
        self.UpSample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            CBR(in_channels, out_channels, kernel_size, stride, padding),
            nn.Dropout2d(p)
        )
        self.DoubleConv2d = DoubleConv2d(in_channels, out_channels, kernel_size, stride, padding, p)

    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        return self.DoubleConv2d(torch.cat((self.UpSample(x1),x2),dim=1))
    
class Attention_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 1,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0):
        super().__init__()
        self.W_x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
            nn.BatchNorm2d(out_channels)
            )
        
        self.W_x2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
            nn.BatchNorm2d(out_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        psi = self.psi(self.relu(self.W_x1(x1)+self.W_x2(x2)))
        return x2*psi
    
class AttUpSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 1,
                 scale_factor:int =2,
                 p:float=0.05):
        super().__init__()
        self.UpSample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            CBR(in_channels, out_channels, kernel_size, stride, padding),
            nn.Dropout2d(p)
        )
        self.attention = Attention_block(out_channels, out_channels//2, kernel_size=1, stride=1, padding=0)
        self.DoubleConv2d = DoubleConv2d(in_channels, out_channels, kernel_size, stride, padding, p)

    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        x1 = self.UpSample(x1)
        x2 = self.attention(x1,x2)
        return self.DoubleConv2d(torch.cat((x1,x2),dim=1))
    
def expend_as(tensor, rep):
    """Repeat elements along the channel dimension"""
    return tensor.repeat(1, 1, 1, rep)

class ChannelBlock(nn.Module):
    """Channel attention block"""
    def __init__(self,
                 in_channels:int,
                 out_channels:int):
        super(ChannelBlock, self).__init__()
        self.out_channels = out_channels

        self.CBR1 = CBR(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.CBR2 = CBR(in_channels, out_channels, kernel_size=5, padding=2)
        self.CBR3 = CBR(out_channels*2, out_channels, kernel_size=1, padding=0)

        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.Attention = nn.Sequential(
            nn.Linear(out_channels*2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1 = self.CBR1(x)
        x2 = self.CBR2(x)
        x3 = self.AvgPool(torch.cat([x1, x2], dim=1))
        x3 = x3.view(x3.size(0), -1)
        att = self.Attention(x3)
        att = att.view(att.size(0), self.out_channels, 1, 1)
        x4 = self.CBR3(torch.cat([x1*att, x2*(1-att)], dim=1))
        return x4

class SpatialBlock(nn.Module):
    """Spatial attention block"""
    def __init__(self,
                 in_channels:int,
                 out_channels:int):
        super(SpatialBlock, self).__init__()
        self.CBR1 = CBR(in_channels, out_channels, kernel_size=3, padding=1)
        self.CBR2 = CBR(out_channels, out_channels, kernel_size=1, padding=0)
        self.CBR3 = CBR(out_channels*2, out_channels, kernel_size=3, padding=1)
        self.Attention = nn.Sequential(
            nn.Conv2d(out_channels*2, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        x1 = self.CBR2(self.CBR1(x1))
        att = self.Attention(torch.cat([x1, x2], dim=1))
        att = att.expand_as(x1)
        x3 = self.CBR3(torch.cat([x2*att, x1*(1-att)], dim=1))   
        return x3

class HAAM(nn.Module):
    """Hierarchical Attention Aggregation Module"""
    def __init__(self,
                 in_channels:int,
                 out_channels:int):
        super(HAAM, self).__init__()
        self.channel_block = ChannelBlock(in_channels, out_channels)
        self.spatial_block = SpatialBlock(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        channel_data = self.channel_block(x)
        haam_data = self.spatial_block(x, channel_data)
        haam_data = self.relu(self.bn(haam_data))
        return haam_data

class DoubleHAAM(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 p:float=0.05):
        super().__init__()
        self.haam1= HAAM(in_channels, out_channels)
        self.dropout = nn.Dropout2d(p)
        self.haam2= HAAM(out_channels, out_channels)

    def forward(self, x):
        return self.haam2(self.dropout(self.haam1(x)))
    
class HAAMUpSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 1,
                 scale_factor:int =2,
                 p:float=0.05):
        super().__init__()
        self.UpSample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            CBR(in_channels, out_channels, kernel_size, stride, padding),
            nn.Dropout2d(p)
        )
        self.DoubleHAAM = DoubleHAAM(in_channels, out_channels, p)

    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        return self.DoubleHAAM(torch.cat((self.UpSample(x1),x2),dim=1))
    
class EntropyBlock(nn.Module):
    def __init__(self,
                 in_channels:int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels//4),
            nn.ReLU(),
            nn.Linear(in_channels//4, in_channels)
        )
        self.sigmod = nn.Sigmoid()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels,1,kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    
    def entorpy(self, x):
        x = torch.clamp(x, 1e-7, 1-1e-7)
        return -x*torch.log2(x)-(1-x)*torch.log2(1-x)
    
    def forward(self, x, a):
        size = x.shape[-1]
        # Channel Attention
        x1 = x * self.sigmod(self.mlp(nn.MaxPool2d(size)(x).squeeze())+self.mlp(nn.AvgPool2d(size)(x).squeeze())).unsqueeze(2).unsqueeze(3)
        # Entropy Space Attention
        x1 = x1 * (self.entorpy(a) > 0.5)
        # Update Space Attention Map
        a = F.interpolate(a, scale_factor=2, mode='bilinear', align_corners=False)
        # Reside Connect
        x1 = x1 + x

        return x1, a

class EntropyUpSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 3,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 1,
                 scale_factor:int =2,
                 p:float = 0.05):
        super().__init__()
        self.UpSample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            CBR(in_channels, out_channels, kernel_size, stride, padding),
        )
        self.EntropyAttention = EntropyBlock(out_channels)
        self.DoubleConv2d = nn.Sequential(
            DoubleConv2dResConnect(in_channels, out_channels, kernel_size, stride, padding, p),
            DoubleConv2dResConnect(out_channels, out_channels, kernel_size, stride, padding, p)
            )

    def forward(self, x1:Tensor, x2:Tensor, a:Tensor) -> Tensor:
        x1 = self.UpSample(x1)
        x2, a = self.EntropyAttention(x2, a)
        x2 = self.DoubleConv2d(torch.cat((x1, x2), dim=1))
        return x2, a