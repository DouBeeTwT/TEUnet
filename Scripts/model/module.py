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
                 padding: _size_2_t = 1
                 ):
        super().__init__()
        self.CBR = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
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
            CBR(out_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.double_conv2d(x)

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