import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .module import DoubleConv2d, EntropyBlock, EntropyUpSample, DoubleConv2dResConnect

class TEUnet2_Encoder(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 hidden_channels:int=32,
                 p:float = 0.05,
                 backbone:str = "res"):
        super().__init__()
        if backbone == "res":
            self.DownSampleLayer1 = DoubleConv2dResConnect(in_channels=in_channels, out_channels=hidden_channels, p=p)
            self.DownSampleLayer2 = nn.Sequential(
                DoubleConv2dResConnect(in_channels=hidden_channels, out_channels=hidden_channels*2, stride=2, p=p),
                *[DoubleConv2dResConnect(in_channels=hidden_channels*2, out_channels=hidden_channels*2, p=p) for _ in range(2)]
                )
            self.DownSampleLayer3 = nn.Sequential(
                DoubleConv2dResConnect(in_channels=hidden_channels*2, out_channels=hidden_channels*4, stride=2, p=p),
                *[DoubleConv2dResConnect(in_channels=hidden_channels*4, out_channels=hidden_channels*4, p=p) for _ in range(3)]
                )
            self.DownSampleLayer4 = nn.Sequential(
                DoubleConv2dResConnect(in_channels=hidden_channels*4, out_channels=hidden_channels*8, stride=2, p=p),
                *[DoubleConv2dResConnect(in_channels=hidden_channels*8, out_channels=hidden_channels*8, p=p) for _ in range(5)]
                )
            self.DownSampleLayer5 = nn.Sequential(
                DoubleConv2dResConnect(in_channels=hidden_channels*8, out_channels=hidden_channels*16, stride=2, p=p),
                *[DoubleConv2dResConnect(in_channels=hidden_channels*16, out_channels=hidden_channels*16, p=p) for _ in range(2)]
                )
        else:
            self.DownSampleLayer1 = DoubleConv2d(in_channels=in_channels, out_channels=hidden_channels, p=p)
            self.DownSampleLayer2 = DoubleConv2d(in_channels=hidden_channels, out_channels=hidden_channels*2, p=p)
            self.DownSampleLayer3 = DoubleConv2d(in_channels=hidden_channels*2, out_channels=hidden_channels*4, p=p)
            self.DownSampleLayer4 = DoubleConv2d(in_channels=hidden_channels*4, out_channels=hidden_channels*8, p=p)
            self.DownSampleLayer5 = DoubleConv2d(in_channels=hidden_channels*8, out_channels=hidden_channels*16, p=p)
            self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.AttentionHead = nn.Sequential(
            nn.Conv2d(hidden_channels*16, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x:Tensor) -> List:
        # encoding path
        x1 = self.DownSampleLayer1(x)
        x2 = self.DownSampleLayer2(x1)
        x3 = self.DownSampleLayer3(x2)
        x4 = self.DownSampleLayer4(x3)
        x5 = self.DownSampleLayer5(x4)

        a1 = self.AttentionHead(x5)
        
        return [a1, x1, x2, x3, x4, x5]

class TEUnet2_Decoder(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 hidden_channels:int=32,
                 p:float = 0.05,
                 show_attention:bool=False):
        super().__init__()
        self.show_attention = show_attention
        self.EntropyBlock = EntropyBlock(in_channels=hidden_channels*16)

        self.UpSampleLayer5 = EntropyUpSample(in_channels=hidden_channels*16, out_channels=hidden_channels*8)
        self.UpSampleLayer4 = EntropyUpSample(in_channels=hidden_channels*8, out_channels=hidden_channels*4)
        self.UpSampleLayer3 = EntropyUpSample(in_channels=hidden_channels*4, out_channels=hidden_channels*2)
        self.UpSampleLayer2 = EntropyUpSample(in_channels=hidden_channels*2, out_channels=hidden_channels*1)
        
        self.SegmentationHead = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
            )
    def entorpy(self, x):
        x = torch.clamp(x, 1e-7, 1-1e-7)
        return -x*torch.log2(x)-(1-x)*torch.log2(1-x)
    
    def forward(self, x_list:List) -> List:
        a1, x1, x2, x3, x4, x5 = x_list[0],x_list[1],x_list[2],x_list[3],x_list[4],x_list[5]
        # decoding path
        a1 = self.entorpy(a1)
        x5, a2 = self.EntropyBlock(x5, a1)
        
        #x5, a2 = x5*a1, a1
        x4, a3 = self.UpSampleLayer5(x5, x4, a2)
        x3, a4 = self.UpSampleLayer4(x4, x3, a3)
        x2, a5 = self.UpSampleLayer3(x3, x2, a4)
        x1, a6 = self.UpSampleLayer2(x2, x1, a5)

        # Segmentation
        x1 = self.SegmentationHead(x1)
        if self.show_attention:
            return [x1, a1, a2, a3, a4, a5]
        else:
            return x1

class TEUnet2(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 hidden_channels:int=32,
                 p:float = 0.05,
                 backbone:str = "res",
                 show_attention:bool=False):
        super().__init__()
        self.encoder = TEUnet2_Encoder(in_channels, out_channels, hidden_channels, p, backbone)
        self.decoder = TEUnet2_Decoder(in_channels, out_channels, hidden_channels, p, show_attention)

    def forward(self, x:Tensor) -> List:
        x_list = self.encoder(x)
        return self.decoder(x_list)