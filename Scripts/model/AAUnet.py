from torch import Tensor
import torch.nn as nn
from .module import DoubleHAAM, HAAMUpSample

class AAUnet(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 hidden_channels:int=32,
                 p:float = 0.05):
        super().__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.DownSampleLayer1 = DoubleHAAM(in_channels=in_channels, out_channels=hidden_channels, p=p)
        self.DownSampleLayer2 = DoubleHAAM(in_channels=hidden_channels, out_channels=hidden_channels*2, p=p)
        self.DownSampleLayer3 = DoubleHAAM(in_channels=hidden_channels*2, out_channels=hidden_channels*4, p=p)
        self.DownSampleLayer4 = DoubleHAAM(in_channels=hidden_channels*4, out_channels=hidden_channels*8, p=p)
        self.DownSampleLayer5 = DoubleHAAM(in_channels=hidden_channels*8, out_channels=hidden_channels*16, p=p)

        self.UpSampleLayer5 = HAAMUpSample(in_channels=hidden_channels*16, out_channels=hidden_channels*8)
        self.UpSampleLayer4 = HAAMUpSample(in_channels=hidden_channels*8, out_channels=hidden_channels*4)
        self.UpSampleLayer3 = HAAMUpSample(in_channels=hidden_channels*4, out_channels=hidden_channels*2)
        self.UpSampleLayer2 = HAAMUpSample(in_channels=hidden_channels*2, out_channels=hidden_channels*1)

        self.SegmentationHead = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
            )

    def forward(self, x:Tensor) -> Tensor:
        # encoding path
        x1 = self.DownSampleLayer1(x)
        x2 = self.MaxPool(x1)
        x2 = self.DownSampleLayer2(x2)
        x3 = self.MaxPool(x2)
        x3 = self.DownSampleLayer3(x3)
        x4 = self.MaxPool(x3)
        x4 = self.DownSampleLayer4(x4)
        x5 = self.MaxPool(x4)
        x5 = self.DownSampleLayer5(x5)

        # decoding path
        x4 = self.UpSampleLayer5(x5, x4)
        x3 = self.UpSampleLayer4(x4, x3)
        x2 = self.UpSampleLayer3(x3, x2)
        x1 = self.UpSampleLayer2(x2, x1)

        # Segmentation
        x1 = self.SegmentationHead(x1)

        return x1