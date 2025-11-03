from torch import Tensor
import torch.nn as nn
from .module import DoubleConv2d, UpSample

class TEUnet(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 hidden_channels:int=64,
                 p:float = 0.05):
        super().__init__()

    def forward(self, x:Tensor) -> Tensor:
        return x