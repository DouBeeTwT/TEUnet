import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x
    
class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=64):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = Up()
  
        self.conv0_0 = VGGBlock(in_channels, hidden_channels, hidden_channels)
        self.conv1_0 = VGGBlock(hidden_channels, hidden_channels*2, hidden_channels*2)
        self.conv2_0 = VGGBlock(hidden_channels*2, hidden_channels*4, hidden_channels*4)
        self.conv3_0 = VGGBlock(hidden_channels*4, hidden_channels*8, hidden_channels*8)
        self.conv4_0 = VGGBlock(hidden_channels*8, hidden_channels*16, hidden_channels*16)

        self.conv0_1 = VGGBlock(hidden_channels*3, hidden_channels, hidden_channels)
        self.conv1_1 = VGGBlock(hidden_channels*6, hidden_channels*2, hidden_channels*2)
        self.conv2_1 = VGGBlock(hidden_channels*12, hidden_channels*4, hidden_channels*4)
        self.conv3_1 = VGGBlock(hidden_channels*24, hidden_channels*8, hidden_channels*8)

        self.conv0_2 = VGGBlock(hidden_channels*4, hidden_channels, hidden_channels)
        self.conv1_2 = VGGBlock(hidden_channels*8, hidden_channels*2, hidden_channels*2)
        self.conv2_2 = VGGBlock(hidden_channels*16, hidden_channels*4, hidden_channels*4)

        self.conv0_3 = VGGBlock(hidden_channels*5, hidden_channels, hidden_channels)
        self.conv1_3 = VGGBlock(hidden_channels*10, hidden_channels*2, hidden_channels*2)

        self.conv0_4 = VGGBlock(hidden_channels*6, hidden_channels, hidden_channels)

        self.final1 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.up(x1_0, x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.up(x2_0, x1_0))
        x0_2 = self.conv0_2(self.up(x1_1, torch.cat([x0_0, x0_1], 1)))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.up(x3_0, x2_0))   
        x1_2 = self.conv1_2(self.up(x2_1, torch.cat([x1_0, x1_1], 1)))
        x0_3 = self.conv0_3(self.up(x1_2, torch.cat([x0_0, x0_1, x0_2], 1)))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.up(x4_0, x3_0))
        x2_2 = self.conv2_2(self.up(x3_1, torch.cat([x2_0, x2_1], 1)))
        x1_3 = self.conv1_3(self.up(x2_2, torch.cat([x1_0, x1_1, x1_2], 1)))
        x0_4 = self.conv0_4(self.up(x1_3, torch.cat([x0_0, x0_1, x0_2, x0_3], 1)))

        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        return (output1 + output2 + output3 + output4)/4
