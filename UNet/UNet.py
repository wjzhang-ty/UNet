import torch
import torch.nn as nn
import torch.nn.functional as F

# UNet 主体结构
class UNet(nn.Module):
    def __init__(self,in_cannel,n_classes) -> None:
        super(UNet, self).__init__()

        # 下采样
        self.start = DoubleConv(in_cannel,64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 上采样
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # 最后一层1*1卷积
        self.end = nn.Conv2d(
            in_channels=64, out_channels=n_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x1 = self.start(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        return self.end(x)

# (卷积+BN+relu)*2
class DoubleConv(nn.Module):
    def __init__(self,inCannel,outCannel) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=inCannel,
                out_channels=outCannel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(outCannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=outCannel,
                out_channels=outCannel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(outCannel),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.conv(x)

# 下采样
class Down(nn.Module):
    def __init__(self,inCannel,outCannel) -> None:
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(inCannel,outCannel)
    
    def forward(self,x):
        x = self.down(x)
        return self.conv(x)
        
# 上采样
class Up(nn.Module):
    def __init__(self,inCannel,outCannel) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(inCannel, outCannel, kernel_size=2, stride=2)
        self.conv = DoubleConv(inCannel,outCannel)
    
    def forward(self,x,connection):
        x=self.up(x)

        # 跨层连接
        x = torch.cat([x, connection], dim=1)

        return self.conv(x)
        