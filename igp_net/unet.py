

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,padding=1,bias=False), # downsample
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, cin, cout=None, kernel_size=4,\
                stride=2, padding = 1):
        super().__init__()
        cout = cout or cin / 2

        self.conv1 = nn.ConvTranspose2d(cin, cin//2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.double_conv = DoubleConv(cin, cout)


    def forward(self, x1,x2):
        """
        args:
            x1: N, cin, H, W
            x2: N, cin/2, H, W
        """
        out = self.conv1(x1)
        out = torch.cat([out, x2], dim=1) # N, cin, H, W
        out = self.double_conv(out)
        return out
    

class UNetBackBone(nn.Module):

    def __init__(self, cin:int, cout:int, c0:int=64):
        super().__init__()
    
        self.inc = (DoubleConv(cin, c0))
        self.down1 = (Down(c0, 2*c0))
        self.down2 = (Down(c0*2, c0*4))
        self.down3 = (Down(c0*4, c0*8))
        self.down4 = (Down(c0*8, c0*16))
        self.up1 = (Up(c0*16, c0*8))
        self.up2 = (Up(c0*8, c0*4))
        self.up3 = (Up(c0*4, c0*2))
        self.up4 = (Up(c0*2, c0))

        self.outc = (OutConv(c0, cout))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)