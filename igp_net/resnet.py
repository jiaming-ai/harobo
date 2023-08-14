
import torch
import torch.nn as nn



def convrelu(in_channels, out_channels, kernel, padding=0,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding,stride=stride),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


    
class ResConvBlock(nn.Module):

    def __init__(self, cin, cout=None,stride=2):
        """_summary_
        if cout == cin and stride = 1, then no downsample

        Args:
            cin (_type_): _description_
            cout (_type_, optional): _description_. Defaults to None.
            stride (int, optional): _description_. Defaults to 2.
        """
        super().__init__()
        cout = cout or cin * 2
        norm_layer = nn.BatchNorm2d
        downsample = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=1, stride=2),
            nn.BatchNorm2d(cout),
        )
        # Both self.conv1 and self.downsample layers downsample the input 
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.bn1 = norm_layer(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, stride=stride,padding=1)
        self.bn2 = norm_layer(cout)
        self.downsample = downsample if stride != 1 else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class DeconvBlock(nn.Module):
    def __init__(self, cin, cout=None, kernel_size=4,\
                stride=2, padding = 1):
        super().__init__()
        cout = cout or cin // 2
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.upsample layers upsample the input 
        self.conv1 = nn.ConvTranspose2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = norm_layer(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, stride=1,padding=1)
        self.bn2 = norm_layer(cout)

    def forward(self, x):
        # identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
class ResNetBackbone(nn.Module):
    def __init__(self,cin,cout, img_size, c0=8,layer=5) -> None:
        super().__init__()

        self.conv0 = convrelu(cin, c0, kernel=3, stride=2, padding=1) # out 8 x 384 x 384
        
        down = []
        for i in range(layer):
            down.append(ResConvBlock(c0*2**i))     
        self.down = nn.Sequential(*down)
        
        # self.down1 = ResConvBlock(c0) # out 16 x 192 x 192
        # self.down2 = ResConvBlock(c0*2) # out 32 x 96 x 96
        # self.down3 = ResConvBlock(c0*4) # out 64 x 48 x 48
        # self.down4 = ResConvBlock(c0*8) # out 128 x 24 x 24
        # self.down5 = ResConvBlock(c0*16) 
        
        last_szie = img_size // 2**(layer+1)
        fc_in_dim = c0*2**(layer)* last_szie * last_szie
        self.last_size = last_szie
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, fc_in_dim),

        )
        up = []
        for i in range(layer+1):
            up.append(DeconvBlock(c0*2**(layer-i)))
        
        self.up = nn.Sequential(*up)

        # self.up1 = DeconvBlock(c0*32)
        # self.up2 = DeconvBlock(c0*16)
        # self.up3 = DeconvBlock(c0*8)
        # self.up4 = DeconvBlock(c0*4)
        # self.up5 = DeconvBlock(c0*2)
        # self.up6 = DeconvBlock(c0)
        self.conv_out = nn.Conv2d(c0//2, cout, kernel_size=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.down(x)
        # x = self.down1(x)
        # x = self.down2(x)
        # x = self.down3(x)
        # x = self.down4(x)
        # x = self.down5(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, self.last_size, self.last_size)

        x = self.up(x)        
        # x = self.up1(x)
        # x = self.up2(x)
        # x = self.up3(x)
        # x = self.up4(x)
        # x = self.up5(x)
        # x = self.up6(x)
        x = self.conv_out(x)

        return x