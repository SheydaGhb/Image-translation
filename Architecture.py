"""
Structure of UNet by assembling instances of Down and Up classes from UNet_parts script
"""


from UNet_parts import *

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        y = self.outconv(x)
        return y

"""
if you run this script, models structure will be printed
"""
def main():
    model = UNet(3,3)
    print(model)

if __name__ == '__main__':
    main()