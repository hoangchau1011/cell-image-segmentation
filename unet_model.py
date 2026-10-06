import torch
import torch.nn as nn
import torchvision.transforms.functional as TF



# ==========================================================
# Double Conv Block
# Conv3x3 + BN + ReLU
# Conv3x3 + BN + ReLU
# ==========================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),


            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )


    def forward(self,x):

        return self.block(x)




# ==========================================================
# Down sampling
# ==========================================================

class Down(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.block = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(
                in_channels,
                out_channels
            )

        )


    def forward(self,x):

        return self.block(x)




# ==========================================================
# Up sampling
# ==========================================================

class Up(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.up = nn.ConvTranspose2d(

            in_channels,

            in_channels // 2,

            kernel_size=2,

            stride=2

        )


        self.conv = DoubleConv(

            in_channels,

            out_channels

        )



    def forward(
        self,
        x1,
        x2
    ):


        x1 = self.up(x1)



        diffY = x2.size(2)-x1.size(2)

        diffX = x2.size(3)-x1.size(3)



        x1 = TF.pad(

            x1,

            [

                diffX//2,
                diffX-diffX//2,

                diffY//2,
                diffY-diffY//2

            ]

        )



        x = torch.cat(

            [
                x2,
                x1
            ],

            dim=1

        )


        return self.conv(x)





# ==========================================================
# U-Net for DSB2018
#
# Input:
#   RGB image
#
# Output:
#   3 channels
#
# Channel:
#   0 Background
#   1 Nucleus
#   2 Boundary
#
# ==========================================================


class UNet(nn.Module):


    def __init__(
        self,
        in_channels=3,
        num_classes=3
    ):

        super().__init__()



        # Encoder

        self.inc = DoubleConv(
            in_channels,
            32
        )


        self.down1 = Down(
            32,
            64
        )


        self.down2 = Down(
            64,
            128
        )


        self.down3 = Down(
            128,
            256
        )


        self.down4 = Down(
            256,
            512
        )



        # Decoder

        self.up1 = Up(
            512,
            256
        )


        self.up2 = Up(
            256,
            128
        )


        self.up3 = Up(
            128,
            64
        )


        self.up4 = Up(
            64,
            32
        )



        # Output

        self.out = nn.Conv2d(

            32,

            num_classes,

            kernel_size=1

        )




    def forward(self,x):


        # Encoder

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)



        # Decoder

        x = self.up1(
            x5,
            x4
        )


        x = self.up2(
            x,
            x3
        )


        x = self.up3(
            x,
            x2
        )


        x = self.up4(
            x,
            x1
        )



        return self.out(x)





# ==========================================================
# Test
# ==========================================================


if __name__ == "__main__":


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model = UNet(
        num_classes=3
    ).to(device)



    x = torch.randn(

        2,

        3,

        256,

        256

    ).to(device)



    y = model(x)



    print(
        "Input:",
        x.shape
    )


    print(
        "Output:",
        y.shape
    )