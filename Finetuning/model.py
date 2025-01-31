import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """
        A module that applies two consecutive convolutional layers, 
        each followed by batch normalization and a ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A downsampling block that consists of a DoubleConv layer followed by max pooling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        """
        An upsampling block that upsamples the input and concatenates it with a skip connection.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            up_sample_mode (str): Upsampling method, either 'conv_transpose' or 'bilinear'.
        """
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        """
        Forward pass of the UpBlock.

        Args:
            down_input (torch.Tensor): The downsampled input tensor.
            skip_input (torch.Tensor): The corresponding skip connection tensor.

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, H, W).
        """
        x = self.up_sample(down_input)
        # print(x.shape)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

    
class UNet(nn.Module):
    """
    A U-Net model for image segmentation, consisting of downsampling, a bottleneck, and upsampling.

    Args:
        out_classes (int, optional): Number of output classes. Defaults to 2.
        up_sample_mode (str, optional): Upsampling mode, either 'conv_transpose' or 'bilinear'. Defaults to 'conv_transpose'.
    """
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_classes, H, W).
        """
        x = x.unsqueeze(1)        
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x