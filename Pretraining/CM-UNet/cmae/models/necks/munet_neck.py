# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from cmae.registry import MODELS

from ..backbones.UNet_encoder import DoubleConv


@MODELS.register_module()
class UpBlock(nn.Module):
    """
    Upsampling block used in the decoder of a U-Net-based model.

    This block consists of an upsampling operation (either transposed convolution or bilinear interpolation)
    followed by a double convolution to refine the upsampled features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        up_sample_mode (str): Upsampling method, either 'conv_transpose' (for transposed convolution) 
                              or 'bilinear' (for bilinear interpolation).
    """
    def __init__(self, in_channels, out_channels, up_sample_mode):
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
        Forward pass for the upsampling block.

        Args:
            down_input (torch.Tensor): The input tensor from the previous layer in the decoder.
            skip_input (torch.Tensor): The corresponding skip connection from the encoder.

        Returns:
            torch.Tensor: The output tensor after upsampling and convolution.
        """
        x = self.up_sample(down_input)
        # print(x.shape)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

@MODELS.register_module()
class MUNetPretrainDecoder(BaseModule):
    """
    Decoder for the U-Net-based architecture used in pretraining.

    This module consists of multiple `UpBlock` layers that progressively upsample and refine the 
    feature maps, followed by a final 1x1 convolution to produce the output.

    Args:
        out_classes (int, optional): Number of output channels. Defaults to 2.
        up_sample_mode (str, optional): Upsampling method ('conv_transpose' or 'bilinear'). Defaults to 'conv_transpose'.
    """
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose'):
        super(MUNetPretrainDecoder, self).__init__()
        self.up_sample_mode = up_sample_mode
        
        self.up_conv4 = UpBlock(1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)
        #TODO : add softmax and give forground to dice loss, (dice for each object)
        
    def forward(self, x, skip):        
        
        x = self.up_conv4(x, skip[3])
        x = self.up_conv3(x, skip[2])
        x = self.up_conv2(x, skip[1])
        x = self.up_conv1(x, skip[0])
        x = self.conv_last(x)
        return x

    def init_weights(self):
        """
        Initialize the weights of the model using custom initialization.
        """
        super(MUNetPretrainDecoder, self).init_weights()

    def _init_weights(self, m):
        """
        Custom weight initialization function.

        Args:
            m (nn.Module): The layer or module to apply initialization to.
        """
        if isinstance(m, nn.Conv2d):
            # Apply Kaiming Normal initialization to Conv2D layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            # Apply constant initialization to BatchNorm layers (weight=1, bias=0)
            nn.init.constant_(m.weight, 1)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Apply Xavier Normal initialization to Linear layers
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)