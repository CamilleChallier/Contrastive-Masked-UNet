import torch
from torch import nn
import numpy as np  
import copy

from cmae.registry import MODELS

@MODELS.register_module()
class DoubleConv(nn.Module):
    """
    A module consisting of two convolutional layers with batch normalization 
    and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
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
    
@MODELS.register_module()    
class DownBlock(nn.Module):
    """
    A downsampling block consisting of a DoubleConv followed by max pooling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

@MODELS.register_module()
class UNet_encoder(nn.Module):
    """
    U-Net encoder with downsampling layers and a bottleneck.

    Args:
        out_classes (int, optional): Number of output classes. Defaults to 2.
        up_sample_mode (str, optional): Upsampling mode. Defaults to 'conv_transpose'.
        patch_size (int, optional): Size of patches for random masking. Defaults to 16.
        mask_ratio (float, optional): Proportion of image to be masked. Defaults to 0.65.
    """
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', patch_size=16, mask_ratio=0.65):
        super(UNet_encoder, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        
        self.patch_size=patch_size
        self.mask_ratio = mask_ratio
        
    def forward(self, x):
        x = x.unsqueeze(1)      
        x, mask= self.random_masking(x)  
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        return x, torch.from_numpy(mask).cuda(), [skip1_out, skip2_out, skip3_out, skip4_out]
    
    def init_weights(self):
        # Apply custom weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Kaiming Normal Initialization for Conv2d layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            # Xavier Normal Initialization for BatchNorm layers
            nn.init.constant_(m.weight, 1)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Xavier Normal Initialization for Linear layers
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
      
    def create_random_patch_mask(self, batch_size, img_size=256):
        """
        Generates a random binary mask by masking patches of the image.

        Args:
            batch_size (int): Number of images in batch.
            img_size (int, optional): Size of the image. Defaults to 256.

        Returns:
            np.ndarray: Random binary mask of shape (batch_size, img_size, img_size).
        """
        num_patches = (img_size // self.patch_size) ** 2  # Total number of patches per image
        target_mask_area = int(self.mask_ratio * img_size * img_size)  # Target area to mask
        mask = np.zeros((batch_size, img_size, img_size), dtype=np.uint8) 

        for i in range(batch_size):
            current_mask_area = 0
            patch_indices = np.arange(num_patches)
            np.random.shuffle(patch_indices)  # Shuffle patch indices to select random patches
            
            for idx in patch_indices:
                # Calculate patch's top-left corner
                row = (idx // (img_size // self.patch_size)) * self.patch_size
                col = (idx % (img_size // self.patch_size)) * self.patch_size
                
                # Add patch to the mask if it doesn't exceed the target area
                if current_mask_area + self.patch_size * self.patch_size <= target_mask_area:
                    mask[i, row:row + self.patch_size, col:col + self.patch_size] = 1
                    current_mask_area += self.patch_size * self.patch_size

                # Stop if target mask area is reached
                if current_mask_area >= target_mask_area:
                    break
        return mask

    def random_masking(self, x):
        """
        Applies random masking to the input tensor using generated patch masks.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            tuple:
                - torch.Tensor: Masked input tensor.
                - np.ndarray: Binary mask applied to the input tensor.
        """
        batch_size, img_rows, _ = x.shape[0], x.shape[2], x.shape[3] #, img.shape[4]

        mask = self.create_random_patch_mask(batch_size, img_rows)
        x_masked = x * (1 - torch.from_numpy(mask[0]).to("cuda:0"))

        return x_masked, mask

