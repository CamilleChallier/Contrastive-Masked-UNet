import torch
import sys

import torch.nn as nn

from typing import List
from timm.models.registry import register_model
from encoder import SparseConv2d, SparseMaxPooling, SparseBatchNorm2d
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))

from UNET.model import DoubleConv, DownBlock

class DoubleConv_sparse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_sparse, self).__init__()
        self.double_conv = nn.Sequential(
            SparseConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            SparseBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SparseConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            SparseBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class DownBlock_sparse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock_sparse, self).__init__()
        self.double_conv = DoubleConv_sparse(in_channels, out_channels)
        self.down_sample = SparseMaxPooling(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UNET_MAE(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    def __init__(self, in_chans=1, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., global_pool='avg',
                 sparse=True):
        super().__init__()
        self.up_sample_mode='conv_transpose'
        self.down_conv1 = DownBlock(1, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
    
    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 16
    
    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return [64, 128, 256, 512, 1024]
    
    def forward(self, inp_bchw: torch.Tensor, hierarchical=False):
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).
        
        :param inp_bchw: input image tensor, shape: (batch_size, channels, height, width).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
              E.g., for a ResNet-50, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14), (B, 2048, 7, 7)]
        """
        feature_maps = []

        # Downsampling path
        x, skip1_out = self.down_conv1(inp_bchw)
        if hierarchical: feature_maps.append(x)
        
        x, skip2_out = self.down_conv2(x)
        if hierarchical: feature_maps.append(x)
        
        x, skip3_out = self.down_conv3(x)
        if hierarchical: feature_maps.append(x)
        
        x, skip4_out = self.down_conv4(x)
        if hierarchical: feature_maps.append(x)
        
        # Bottleneck layer
        x = self.double_conv(x)
        if hierarchical: feature_maps.append(x)
        
        if hierarchical:
            return feature_maps

class UNET_MAE_SPARSE(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    def __init__(self, in_chans=1, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., global_pool='avg',
                 sparse=True):
        super().__init__()
        self.up_sample_mode='conv_transpose'
        self.down_conv1 = DownBlock_sparse(1, 64)
        self.down_conv2 = DownBlock_sparse(64, 128)
        self.down_conv3 = DownBlock_sparse(128, 256)
        self.down_conv4 = DownBlock_sparse(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
    
    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 16
    
    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return [64, 128, 256, 512, 1024]
    
    def forward(self, inp_bchw: torch.Tensor, hierarchical=False):
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).
        
        :param inp_bchw: input image tensor, shape: (batch_size, channels, height, width).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
              E.g., for a ResNet-50, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14), (B, 2048, 7, 7)]
        """
        feature_maps = []

        # Downsampling path
        x, skip1_out = self.down_conv1(inp_bchw)
        if hierarchical: feature_maps.append(skip1_out)
        
        x, skip2_out = self.down_conv2(x)
        if hierarchical: feature_maps.append(skip2_out)
        
        x, skip3_out = self.down_conv3(x)
        if hierarchical: feature_maps.append(skip3_out)
        
        x, skip4_out = self.down_conv4(x)
        if hierarchical: feature_maps.append(skip4_out)
        
        # Bottleneck layer
        x = self.double_conv(x)
        if hierarchical: feature_maps.append(x)
        
        if hierarchical:
            return feature_maps

@register_model
def unet(pretrained=False, **kwargs):
    return UNET_MAE(**kwargs)

@register_model
def unet_sparse(pretrained=False, **kwargs):
    return UNET_MAE_SPARSE(**kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('unet_encoder')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = 4, 1, 256, 256
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
