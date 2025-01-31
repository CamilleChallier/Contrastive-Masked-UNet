import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from moco_data_set import MoCoDataset
import torchvision.transforms.functional as F
# import albumentations as album

import torch.nn as nn

class DoubleConv(nn.Module):
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
    
    
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

    
class UNet_encoder(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose'):
        super(UNet_encoder, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        
    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = torch.mean(x, dim=[2, 3])
        return x

class MoCoDataModule(LightningDataModule):
    """Example of LightningDataModule. A DataModule implements 5 key methods:

        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle: bool = True,
        global_crop_size: int = 224,
        global_crop_scale: Tuple[float, float] = (0.2, 1.0),
        local_crop_size: int = 64,
        local_crop_scale: Tuple[float, float] = (0.5, 1.0),
        drop_last: bool = True,
        mean: list = None,
        std: list = None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.global_crop_size = global_crop_size
        self.global_crop_scale = global_crop_scale
        self.local_crop_size = local_crop_size
        self.local_crop_scale = local_crop_scale
        self.drop_last = drop_last
        self.mean = mean
        self.std = std

        # normalize = transforms.Normalize(mean=self.mean, std=self.std)
        
        # self.tau_g = [get_training_augmentation()]*2
        # self.tau_l = [get_validation_augmentation()]*2

        # MoCov2 image transforms
        self.tau_g = [
            transforms.Compose(
                [
                    transforms.RandomApply([transforms.RandomRotation(180)], p=0.5),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                    # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply([GaussNoise()], p=0.5)
                    # normalize,
                ]
            )
        ] * 2  # make 2 global crops from one image

        self.tau_l = [
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.local_crop_size, scale=self.local_crop_scale),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # normalize,
                ]
            )
        ] * 2
            
        import os
        from sklearn.model_selection import train_test_split

        if self.data_dir == None : 
            data = "../dataset/imgs"
            imagePaths = [os.path.join(data, image_id) for image_id in sorted(os.listdir(data))]
            maskPaths = [os.path.join(data, image_id) for image_id in sorted(os.listdir(data))]
            X_train, X_test, y_train, _ = train_test_split(imagePaths, maskPaths, test_size=0.2, random_state=42)
            X_pretrain, _, _, _ = train_test_split(X_train, y_train, test_size=0.625, random_state=42)
            # X_pretrain = X_train
            print(len(X_pretrain))
            self.data_dir = X_pretrain
            # if args.arcane == True :
            data_path = "../dataset_arcane/train/imgs"
            imagePathsArcane_train = [os.path.join(data_path, image_id) for image_id in sorted(os.listdir(data_path))]
            data_path = "../dataset_arcane/test/imgs"
            imagePathsArcane_test = [os.path.join(data_path, image_id) for image_id in sorted(os.listdir(data_path))]
            
            self.data_dir.extend(imagePathsArcane_train)
            X_test.extend(imagePathsArcane_test)
            import random
            random.shuffle(X_pretrain)
            random.shuffle(X_test)
            print(len(self.data_dir))

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        """Load data.

        Set variables: self.data_train, self.data_val, self.data_test.
        """
        if stage == "fit":
            self.train_set = MoCoDataset(self.data_dir, self.tau_g)
        else:
            print("Not implemented")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )


# class GaussianBlur:
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

#     def __init__(self, sigma=[0.1, 2.0]):
#         self.sigma = sigma

#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         print("gaussian av",torch.unique(x))
#         x = F.to_pil_image(x)
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         x = F.pil_to_tensor(x)
#         print("gaussian",torch.unique(x))
#         return x

class GaussNoise:

    def __init__(self):
        pass

    def __call__(self, image):
        dtype = image.dtype
        sigma = torch.max(image)/10
        
        out = image + sigma * torch.randn_like(image)
        
        if out.dtype != dtype:
            out = out.to(dtype)
            
        return out
    
if __name__ == "__main__":

    # testing the data module and data set
    dm = MoCoDataModule("/mnt/hdd/datasets/Imagenet100/train")
    dm.setup(stage="fit")
    dl = dm.train_dataloader()
    imgs = next(iter(dl))
    breakpoint()
