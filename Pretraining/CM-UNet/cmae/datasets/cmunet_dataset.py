import os
import json

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mmengine.registry import build_from_cfg

from cmae.registry import DATASETS,TRANSFORMS

from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

@DATASETS.register_module()
class CMUNetDataset(Dataset):
    """
    Custom dataset for CMUNet, handling image loading, transformations, and augmentation.

    Args:
        data_root (str): Path to the root directory containing image data.
        data_ann (str): Path to annotation file (not used in current implementation).
        pipeline (list): List of transformation configurations to be applied.
        pixel (int, optional): Pixel shift magnitude for augmentation. Defaults to 31.
        test (bool, optional): If True, uses the test split of the dataset. Defaults to False.
    """
    def __init__(self, data_root, data_ann, pipeline, pixel=31, test=False):
        self.data_root = data_root
        imagePaths = [os.path.join(data_root, image_id) for image_id in sorted(os.listdir(data_root))]
        maskPaths = imagePaths.copy()
        X_train, X_test, y_train, _ = train_test_split(imagePaths, maskPaths, test_size=0.2, random_state=42)
        images_dir, _, _, _ = train_test_split(X_train, y_train, test_size=0.0125, random_state=42)
        
        self.image_paths = images_dir
        print(len( self.image_paths))
        self.test = X_test
        print(len(X_test))

        pipeline_base = [build_from_cfg(p, TRANSFORMS) for p in pipeline[:2]]
        self.pipeline_base = Compose(pipeline_base)
        pipeline_final = [build_from_cfg(p, TRANSFORMS) for p in pipeline[2:]]
        self.shift = build_from_cfg(dict(
            type='ShiftPixel',
            pixel=0), TRANSFORMS)
        self.pipeline_final = Compose(pipeline_final)

        pipeline_aug = [
            dict(
                type='ShiftPixel',
                pixel=pixel,
            ),
            dict(type='GaussNoise', magnitude_range=(0.1, 2.0), magnitude_std='inf', prob=0.5)
        ]
        pipeline_aug_l = [build_from_cfg(p, TRANSFORMS) for p in pipeline_aug]
        self.pipeline_aug = Compose(pipeline_aug_l)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image from the dataset, applies transformations, and returns augmented versions.

        Args:
            idx (int): Index of the image.

        Returns:
            dict: A dictionary containing:
                - 'img' (np.ndarray): Transformed patch image.
                - 'img_t' (np.ndarray): Augmented transformed image.
        """
        imagePath = self.image_paths[idx]
        image = np.load(imagePath)

        image = Image.fromarray(image)
        image = image.resize((256, 256), resample= Image.BICUBIC)

        results = {'img':  np.asarray(image)}
        src_img = self.pipeline_base(results)
        patch_results = {'img': src_img['img']}
        img_t_results = {'img': src_img['img'].copy()}

        patch = self.pipeline_final(self.shift(patch_results))
        img_t = self.pipeline_final(self.pipeline_aug(img_t_results))
        
        out = {'img':patch['img'],'img_t':img_t['img']}

        return out

