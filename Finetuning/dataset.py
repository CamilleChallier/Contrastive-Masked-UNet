# code inspired from https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
import config as config
from sklearn.model_selection import train_test_split
import albumentations as album

class SegmentationDataset(Dataset):
    """
    A custom PyTorch Dataset for image segmentation tasks.

    This dataset loads images and corresponding masks from provided directories, 
    applies optional augmentations, resizes them to a fixed size (256x256), 
    and performs one-hot encoding on the masks.

    Attributes:
        image_paths (list): List of file paths to images.
        mask_paths (list): List of file paths to corresponding masks.
        augmentation (albumentations.Compose, optional): Augmentations to apply.
        class_values (list, optional): List of class values for one-hot encoding.
        last_axis (bool, optional): If True, adds an extra axis to images.
    """
    def __init__(self, images_dir, masks_dir, augmentation=None, class_values=None, last_axis=False):
        self.image_paths = images_dir
        self.mask_paths = masks_dir
        self.class_values = class_values
        self.augmentation = augmentation
        self.last_axis = last_axis
     
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imagePath = self.image_paths[idx]
        image = np.load(imagePath)
        mask = np.load(self.mask_paths[idx])
        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']	
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image = image.resize((256, 256), resample= Image.BICUBIC)
        mask = mask.resize((256, 256), resample=Image.NEAREST)
        mask = one_hot_encode(mask, self.class_values).astype('float')
        
        if self.last_axis==True :
            image = np.asarray(image)[..., np.newaxis]
            image = np.transpose(image, (2,0,1))
        else :
            image = np.asarray(image)
        return (image, np.asarray(mask))

def visualize(**images):
    """
    Plots multiple images in a single row.

    Args:
        **images: Keyword arguments where keys are titles and values are images.

    Returns:
        None: Displays the images using matplotlib.
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):

        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image, cmap = "gray")
    plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values (list): List of class values to encode.

    Returns:
        numpy.ndarray: 3D one-hot encoded mask with shape (num_classes, height, width).
    """
    semantic_map = []
    
    for colour in label_values:
        equality = np.equal(label, colour)
        semantic_map.append(equality)
    semantic_map = np.stack(semantic_map, axis=0)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = 0)
    return x

def prepare_train_test():
    """
    Prepares lists of file paths for training and testing datasets.

    This function retrieves all image and mask file paths from the directories
    specified in the `config` module, sorts them, and returns them as lists.

    Returns:
        tuple: A tuple containing:
            - imagePaths (list): List of file paths to images.
            - maskPaths (list): List of file paths to corresponding masks.
    """

    imagePaths = [os.path.join(config.IMG_DIR, image_id) for image_id in sorted(os.listdir(config.IMG_DIR))]
    maskPaths = [os.path.join(config.MSK_DIR, image_id) for image_id in sorted(os.listdir(config.MSK_DIR))]
    
    return imagePaths, maskPaths

def get_training_augmentation():
    """
    Defines and returns a set of augmentation transformations for training.

    The augmentations include random cropping, noise addition, blurring, 
    brightness adjustments, downscaling, and random flipping/rotation.

    Returns:
        albumentations.Compose: A composed augmentation pipeline for training images.
    """

    train_transform = [
        album.RandomCrop(height=475, width=475, always_apply=True),
        album.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
        album.GaussianBlur(blur_limit=(5, 11), sigma_limit=(0.5, 1.0), p=0.2),
        album.RandomBrightnessContrast(brightness_limit=0.25, p=0.15),
        album.Downscale(scale_min=0.5, scale_max=1.0, p=0.25),

        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
                album.GaussNoise(p=1)
            ],
            p=0.75,
        ),
    ]

    return album.Compose(train_transform)

def get_validation_augmentation():   
    """
    Defines and returns a set of augmentation transformations for validation.

    The validation augmentations include padding images to a minimum size.

    Returns:
        albumentations.Compose: A composed augmentation pipeline for validation images.
    """

    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)