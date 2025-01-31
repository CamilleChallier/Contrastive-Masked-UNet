from torchvision.transforms import transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC

import numpy as np
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.image_paths = images_dir
        self.transform = transform
     
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imagePath = self.image_paths[idx]
        image = np.load(imagePath)
        image = Image.fromarray(image)
        image = image.resize((256, 256), resample= Image.BICUBIC)
        if self.transform is not None:
            image = self.transform(image)
            
        return (image)


def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:

    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    
    dataset_train = SegmentationDataset(images_dir=dataset_path, transform=trans_train)
    print_transform(trans_train, '[pre-train]')
    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
