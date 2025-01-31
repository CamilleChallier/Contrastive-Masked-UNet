from logging import getLogger
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
import torch


logger = getLogger()


class MoCoDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        tau_g,
    ):
        self.tau_g = tau_g  # set of global transforms
        self.data_path= data_path

    def __len__(self):
        return len(self.data_path)

    def __str__(self):
        return f"LoGoDataset with {self.__len__()} images"

    def __getitem__(self, idx):
        imagePath = self.data_path[idx]
        image = np.load(imagePath)	
        image = Image.fromarray(image)
        image = image.resize((256, 256), resample= Image.BICUBIC)
        import copy
        img = copy.deepcopy(np.asarray(image))
        img.setflags(write=True)
        image = torch.from_numpy(img[np.newaxis,:])

        global_crops = list(map(lambda transform: transform(image), self.tau_g))
        
        return (global_crops[0], global_crops[1]), 0
