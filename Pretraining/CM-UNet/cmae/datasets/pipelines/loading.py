import os.path as osp
import mmcv
from PIL import Image
import numpy as np

from mmcv.transforms.base import BaseTransform
from cmae.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadImageNetFromFile(BaseTransform):
    def __init__(self,to_float32=False,to_rgb=True):
        self.to_float32 = to_float32
        self.to_rgb = to_rgb

    def transform(self,results):
        # if results['prefix'].startswith("ImageNet"):
        # results['filename'] = results['data_root'],results['prefix'],results['filename']
        # filename = results['filename']
        # img = np.load(filename)
        # img = Image.fromarray(img)
        # img = img.resize((256, 256), resample= Image.BICUBIC)
        # img = results
        # if self.to_float32:
        #     img = img.astype(np.float32)
        # else:
        #     img = img.astype(np.uint8)

        results['img']=results

        return results
