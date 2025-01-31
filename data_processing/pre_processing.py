from abc import ABC, abstractmethod
from typing import Union
from skimage.filters import unsharp_mask
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

import numpy as np
import copy
import cv2

class PreProcessor(ABC):
    """Abstract class for pre-processing data"""

    def __init__(self) -> None:
        pass
    @abstractmethod
    def transform(
        self, data: dict[dict[np.ndarray]],
    ) -> dict[dict[np.ndarray]]:
        """
        Transform the incoming data and return the transformed data 

        Parameters
        ----------
        data : dict
            The data to transform
        """
        pass

    def fit_transform(
        self, data: dict[dict[np.ndarray]],
    ) -> dict[dict[np.ndarray]]:
        """
        Fit the processor on the incoming data and return the transformed data

        Parameters
        ----------
        data : dict
            The data to transform

        Returns
        -------
        dict[dict[np.ndarray]]
            The transformed data
        """
        return self.transform(data)
    
class Unlabelled_Remover(PreProcessor):
    """
    Preprocessor for removing unlabelled data from a dataset.

    This class filters out the images that have no valid labels, i.e., images where the "labelled" field is empty or
    does not exist.
    """
    def __init__(self) -> None:
        super().__init__()

    def transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        clean_data = copy.deepcopy(data)
        clean_data = {k: v for k, v in clean_data.items() if "labelled" in v and len(v["labelled"]) > 0}
        return clean_data

    def fit_transform(
        self, data: dict[dict[np.ndarray]],
    ) -> dict[dict[np.ndarray]]:
        
        return self.transform(data)
    
from sklearn import preprocessing

class skly_normalizer(PreProcessor):
    """
    Preprocessor for normalizing image data using sklearn's normalization.

    This class applies normalization to the raw data using sklearn's `normalize` function.
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        normalized_data = copy.deepcopy(data)
        normalized_data = {k: {**v, "raw": preprocessing.normalize(v["raw"]).astype('float32')} for k, v in normalized_data.items()}
        return normalized_data

    def fit_transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        return self.transform(data)
    
class Intensity_normalizer(PreProcessor):
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.
    """
    def __init__(self) -> None:
        super().__init__()

    def transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        if (
            not hasattr(self, "means")
            or not hasattr(self, "stds")
        ):
            raise ValueError(
                "The means and stds are not set, please call `fit_transform` first."
            )
        normalized_data = copy.deepcopy(data)
        normalized_data = {k: {**v, "raw": ((v["raw"] - self.means[k]) / self.stds[k]).astype('float32')} for k, v in normalized_data.items()}
        return normalized_data

    def fit_transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        
        self.means = {k:np.mean(v["raw"]) for (k,v) in data.items()}
        self.stds = {k:np.std(v["raw"]) for (k,v) in data.items()}
        return self.transform(data)
    
class MinMax_normalizer(PreProcessor):
    """
    Preprocessor for normalizing image intensity to a range [0, 1].

    This class scales the pixel values to the range [0, 1] by applying Min-Max normalization.
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        if (
            not hasattr(self, "min")
            or not hasattr(self, "max")
        ):
            raise ValueError(
                "The means and stds are not set, please call `fit_transform` first."
            )
        normalized_data = copy.deepcopy(data)
        normalized_data = {k: {**v, "raw": (v["raw"] - self.min[k]) / (self.max[k]-self.min[k]).astype('float32')} for k, v in normalized_data.items()}
        return normalized_data

    def fit_transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        
        self.min = {k:np.min(v["raw"]) for (k,v) in data.items()}
        self.max = {k:np.max(v["raw"]) for (k,v) in data.items()}
        return self.transform(data)
    
class Unsharper(PreProcessor):
    """Unsharp masking is a linear image processing technique which sharpens the image. 
    The sharp details are identified as a difference between the original image and its blurred version. 
    These details are then scaled, and added back to the original image.
    """
    def __init__(self, radius: int, amount: int, preserve_range : bool) -> None:
        super().__init__()
        self.radius = radius
        self.amount = amount
        self.preserve_range = preserve_range

    def transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        normalized_data = copy.deepcopy(data)
        normalized_data = {k: {**v, "raw": unsharp_mask(v["raw"], radius=self.radius, amount=self.amount, preserve_range=self.preserve_range)} for k, v in normalized_data.items()}
        return normalized_data

    def fit_transform(
        self, data: dict[dict[np.ndarray]],
    ) -> dict[dict[np.ndarray]]:
        
        return self.transform(data)
    
class Mask_Integrater(PreProcessor):
    """
    Preprocessor for combining individual masks into a single mask.

    This class combines multiple binary masks into a single mask by summing them and converting non-zero values to 1.
    """
    def __init__(self) -> None:
        super().__init__()

    def transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        normalized_data = copy.deepcopy(data)
        normalized_data = {
                            k: {**v, "labelled": np.array(v["labelled"]).sum(axis=0).astype("uint8")} 
                            if len(v["labelled"]) != 1 
                            else {**v, "labelled": v["labelled"][0]}
                            for k, v in normalized_data.items()
                            }
        normalized_data = {
            k: {**v, "labelled": (v["labelled"] != 0).astype(int)}  # Converts non-zero values to 1
            for k, v in normalized_data.items()
        }
        return normalized_data

    def fit_transform(
        self, data: dict[dict[np.ndarray]],
    ) -> dict[dict[np.ndarray]]:
        
        return self.transform(data)
    
class Mask_Contour_Fillier(PreProcessor):
    """ 
    A preprocessor that applies contour filling to the input mask.
    
    The `contour_filling` method fills the contours of the mask with the specified thickness.
    It finds the contours of the inverted mask and draws them with a given thickness. The filled mask is then returned.
    """
   
    def __init__(self,thick=1) -> None:
        super().__init__()
        self.thick = thick
        
    def contour_filling (self, mask, thick=1):
        des = cv2.bitwise_not(mask)
        contour, hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(des, contour, -1, (0,255,0), thick)
        gray = cv2.bitwise_not(des)
        return gray
        

    def transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        data_copy = copy.deepcopy(data)
        data_copy = {k: {**v, "labelled": self.contour_filling(v["labelled"], self.thick)} for k, v in data_copy.items()}
        # data_copy = {k: {**v, "labelled": [self.contour_filling(i, self.thick) for i in v["labelled"]]} for k, v in data_copy.items()}
        
        return data_copy

    def fit_transform(
        self, data: dict[dict[np.ndarray]],
    ) -> dict[dict[np.ndarray]]:
        
        return self.transform(data)

class ReplaceWithBorderPixel(ImageOnlyTransform):
    """
    A transformation that replaces border pixels in an image with pixels from the border of the image.
    
    The `apply` method detects the border of the image and replaces the border pixels with the nearest border pixels
    if certain conditions (such as a threshold) are met. This can be useful for image restoration tasks.

    """
    def __init__(self, always_apply=False, p=1.0, border_ratio=0.25, thresh=20):
        super(ReplaceWithBorderPixel, self).__init__(always_apply, p)
        self.border_ratio=border_ratio
        self.thresh = thresh

    def apply(self, img, **params):
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Create a border mask (focusing only on the borders of the image)
        border_width = int(min(h, w) * self.border_ratio)  # A percentage of the image size, tweak if needed
        
        clean_image=img.copy()
        corners = np.full(img.shape, 255)
        corners[:border_width, :border_width] = img[:border_width, :border_width]
        corners[:border_width, w-border_width:] = img[:border_width, w-border_width:]
        corners[h-border_width:, :border_width] = img[h-border_width:, :border_width]
        corners[h-border_width:, w-border_width:] = img[h-border_width:, w-border_width:]
        
        mask = corners <= self.thresh
        
        if np.sum(mask) < 100 :
            return clean_image
        else :
        
            expanded_mask = np.zeros_like(mask, dtype=bool)  # Create an empty mask for expansion
            for i in range(h):
                for j in range(w):
                    if mask[i, j]:  # If the current pixel is True in the original mask
                        # Set the pixel and its 2-pixel border to True in the expanded mask
                        expanded_mask[max(i - 2, 0):min(i + 3, h), max(j - 2, 0):min(j + 3, w)] = True
                        
            dst = cv2.inpaint(clean_image, expanded_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        
            return dst
        
class CenterCrop(ImageOnlyTransform):
    """
    A transformation that crops the center of an image to a specified height and width.
    
    The `apply` method computes the center coordinates of the image and then crops the image accordingly.
    """
    def __init__(self, height, width, always_apply=False, p=1.0):
        super(CenterCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        """
        Crop the center of the image to the specified height and width.
        """
        img_height, img_width, _ = img.shape
        
        print(np.mean())
        
        # Calculate center coordinates
        center_y, center_x = img_height // 2, img_width // 2
        
        # Calculate the cropping box
        y1 = max(center_y - self.height // 2, 0)
        y2 = min(center_y + self.height // 2, img_height)
        x1 = max(center_x - self.width // 2, 0)
        x2 = min(center_x + self.width // 2, img_width)
        
        # Crop the image
        cropped_img = img[y1:y2, x1:x2, :]
        
        return cropped_img
    
class Cropper(PreProcessor):
    """
    A preprocessor that applies a series of transformations to crop and process images.
    
    The `fit_transform` method composes a set of transformations that apply center cropping, border pixel replacement, 
    and padding. The `transform` method applies these transformations to the input data.
    """
    def __init__(self,size=475, border_ratio=0.25, thresh=20) -> None:
        super().__init__()
        self.size = size  
        self.border_ratio = border_ratio     
        self.thresh = thresh

    def transform(
        self, data: dict[dict[np.ndarray]]
    ) -> dict[dict[np.ndarray]]:
        data_copy = copy.deepcopy(data)
        data_copy = {
                        k: {
                            **v,
                            "raw": augmented['image'], 
                            "labelled": augmented['mask']
                        } 
                        for k, v in data_copy.items() 
                        for augmented in [self.aug(image=v["raw"], mask=v["labelled"])]
}        
        return data_copy

    def fit_transform(
        self, data: dict[dict[np.ndarray]],
    ) -> dict[dict[np.ndarray]]:
        test_transform = [
                            A.CenterCrop(p=1, height=475, width=475),
                            ReplaceWithBorderPixel(border_ratio=self.border_ratio, thresh=self.thresh),
                            A.PadIfNeeded(min_height=475, min_width=475, always_apply=True)#, border_mode=cv2.BORDER_REPLICATE)
                        ]
        self.aug=A.Compose(test_transform)
        
        return self.transform(data)
    
class Pipeline(PreProcessor):
    """
    Implementation of a pipeline of PreProcessors
    """

    def __init__(self, steps: list) -> None:
        self.functions = []
        for step, arg in steps:
            # initialize
            if arg != None:
                self.functions.append(step(**arg))
            else:
                self.functions.append(step())

    def transform(
        self, data: dict[dict[np.ndarray]],
    ) -> dict[dict[np.ndarray]]:
        """
        Transform the data with all steps of the pipeline

        Parameters
        ----------
        data : dict[dict[np.ndarray]]
            images

        Returns
        -------
        dict[dict[np.ndarray]]
            The transformed data
        """
        for function in self.functions:
            print(f"Processing step: {function.__class__.__name__}")
            data = function.transform(data)
        return data

    def fit_transform(
        self, data: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Fit and transform the data with all steps of the pipeline

        Returns
        -------
        dict[dict[np.ndarray]]
            The transformed data

        Raises
        ------
        """
        for function in self.functions:
            print(f"Processing step: {function.__class__.__name__}")
            data = function.fit_transform(data)
        return data
