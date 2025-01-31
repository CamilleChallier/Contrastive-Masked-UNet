from scipy.spatial.distance import directed_hausdorff
import torch.nn as nn
import torch
import re
import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np

class BaseObject(nn.Module):
    """Base class for all objects in the module.

    Args:
        name (str, optional): Custom name for the object. Defaults to None.
    """
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name

class Metric(BaseObject):
    pass


class Loss(BaseObject):
    """Base class for all loss functions, supporting addition and multiplication."""
    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other):
        return self.__mul__(other)


class SumOfLosses(Loss):
    """Represents the sum of two loss functions."""
    def __init__(self, l1, l2):
        name = "{} + {}".format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):
    """Represents a loss function scaled by a multiplier."""
    def __init__(self, loss, multiplier):
        # resolve name
        if len(loss.__name__.split("+")) > 1:
            name = "{} * ({})".format(multiplier, loss.__name__)
        else:
            name = "{} * {}".format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs):
        return self.multiplier * self.loss.forward(*inputs)
    
    def forward(self, *inputs):
        # Calls the forward method of the individual loss and multiplies the result by the multiplier
        return self.multiplier * self.loss.forward(*inputs)

class Activation(nn.Module):
    """Wrapper for activation functions."""
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"None; got {name}"
            )
    def forward(self, x):
        return self.activation(x)
    
def _take_channels(*xs, ignore_channels=None):
    """Helper function to select specific channels from tensors."""
    if ignore_channels is None:
        return xs
    else:
        channels = [
            channel
            for channel in range(xs[0].shape[1])
            if channel not in ignore_channels
        ]
        xs = [
            torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device))
            for x in xs
        ]
        return xs


def _threshold(x, threshold=None):
    """Helper function to apply a threshold for binarization."""
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def f_score(pr, gt, beta=1, eps=1e-5, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)
    return score


class DiceLoss(Loss):
    """Dice loss function for segmentation tasks."""
    def __init__(
        self, eps=1e-5, beta=1.0, activation=None, ignore_channels=None, threshold=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.threshold = threshold

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)

        return 1 - f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
       
def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union
 
class IoU(Metric):
    __name__ = "iou_loss"

    def __init__(
        self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1- iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
     
from scipy.spatial import cKDTree   
from skimage.measure import find_contours
def hausdorff_distance_mask(image0, image1, method = 'modified'):
    """Calculate the Hausdorff distance between the contours of two segmentation masks.
    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a pixel from a segmented object. Both arrays must have the same shape.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of the segmentation mask contours in
        ``image0`` and ``image1``, using the Euclidean distance.
    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on the 
    contour of ``image0`` and its nearest point on the contour of ``image1``, and 
    vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155
    Examples
    --------
    >>> ground_truth = np.zeros((100, 100), dtype=bool)
    >>> predicted = ground_truth.copy()
    >>> ground_truth[30:71, 30:71] = disk(20)
    >>> predicted[25:65, 40:70] = True
    >>> hausdorff_distance_mask(ground_truth, predicted)
    11.40175425099138
    """
    
    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')
    
    a_contours = find_contours(image0 > 0)
    if a_contours:
        a_points = np.concatenate(a_contours)
    else:
        a_points = np.array([])
        
    b_contours = find_contours(image1 > 0)
    if b_contours:
        b_points = np.concatenate(b_contours)
    else:
        b_points = np.array([])

    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf
    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )

    if method == 'standard':  # standard Hausdorff distance
        return max(max(fwd), max(bwd))
    elif method == 'modified':  # modified Hausdorff distance
        return max(np.mean(fwd), np.mean(bwd))


class hausdorff(Metric):
    """Computes Hausdorff distance for segmentation evaluation."""
    __name__ = "hausdorff"

    def __init__(
        self, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        """Calculate Intersection over Union between ground truth and prediction
        Args:
            y_pr (torch.Tensor): predicted tensor
            y_gt (torch.Tensor):  ground truth tensor
        Returns:
            float: Haudorff distance
        """

        y_pr = self.activation(y_pr)
        y_pr = _threshold(y_pr, threshold=self.threshold)
        y_pr, y_gt = _take_channels(y_pr, y_gt, ignore_channels=self.ignore_channels)
        
        if torch.is_tensor(y_pr) : 
            y_pr_np = y_pr.clone().cpu().detach().numpy()
            y_gt_np = y_gt.clone().cpu().detach().numpy()
            
        y_pr_np = y_pr_np.squeeze(1)
        y_gt_np = y_gt_np.squeeze(1)
        
        hausdorff_distances = [
            hausdorff_distance_mask(y_pr_np[i], y_gt_np[i])  # Get the distance (first element) for each pair in the batch
            for i in range(y_pr_np.shape[0])
        ]
        return torch.mean(torch.tensor(hausdorff_distances))
    
class radius_arteries(Metric):
    __name__ = "radius_arteries"

    def __init__(self, **kwargs ):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        if torch.is_tensor(y_pr) : 
            y_pr_np = y_pr.clone().cpu().detach().numpy()
            y_gt_np = y_gt.clone().cpu().detach().numpy()
            
        y_pr_np = np.argmax(y_pr_np, axis=1)  # Shape: (batch_size, height, width)
        y_gt_np = np.argmax(y_gt_np, axis=1)  # Shape: (batch_size, height, width)
        radius = [np.abs(compute_radius_arteries(pr.astype(bool))[1]- compute_radius_arteries(gt.astype(bool))[1]) for pr,gt in zip(y_pr_np, y_gt_np)]
        return torch.tensor(np.mean(radius))

from skimage.morphology import skeletonize
from skimage import measure

import numpy as np
from scipy.spatial import KDTree

def calculate_radius(contour_points, skeleton_points):
    """
    Calculate the radius at each point of the skeleton.
    
    Parameters:
    - contour_points: A list or array of (x, y) coordinates representing the contour.
    - skeleton_points: A list or array of (x, y) coordinates representing the skeleton.
    
    Returns:
    - radii: A list of radii, where each radius corresponds to a skeleton point.
    """
    
    # Create a KDTree for the contour points to efficiently find nearest neighbors
    contour_tree = KDTree(contour_points)
    
    # For each skeleton point, find the nearest contour point and calculate the distance (radius)
    radii = []
    for skeleton_point in skeleton_points:
        # Find the nearest contour point and get the distance
        distance, _ = contour_tree.query(skeleton_point)
        radii.append(distance)
    
    return radii

def compute_radius_arteries(mask):
    """Computes the radius of arteries from a binary segmentation mask."""
    
    mask[0,:]=False
    mask[:,0]=False
    mask[:,-1]=False
    mask[-1,:]=False
    skelet = skeletonize(image=mask)
    contour = measure.find_contours(image =mask, positive_orientation="high")
    if contour == [] :
        return 0,0,0
    contours =  np.vstack(contour)

    # Calculate the radius for each skeleton point
    radii = calculate_radius(contours, np.argwhere(skelet))

    return 2*np.min(radii), 2*np.mean(radii), 2*np.max(radii)

import torch
import torch.nn as nn
import torch.nn.functional as F

class soft_cldice(Loss):
    """Soft clDice loss function for segmentation evaluation."""
    __name__ = "soft_clDice"
    def __init__(self, iter_=3, smooth = 1., exclude_background=False, threshold=0.5, activation=None, ignore_channels=None):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pred, y_true):
        y_pred = self.activation(y_pred)
        y_pred = _threshold(y_pred, threshold=self.threshold)
        y_pred, y_true = _take_channels(y_pred, y_true, ignore_channels=self.ignore_channels)
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)

class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):

        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):

        return self.soft_skel(img)
    

class L1Loss(nn.L1Loss, Loss):
    pass


class MSELoss(nn.MSELoss, Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, Loss):
    pass

import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())    

class NLLLoss(Loss) :

    def __init__(
        self, activation=None, ignore_channels=None, threshold=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.threshold = threshold

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        # y_pr = _threshold(y_pr, threshold=self.threshold)
        y_pr, y_gt = _take_channels(y_pr, y_gt, ignore_channels=self.ignore_channels)
        
        y_gt = torch.argmax(y_gt, axis=1) 
        loss_fn = nn.NLLLoss()
        return loss_fn(
            y_pr,
            torch.tensor(y_gt, dtype=torch.long)
        )


class BCELoss(nn.BCELoss, Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, Loss):
    pass
