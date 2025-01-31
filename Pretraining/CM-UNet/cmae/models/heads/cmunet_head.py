import torch
import torch.nn as nn
from mmengine.model import BaseModule
import torch.nn.functional as F
from mmengine.dist import all_gather

from cmae.registry import MODELS

@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Performs all_gather operation on the provided tensors.

    Args:
        tensor (torch.Tensor): Tensor to be broadcast from current process.

    Returns:
        torch.Tensor: The concatnated tensor.
    """
    tensors_gather = all_gather(tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output


@MODELS.register_module()
class CMUNetPretrainHead(BaseModule):
    """
    Pre-training head for Masked Autoencoders (MAE). This module is used to calculate the reconstruction 
    and contrastive losses during pre-training.

    Args:
        predictor (dict): The configuration for the predictor model used in contrastive learning.
        temperature (float, optional): Temperature value used in contrastive learning. Defaults to 0.07.
        ct_weight (float, optional): Weight for contrastive loss. Defaults to 1.0.
        rc_weight (float, optional): Weight for reconstruction loss. Defaults to 1.0.
    """
    def __init__(self, predictor, temperature=0.07, ct_weight=1.0, rc_weight=1.0 ):
        super(CMUNetPretrainHead, self).__init__()

        self.predictor = MODELS.build(predictor)
        self.t = temperature
        self.ct_weight= ct_weight
        self.rc_weight = rc_weight

        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x,pred_pixel,mask_s,proj_s,proj_t):
        """
        Forward pass for computing the losses during pre-training.

        Args:
            x (torch.Tensor): The input image tensor.
            pred_pixel (torch.Tensor): Predicted pixel values after the encoder.
            mask_s (torch.Tensor): Binary mask for masked pixels.
            proj_s (torch.Tensor): Projected features from source.
            proj_t (torch.Tensor): Projected features from target.

        Returns:
            dict: A dictionary containing the contrastive loss ('loss_ct') 
                  and reconstruction loss ('loss_rc').
        """
        target = x
        
        with torch.no_grad():
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        
        loss_rec = (pred_pixel - target) ** 2
        loss_rc = (loss_rec * mask_s).sum() / mask_s.sum()

        pred_s = self.predictor(proj_s)

        pred_s = F.normalize(pred_s.squeeze(dim=1),dim=1,p=2)
        proj_t = F.normalize(proj_t.squeeze(dim=1),dim=1,p=2)

        proj_t = concat_all_gather(proj_t)

        score = torch.matmul(pred_s, proj_t.transpose(1, 0).detach())

        score = score / self.t

        bs = score.size(0)
        label = (torch.arange(bs, dtype=torch.long) +
                 bs * torch.distributed.get_rank()).cuda()

        losses = dict()
        losses['loss_ct'] = self.ct_weight * 2 * self.t * self.criterion(score, label)
        losses['loss_rc'] = self.rc_weight * loss_rc

        return losses







