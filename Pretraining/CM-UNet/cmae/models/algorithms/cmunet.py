import torch
from .base import BaseModel
from cmae.registry import MODELS
import torch.nn as nn

@MODELS.register_module()
class CM_UNet(BaseModel):
    """
    CM_UNet model implementing a contrastive learning-based U-Net variant.

    Args:
        backbone (dict): Dictionary specifying the online and target backbone networks.
        neck (dict): Dictionary specifying the pixel, feature, and projector modules.
        head (dict): Dictionary specifying the head module.
        base_momentum (float, optional): Momentum factor for updating target networks. Defaults to 0.996.
        init_cfg (dict, optional): Initialization configuration. Defaults to None.
        target_cls (bool, optional): Whether to use the target classifier. Defaults to True.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 base_momentum=0.996,
                 init_cfg=None,
                 target_cls=True,
                 **kwargs):
        super(CM_UNet, self).__init__(backbone=backbone['online'],init_cfg=init_cfg)
        assert neck is not None

        self.target_backbone = MODELS.build(backbone['target'])

        self.pixel_decoder = MODELS.build(neck['pixel'])
        
        for param in self.pixel_decoder.parameters():
            param.requires_grad = True
        for name, param in self.pixel_decoder.named_parameters():
            if "up_conv" in name:  # Replace 'layer_name' with the specific layer name
                param.requires_grad = True
                
        self.feature_decoder = MODELS.build(neck['feature'])

        self.projector = MODELS.build(neck['projector'])
        self.target_projector = MODELS.build(neck['projector'])

        self.target_cls = target_cls

        assert head is not None

        self.head = MODELS.build(head)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

        for param_m in self.target_backbone.parameters():
            param_m.requires_grad = False

        for param_m in self.target_projector.parameters():
            param_m.requires_grad = False

    def init_weights(self):
        """
        Initializes model weights by copying parameters from the online backbone 
        to the target backbone and projector.
        """
        super(CM_UNet, self).init_weights()

        for param_b, param_m in zip(self.backbone.parameters(),
                                    self.target_backbone.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

        for param_b, param_m in zip(self.projector.parameters(),
                                    self.target_projector.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """
        Updates the target backbone and projector parameters using momentum-based 
        exponential moving average.
        """
        for param_b, param_m in zip(self.backbone.parameters(),
                                    self.target_backbone.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)

        for param_b, param_m in zip(self.projector.parameters(),
                                    self.target_projector.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)


    def extract_feat(self, img):
        """
        Extracts features from the input image using the online backbone.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Extracted feature map.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, img_t=None, **kwargs):
        """
        Forward pass for training, including feature extraction, pixel and feature decoding,
        projection, and loss computation.

        Args:
            img (torch.Tensor): Input image tensor.
            img_t (torch.Tensor, optional): Target image tensor. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            dict: Computed losses.
        """
        latent_s, mask_s, skip_s = self.backbone(img)
        latent_t, _, skip_t = self.target_backbone(img_t)
        pred_pixel = self.pixel_decoder(latent_s, skip_s)
        pred_feature = self.feature_decoder(latent_s, skip_s)
        
        proj_s = self.projector(torch.mean(pred_feature, dim=1, keepdim=True))
        
        reduce_channels = nn.Conv2d(1024, 256, kernel_size=1).to(latent_t.dtype).cuda()
        latent_t = reduce_channels(latent_t)
        latent_t = latent_t.view(latent_t.shape[0], -1).view(latent_t.shape[0], 1, 224, 224)
        proj_t = self.target_projector(torch.mean(latent_t,dim=1,keepdim=True))
        
        losses = self.head(img, pred_pixel[:,1], mask_s, proj_s, proj_t)
        
        return losses











