# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from timm import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.models.layers import drop

from models.custom import UNET_MAE


# log more
def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy, drop.DropPath):
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'


pretrain_default_model_kwargs = {
    'unet_sparse': dict(),
    'unet': dict(),
}
for kw in pretrain_default_model_kwargs.values():
    kw['pretrained'] = False
    kw['num_classes'] = 0
    kw['global_pool'] = ''


def build_sparse_encoder(name: str, input_size: int, sbn=False, drop_path_rate=0.0, verbose=False):
    from encoder import SparseEncoder
    
    kwargs = pretrain_default_model_kwargs[name]
    if drop_path_rate != 0:
        kwargs['drop_path_rate'] = drop_path_rate
    print(f'[build_sparse_encoder] model kwargs={kwargs}')
    cnn = create_model(name, in_chans=1, **kwargs)
    
    return SparseEncoder(cnn, input_size=input_size, sbn=sbn, verbose=verbose)

