"""
Adapted from: https://github.com/facebookresearch/moco and https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE

Original work is: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
This implementation is: Copyright (c) PyTorch Lightning, Inc. and its affiliates. All Rights Reserved

This implementation is licensed under Attribution-NonCommercial 4.0 International;
You may not use this file except in compliance with the License.

You may obtain a copy of the License from the LICENSE file present in this folder.
"""

import torch, sys

from argparse import ArgumentParser
from typing import Union
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # weights and biases logger
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from torch import nn, optim
from torch.nn import functional as F
from custom_wandb_logger import log_hyperparameters
from moco_data_module import MoCoDataModule
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))

from pl_bolts.metrics import mean, precision_at_k
from transforms import Moco2EvalImagenetTransforms, Moco2TrainImagenetTransforms
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from moco_data_module import MoCoDataModule, UNet_encoder

if _TORCHVISION_AVAILABLE:
    import torchvision
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CEM500K_MEAN = [0.58331613, 0.58331613, 0.58331613]
CEM500K_STD = [0.09966064, 0.09966064, 0.09966064]

MED_MEAN = [0.1806, 0.1806, 0.1806]
MED_STD = [0.1907, 0.1907, 0.1907]

class Moco_v2(LightningModule):
    """PyTorch Lightning implementation of `Moco <https://arxiv.org/abs/2003.04297>`_

    Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.

    Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by:

        - `William Falcon <https://github.com/williamFalcon>`_

    Example::
        from pl_bolts.models.self_supervised import Moco_v2
        model = Moco_v2()
        trainer = Trainer()
        trainer.fit(model)

    CLI command::

        # cifar10
        python moco2_module.py --gpus 1

        # imagenet
        python moco2_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32
    """

    def __init__(
        self,
        base_encoder: Union[str, torch.nn.Module] = UNet_encoder(),
        emb_dim: int = 1024,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        data_dir: str = "./",
        batch_size: int = 256,
        use_mlp: bool = False,
        num_workers: int = 8,
        *args,
        **kwargs
    ):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            data_dir: the directory to store data
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["encoder_k", "encoder_q"])

        # create the encoders
        # num_classes is the output fc dimension
        base_encoder =  UNet_encoder()
        self.encoder_q, self.encoder_k = self.init_encoders(base_encoder)


        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the validation queue
        self.register_buffer("val_queue", torch.randn(emb_dim, num_negatives))
        self.val_queue = nn.functional.normalize(self.val_queue, dim=0)

        self.register_buffer("val_queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self, base_encoder):
        """Override to add your own encoders."""

        import copy
        encoder_q = copy.deepcopy(base_encoder)
        encoder_k = copy.deepcopy(base_encoder)

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ptr, queue):
        # gather keys before updating queue
        if self._use_ddp_or_ddp2(self.trainer):
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no cover
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no cover
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, img_k, queue):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            queue: a queue from which to pick negative samples
        Output:
            logits, targets
        """

        # compute query features
    
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            # shuffle for making use of BN
            if self._use_ddp_or_ddp2(self.trainer):
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self._use_ddp_or_ddp2(self.trainer):
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        return logits, labels, k, q

    def _compute_l_s(self, output, target, keys, queue):
        """computes the $l_s$ loss from the LoGo paper.
        In this case this is equal to the Info-NCE loss.

        :param img_q: query image (torch.tensor)
        :param img_k: key image (torch.tensor)
        :param queue: queue List if negative samples
        :returns: loss

        """
        # output, target, keys, queries = self(img_q=img_q, img_k=img_k, queue=self.queue)
        self._dequeue_and_enqueue(keys, queue=self.queue, queue_ptr=self.queue_ptr)  # dequeue and enqueue
        loss = F.cross_entropy(output.float(), target.long())
        return loss

    def training_step(self, batch, batch_idx):
        # in STL10 we pass in both lab+unl for online ft
        if self.trainer.datamodule.name == "stl10":
            # labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (x_g), pos_crop = batch  # get global and local crops

        self._momentum_update_key_encoder()  # update the key encoder

        loss = []
        output, target, keys, queries = self(img_q=x_g[0], img_k=x_g[1], queue=self.queue)
        loss_gg = self._compute_l_s(output, target, keys, queue=self.queue)
        loss.append(loss_gg)

        # if batch_idx % 2000 == 0:
        # Combine all the individual loss terms to one single loss
        loss = sum(loss)

        log = {"train_loss": loss}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        # in STL10 we pass in both lab+unl for online ft
        if self.trainer.datamodule.name == "stl10":
            # labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), labels = batch

        output, target, keys, _ = self(img_q=img_1, img_k=img_2, queue=self.val_queue)
        self._dequeue_and_enqueue(keys, queue=self.val_queue, queue_ptr=self.val_queue_ptr)  # dequeue and enqueue

        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.trainer.max_epochs,
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--base_encoder", type=str, default="resnet50")
        parser.add_argument("--emb_dim", type=int, default=1024)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--num_negatives", type=int, default=65536)
        parser.add_argument("--encoder_momentum", type=float, default=0.999)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=0.08)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--img_dir", type=str)
        parser.add_argument("--mask_dir", type=str)
        parser.add_argument("--dicom_dir", type=str)
        parser.add_argument(
            "--dataset",
            type=str,
            default="cifar10",
            choices=[
                "medical",
            ],
        )
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--use_mlp", action="store_true")
        parser.add_argument("--meta_dir", default=".", type=str, help="path to meta.bin for imagenet")
        parser.add_argument("--global_to_local", action="store_true")
        parser.add_argument("--local_to_local", action="store_true")
        parser.add_argument("--lambda_", type=float, default=0.0005)
        parser.add_argument("--crop_size", type=int, default=64)
        parser.add_argument("--not_load_from_ram", action="store_false")
        parser.add_argument("--global_size", type=int, default=224)
        parser.add_argument("--reduced", action="store_true")
        # parser.add_argument("--accumulate_grad_batches", type=int, default=1)

        # wandb arguemnts
        parser.add_argument("--savepath", default="../selfsupervised_pretraining/MOCO/pl_bolts/models/self_supervised/moco/results/5_v2", type=str, help="Path to save checkpoints")
        parser.add_argument("--offline", action="store_true", help="Offline does not save metrics on wandb")
        parser.add_argument("--wandb_group", default="MoCoV2", type=str, help="Wandb group name")
        parser.add_argument("--wandb_job_type", default="Pre-training", type=str, help="Wandb job type")
        parser.add_argument("--tags", nargs="*", default=["MoCoV2"], type=str, help="Wandb tags, default = MoCoV2")
        parser.add_argument("--name", default=None, type=str, help="Set run name to identify run in wandb web UI.")

        return parser

    @staticmethod
    def _use_ddp_or_ddp2(trainer: Trainer) -> bool:
        return isinstance(trainer.strategy, (DDPPlugin, DDP2Plugin))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    from pl_bolts.datamodules import SSLImagenetDataModule

    parser = ArgumentParser()

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = Moco_v2.add_model_specific_args(parser)
    args = parser.parse_args()

    # weights and biases
    wandb_logger = WandbLogger(
        project="MoCoV2",
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        tags=args.tags,
        offline=args.offline,
        name=args.name,
    )

    # callbacks
    model_checkpoint = ModelCheckpoint(
        dirpath=args.savepath,
        filename="{epoch}-{train_loss:.2f}",
        save_last=True,
        save_top_k=1000,
        # every_n_epochs=20,
        monitor="train_loss",
    )

    datamodule = MoCoDataModule(**args.__dict__)

    model = Moco_v2(**args.__dict__)

    trainer = Trainer.from_argparse_args(args, gpus=1, logger=wandb_logger, callbacks=model_checkpoint)

    # logging
    object_dict = {
        "cfg": vars(args),
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    log_hyperparameters(object_dict)

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    cli_main()
