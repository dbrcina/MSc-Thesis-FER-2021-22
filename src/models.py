from typing import Any

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim


class ResNetBlock(nn.Module):
    def __init__(self, c_in: int, act_fn: nn.Module, subsample: bool = False, c_out: int = -1) -> None:
        """
        Inputs:
            c_in - Number of input channels.
            subsample - If True, input image width and height is reduced by 2 times.
            c_out - Number of output channels. This is only relevant if subsample is True, otherwise c_out = c_in.
        """

        super().__init__()

        if not subsample:
            c_out = c_in
        else:
            if c_out <= 0:
                raise RuntimeError(f"'c_out' needs to be > 0 when subsampling, but it is {c_out}!")

        self.act_fn = act_fn
        self.net = nn.Sequential(  # Bias is not necessary when using batch normalization.
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1 if not subsample else 2, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            self.act_fn,
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, padding=1, stride=2) if subsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        residual = f + x
        non_linearity_residual = self.act_fn(residual)
        return non_linearity_residual


class ResNet(nn.Module):
    def __init__(self, n_classes: int, groups: list[int], c_hidden: list[int], c_in: int = 3) -> None:
        """
        Inputs:
            n_classes - Number of classification outputs.
            groups - List with the number of ResNet blocks to use. The first block of each group uses
            down-sampling, except the first.
            c_hidden - List with the hidden dimensions. Usually multiplied by 2 the deeper network goes.
            c_in - Number of input channels. Defaults to 3.
        """

        super().__init__()

        assert len(groups) == len(c_hidden)
        assert len(groups) > 0

        act_fn = nn.ReLU(inplace=True)

        # Construct input layer.
        self.input_net = nn.Sequential(
            nn.Conv2d(c_in, c_hidden[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_hidden[0]),
            act_fn
        )

        # Construct ResNet blocks.
        blocks = []
        for group_idx, block_count in enumerate(groups):
            for i in range(block_count):
                subsample = i == 0 and group_idx > 0
                # If subsample is True, then we are at the beginning of a group, so take dims from previous group,
                # otherwise take dims from current group.
                block_c_in = c_hidden[(group_idx - 1) if subsample else group_idx]
                block_c_out = c_hidden[group_idx]
                blocks.append(ResNetBlock(block_c_in, act_fn, subsample, block_c_out))
        self.blocks = nn.Sequential(*blocks)

        # Construct output layer.
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_net(x)
        out = self.blocks(out)
        out = self.output_net(out)
        return out


class ALPRLightningModule(pl.LightningModule):
    def __init__(self,
                 model_hparams: dict[str, Any],
                 optimizer_hparams: dict[str, Any],
                 lr_scheduler_hparams: dict[str, Any]) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.model = ResNet(**model_hparams)
        self.loss_module = nn.BCEWithLogitsLoss()
        self.accuracy_fn = torchmetrics.functional.accuracy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.hparams.lr_scheduler_hparams)
        return [optimizer], [scheduler]

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, labels = batch
        preds = self(x)
        loss = self.loss_module(preds, labels)
        acc = self.accuracy_fn(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, labels = batch
        preds = self(x)
        loss = self.loss_module(preds, labels)
        acc = self.accuracy_fn(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
