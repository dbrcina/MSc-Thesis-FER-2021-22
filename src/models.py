from typing import Tuple, Dict, Any, Union, List

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F


def conv_layer(in_channels: int,
               out_channels: int,
               kernel_size: Union[int, Tuple[int, int]],
               stride: Union[int, Tuple[int, int]],
               padding: Union[int, Tuple[int, int]],
               act_fn: nn.Module,
               batch_normalization: bool = False) -> nn.Module:
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if batch_normalization:
        # When using BatchNorm2d, Conv2d bias is not needed
        modules[-1].bias = None
        modules.append(nn.BatchNorm2d(out_channels))
    modules.append(act_fn)
    return nn.Sequential(*modules)


class CNNBackbone(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 in_width: int = 100,
                 in_height: int = 32,
                 act_fn: nn.Module = nn.ReLU(inplace=True)) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height

        self.net = nn.Sequential(
            conv_layer(in_channels, 16, 3, 1, 1, act_fn),
            nn.MaxPool2d(2, 2),  # HEIGHT//2 x WIDTH//2
            conv_layer(16, 32, 3, 1, 1, act_fn),
            nn.MaxPool2d(2, 2),  # HEIGHT//4 x WIDTH//4
            conv_layer(32, 64, 3, 1, 1, act_fn),
            conv_layer(64, 64, 3, 1, 1, act_fn),
            nn.MaxPool2d((2, 1), (2, 1)),  # HEIGHT//8 x WIDTH//4
            conv_layer(64, 128, 3, 1, 1, act_fn, batch_normalization=True),
            conv_layer(128, 128, 3, 1, 1, act_fn, batch_normalization=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # HEIGHT//16 x WIDTH//4
            conv_layer(128, 128, 2, 1, 0, act_fn)  # HEIGHT//16-1 x WIDTH//4-1
        )

        self.out_channels = 128
        self.out_width = in_width // 4 - 1
        self.out_height = in_height // 16 - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (self.in_channels, self.in_height, self.in_width)
        x = self.net(x)
        assert x.shape[1:] == (self.out_channels, self.out_height, self.out_width)
        return x


class Detector(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.2) -> None:
        super().__init__()

        act_fn = nn.ReLU(inplace=True)
        drop_fn = nn.Dropout(p=dropout)

        self.features = CNNBackbone(act_fn=act_fn)

        self.classifier = nn.Sequential(
            drop_fn,
            nn.Linear(self.features.out_channels * self.features.out_height * self.features.out_width, 1024),
            act_fn,
            drop_fn,
            nn.Linear(1024, 256),
            act_fn,
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


class Recognizer(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.2) -> None:
        super().__init__()

        self.conv_layers = CNNBackbone()
        self.recurrent_layers = nn.LSTM(self.conv_layers.out_channels, 64, 2, dropout=dropout, bidirectional=True)
        self.output_projection = nn.Linear(2 * 64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        batch_size, n_channels, height, width = x.shape

        # view channels and height as features
        x = x.view(batch_size, n_channels * height, width)
        # (seq_length,batch_size,features)
        x = x.permute(2, 0, 1)
        x, _ = self.recurrent_layers(x)

        x = self.output_projection(x)
        return x


_MODELS = {
    "detector": Detector,
    "recognizer": Recognizer
}

_OPTIMIZERS = {
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
    "adadelta": optim.Adadelta
}

_LR_SCHEDULERS = {
    "reducelronplateau": optim.lr_scheduler.ReduceLROnPlateau
}


class ALPRLightningModule(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 model_hparams: Dict[str, Any],
                 optimizer_name: str,
                 optimizer_hparams: Dict[str, Any],
                 lr_scheduler_name: str,
                 lr_scheduler_hparams: Dict[str, Any],
                 loss_name: str) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = _MODELS[model_name.lower()](**model_hparams)
        self.optimizer = _OPTIMIZERS[optimizer_name.lower()](self.parameters(), **optimizer_hparams)
        self.lr_scheduler = _LR_SCHEDULERS[lr_scheduler_name.lower()](self.optimizer, **lr_scheduler_hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss"
            }
        }

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss = self._step(batch)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}

    def _step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        loss = None

        loss_name = self.hparams["loss_name"]

        if loss_name == "ctc":
            x, targets, target_lengths = batch
            logits = self(x)
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = torch.tensor(len(x) * [logits.shape[0]])
            loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        elif loss_name == "bce":
            x, labels = batch
            logits = self(x)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float())

        return loss

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._epoch_end(outputs, "Train/Loss")

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._epoch_end(outputs, "Val/Loss")

    def _epoch_end(self, outputs: List[Dict[str, torch.Tensor]], graph_tag: str) -> None:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(graph_tag, avg_loss, self.current_epoch)
