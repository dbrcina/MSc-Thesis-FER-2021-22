from typing import Tuple, Dict, Any, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim
from torch.nn import functional as F


class CRNN(nn.Module):
    def __init__(self, n_classes: int, img_width: int = 100, img_height: int = 32, dropout: float = 0.2) -> None:
        super().__init__()

        self.img_width = img_width
        self.img_height = img_height

        act_fun = nn.ReLU(inplace=True)

        def conv_layer(in_channels: int,
                       out_channels: int,
                       kernel_size: Union[int, Tuple[int, int]],
                       stride: Union[int, Tuple[int, int]],
                       padding: Union[int, Tuple[int, int]],
                       batch_normalization: bool = False) -> nn.Module:
            modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if batch_normalization:
                # When using BatchNorm2d, Conv2d bias is not needed
                modules[-1].bias = None
                modules.append(nn.BatchNorm2d(out_channels))
            modules.append(act_fun)
            return nn.Sequential(*modules)

        self.conv_layers = nn.Sequential(
            conv_layer(1, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # HEIGHT//2 x WIDTH//2
            conv_layer(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # HEIGHT//4 x WIDTH//4
            conv_layer(128, 256, 3, 1, 1),
            conv_layer(256, 256, 3, 1, 1),
            nn.MaxPool2d((2, 1), (2, 1)),  # HEIGHT//8 x WIDTH//4
            conv_layer(256, 512, 3, 1, 1, batch_normalization=True),
            conv_layer(512, 512, 3, 1, 1, batch_normalization=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # HEIGHT//16 x WIDTH//4
            conv_layer(512, 512, 2, 1, 0)  # HEIGHT//16-1 x WIDTH//4-1
        )
        self.output_img_width = img_width // 4 - 1
        self.output_img_height = img_height // 16 - 1

        self.recurrent_layers = nn.LSTM(512, 256, 2, dropout=dropout, bidirectional=True)
        self.output_projection = nn.Linear(2 * 256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (1, self.img_height, self.img_width)

        x = self.conv_layers(x)
        batch_size, n_channels, height, width = x.shape
        assert (height, width) == (self.output_img_height, self.output_img_width)

        # view channels and height as features
        x = x.view(batch_size, n_channels * height, width)
        # (seq_length,batch_size,features)
        x = x.permute(2, 0, 1)
        x, _ = self.recurrent_layers(x)

        x = self.output_projection(x)

        return x


class ALPRLightningModuleCTC(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 model_hparams: Dict[str, Any],
                 optimizer_name: str,
                 optimizer_hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = _MODELS[model_name.lower()](**model_hparams)
        self.optimizer = _OPTIMIZERS[optimizer_name.lower()](self.parameters(), **optimizer_hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return self.optimizer

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, acc = self._step(batch)

        self.log("train_loss", loss)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, acc = self._step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.tensor]:
        x, labels = batch
        logits = self(x)

        loss = self._loss(logits, labels)
        acc = torchmetrics.functional.accuracy(logits, labels)
        return loss, acc


class LeNet5(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.dropout = dropout

        act_fun = nn.ReLU(inplace=True)
        drop_fun = nn.Dropout(p=dropout)

        # In the paper, 32x32 image is expected, but here 28x28 is given, so padding is set to 2
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # Nx16x28x28
            act_fun,
            nn.MaxPool2d(kernel_size=2, stride=2),  # Nx16x14x14
            # 2
            nn.Conv2d(16, 32, kernel_size=5),  # Nx32x10x10
            act_fun,
            nn.MaxPool2d(kernel_size=2, stride=2),  # Nx32x5x5
            # 3
            nn.Conv2d(32, 64, kernel_size=5),  # Nx64x1x1
            act_fun,
            nn.Flatten()  # Nx64 (64*1*1)
        )
        self.classifier = nn.Sequential(
            # 4
            nn.Linear(64, 128),  # Nx128
            act_fun,
            # 5
            drop_fun,
            nn.Linear(128, n_classes)  # Nxn_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (1, 28, 28), "Expecting a grayscale image with size 28x28"

        x = self.features(x)
        x = self.classifier(x)

        return x


class AlexNet(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.dropout = dropout

        act_fun = nn.ReLU(inplace=True)
        drop_fun = nn.Dropout(p=dropout)

        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 16, kernel_size=11, stride=4),  # Nx16x55x55
            act_fun,
            nn.MaxPool2d(kernel_size=3, stride=2),  # Nx16x27x27
            # 2
            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # Nx32x27x27
            act_fun,
            nn.MaxPool2d(kernel_size=3, stride=2),  # Nx32x13x13
            # 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Nx64x13x13
            act_fun,
            # 4
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Nx32x13x13
            act_fun,
            # 5
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # Nx16x13x13
            act_fun,
            nn.MaxPool2d(kernel_size=3, stride=2),  # Nx16x6x6
            nn.Flatten()  # Nx576 (16*6*6)
        )
        self.classifier = nn.Sequential(
            # 6
            drop_fun,
            nn.Linear(576, 512),  # Nx512
            act_fun,
            # 7
            drop_fun,
            nn.Linear(512, 256),  # Nx256
            act_fun,
            # 8
            nn.Linear(256, n_classes)  # Nxn_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (3, 227, 227), "Expecting a color image with size 227x227"

        x = self.features(x)
        x = self.classifier(x)

        return x


_MODELS = {
    "lenet5": LeNet5,
    "alexnet": AlexNet,
    "crnn": CRNN
}

_OPTIMIZERS = {
    "sgd": optim.SGD,
    "adamw": optim.AdamW
}

_LOSSES = {
    "bce": F.binary_cross_entropy_with_logits,
    "ce": F.cross_entropy,
    "ctc": F.ctc_loss
}


class ALPRLightningModule(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 model_hparams: Dict[str, Any],
                 optimizer_name: str,
                 optimizer_hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = _MODELS[model_name.lower()](**model_hparams)
        self.optimizer = _OPTIMIZERS[optimizer_name.lower()](self.parameters(), **optimizer_hparams)

        self.is_binary = model_hparams["n_classes"] == 1
        if self.is_binary:
            self.loss_with_logits = F.binary_cross_entropy_with_logits
        else:
            self.loss_with_logits = F.cross_entropy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return self.optimizer

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self(x)

        if self.is_binary:
            return torch.sigmoid(logits)

        return torch.softmax(logits, dim=1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, acc = self._step(batch)

        self.log("train_loss", loss)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, acc = self._step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.tensor]:
        x, labels = batch
        logits = self(x)

        loss = self._loss(logits, labels)
        acc = torchmetrics.functional.accuracy(logits, labels)
        return loss, acc

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.is_binary:
            return self.loss_with_logits(logits.view(-1), labels.float())

        return self.loss_with_logits(logits, labels)
