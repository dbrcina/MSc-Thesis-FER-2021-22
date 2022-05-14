from typing import Tuple, Dict, Any

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim
from torch.nn import functional as F


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
    "alexnet": AlexNet
}

_OPTIMIZERS = {
    "sgd": optim.SGD,
    "adamw": optim.AdamW
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
