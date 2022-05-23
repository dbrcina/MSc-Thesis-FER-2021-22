from typing import Tuple, Dict, Any, Union

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
    def __init__(self, in_channels: int = 1, in_width: int = 100, in_height: int = 32) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height

        act_fn = nn.ReLU(inplace=True)

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

        self.out_width = in_width // 4 - 1
        self.out_height = in_height // 16 - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (self.in_channels, self.in_height, self.in_width)

        x = self.net(x)

        assert x.shape[2:] == (self.out_height, self.out_width)

        return x


class CRNN(nn.Module):
    def __init__(self, n_classes: int, img_width: int = 100, img_height: int = 32, dropout: float = 0.2) -> None:
        super().__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.output_img_width = img_width // 4 - 1
        self.output_img_height = img_height // 16 - 1

        act_fn = nn.ReLU(inplace=True)

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
            modules.append(act_fn)
            return nn.Sequential(*modules)

        self.conv_layers = nn.Sequential(
            conv_layer(1, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # HEIGHT//2 x WIDTH//2
            conv_layer(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # HEIGHT//4 x WIDTH//4
            conv_layer(32, 64, 3, 1, 1),
            conv_layer(64, 64, 3, 1, 1),
            nn.MaxPool2d((2, 1), (2, 1)),  # HEIGHT//8 x WIDTH//4
            conv_layer(64, 128, 3, 1, 1, batch_normalization=True),
            conv_layer(128, 128, 3, 1, 1, batch_normalization=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # HEIGHT//16 x WIDTH//4
            conv_layer(128, 128, 2, 1, 0)  # HEIGHT//16-1 x WIDTH//4-1
        )

        self.recurrent_layers = nn.LSTM(128, 64, 2, dropout=dropout, bidirectional=True)
        self.output_projection = nn.Linear(2 * 64, n_classes)

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


class AlexNet(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

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
    "alexnet": AlexNet,
    "crnn": CRNN
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
                 optimizer_hparams: Dict[str, Any],
                 loss_name: str) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = _MODELS[model_name.lower()](**model_hparams)
        self.optimizer = _OPTIMIZERS[optimizer_name.lower()](self.parameters(), **optimizer_hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return self.optimizer

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def _step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        loss = None

        loss_name = self.hparams["loss_name"]

        if loss_name == "ctc":
            x, targets, target_lengths = batch
            logits = self(x)
            log_probs = F.log_softmax(logits, dim=2)
            input_lengths = torch.tensor(len(x) * [logits.shape[0]])
            loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        elif loss_name == "bce":
            x, labels = batch
            logits = self(x)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float())

        return loss
