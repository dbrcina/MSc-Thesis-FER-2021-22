import os.path
from typing import Dict, Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from torch.utils.data import DataLoader

import src.config as config
from src.datasets import ALPROCRDataset, ALPRODDataset
from src.models import LeNet5, AlexNet

_MODELS = {
    "lenet5": LeNet5,
    "alexnet": AlexNet
}

_OPTIMIZERS = {
    "sgd": optim.SGD,
    "adamw": optim.AdamW
}

_DATASETS = {
    "od": ALPRODDataset,
    "ocr": ALPROCRDataset
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


def create_trainer_and_fit(args: Dict[str, Any]) -> pl.Trainer:
    data_path = args["data_path"]
    dataset_class = _DATASETS[args["dataset_name"]]
    train_path = os.path.join(data_path, config.TRAIN_PATH)
    val_path = os.path.join(data_path, config.VAL_PATH)

    train_dataset = dataset_class(train_path, train=True)
    val_dataset = dataset_class(val_path, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args["val_batch_size"])

    model = ALPRLightningModule(**args["alpr_module"])

    ckp_callback = ModelCheckpoint(dirpath=args["ckp_dir"],
                                   filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
                                   monitor="val_loss",
                                   save_weights_only=True)

    trainer = pl.Trainer(**args["trainer"], callbacks=[ckp_callback])
    trainer.logger._default_hp_metric = None
    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer


if __name__ == "__main__":
    args = {
        "data_path": "data_ocr",
        "dataset_name": "ocr",
        "train_batch_size": 1024,
        "val_batch_size": 256,
        "alpr_module": {
            "model_name": "lenet5",
            "model_hparams": {
                "n_classes": 36,
                "dropout": 0.3
            },
            "optimizer_name": "sgd",
            "optimizer_hparams": {
                "lr": 1,
                # "momentum": 0.9,
                # "betas": (0.9, 0.999),
                # "weight_decay": 1e-4
            }
        },
        "ckp_dir": "ocr/models",
        "trainer": {
            "default_root_dir": "ocr",
            # "gpus": 1,
            "max_epochs": 10
        }
    }
    create_trainer_and_fit(args)
