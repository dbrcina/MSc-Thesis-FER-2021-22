import os
from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets import RecognitionDataset, DetectionDataset
from models import ALPRLightningModule

DATASETS = {
    "detection": DetectionDataset,
    "recognition": RecognitionDataset
}

COLLATE_FNS = {
    "recognition": RecognitionDataset.collate_fn
}


def create_trainer_and_fit(args: Dict[str, Any]) -> pl.Trainer:
    data_path = args["data_path"]
    dataset_class = DATASETS[args["dataset_name"]]
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")

    train_dataset = dataset_class(train_path, True)
    val_dataset = dataset_class(val_path, False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args["train_batch_size"],
                                  shuffle=True,
                                  collate_fn=COLLATE_FNS.get(args.get("collate_fn")))
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args["val_batch_size"],
                                collate_fn=COLLATE_FNS.get(args.get("collate_fn")))

    model = ALPRLightningModule(**args["alpr_module"])

    ckp_callback = ModelCheckpoint(dirpath=args["ckp_dir"],
                                   filename="{epoch:02d}-{val_loss:.4f}",
                                   monitor="val_loss",
                                   save_weights_only=True)

    trainer = pl.Trainer(**args["trainer"], callbacks=[ckp_callback])
    trainer.logger._default_hp_metric = None
    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer


if __name__ == "__main__":
    recognition_config = {
        "data_path": "data/recognition",
        "dataset_name": "recognition",
        "train_batch_size": 512,
        "val_batch_size": 256,
        "collate_fn": "recognition",
        "alpr_module": {
            "model_name": "recognizer",
            "model_hparams": {
                "n_classes": 37,
                "dropout": 0.2
            },
            "optimizer_name": "adamw",
            "optimizer_hparams": {
                "lr": 1e-2,
                "betas": (0.9, 0.999),
                "weight_decay": 1e-4
            },
            "loss_name": "ctc"
        },
        "ckp_dir": "pl/recognition",
        "trainer": {
            "default_root_dir": "pl/recognition",
            "gpus": 0,
            "max_epochs": 1,
            "fast_dev_run": True,
            "limit_train_batches": 0.1,
            "limit_val_batches": 0.1
        }
    }

    detection_config = {
        "data_path": "data/detection",
        "dataset_name": "detection",
        "train_batch_size": 256,
        "val_batch_size": 256,
        "alpr_module": {
            "model_name": "detector",
            "model_hparams": {
                "n_classes": 1,
                "dropout": 0.5
            },
            "optimizer_name": "adamw",
            "optimizer_hparams": {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "weight_decay": 1e-4
            },
            "loss_name": "bce"
        },
        "ckp_dir": "pl/detection",
        "trainer": {
            "default_root_dir": "pl/detection",
            "gpus": 0,
            "max_epochs": 1,
            "fast_dev_run": True,
            "limit_train_batches": 0.1,
            "limit_val_batches": 0.1
        }
    }

    create_trainer_and_fit(recognition_config)
