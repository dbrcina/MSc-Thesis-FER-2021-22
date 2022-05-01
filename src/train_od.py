import argparse
import os.path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import config
from datasets import ALPRODDataset
from models import ALPRLightningModule


def main(args: argparse.Namespace) -> None:
    data_path = args.data_path
    if not os.path.isdir(data_path):
        print(f"Provided '{data_path}' is not a directory!")
        return

    pl.seed_everything(config.RANDOM_SEED)

    train_dataset = ALPRODDataset(os.path.join(data_path, config.TRAIN_PATH), True)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataset = ALPRODDataset(os.path.join(data_path, config.VAL_PATH), False)
    val_dataloader = DataLoader(val_dataset, batch_size=256)

    model_hparams = {
        "n_classes": 1,
        "groups": [2, 2, 2],
        "c_hidden": [16, 32, 64]
    }
    optimizer_hparams = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-2,
    }
    model = ALPRLightningModule(model_hparams, optimizer_hparams)

    ckp_callback = ModelCheckpoint(dirpath=args.ckp_dir,
                                   filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
                                   monitor="val_loss",
                                   save_weights_only=True)
    callbacks = [ckp_callback]

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("data_path", type=str)
    parser.add_argument("--default_root_dir", default=config.OD_PL_PATH, type=str)
    parser.add_argument("--ckp_dir", default=config.OD_PL_MODELS_PATH, type=str)
    main(parser.parse_args())
