import argparse
import os.path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from torch.utils.data import DataLoader

import config
import utils
from datasets import ALPRDataset
from models import ALPRLightningModule


def main(args: argparse.Namespace) -> None:
    data_path = args.data_path
    if not os.path.isdir(data_path):
        print(f"Provided '{data_path}' is not a directory!")
        return

    pl.seed_everything(config.RANDOM_SEED)

    train_dataset = ALPRDataset(utils.join_multiple_paths(data_path, config.TRAIN_PATH), True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = ALPRDataset(utils.join_multiple_paths(data_path, config.VAL_PATH), False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    model_hparams = {"n_classes": 1, "groups": [1], "c_hidden": [2]}
    optimizer_hparams = {"lr": 1e-1, "momentum": 0.9, "weight_decay": 1e-4}
    lr_scheduler_hparams = {"mode": "min", "factor": 0.1, "patience": 5}
    model = ALPRLightningModule(model_hparams, optimizer_hparams, lr_scheduler_hparams)

    pb_callback = TQDMProgressBar(refresh_rate=args.pb_refresh_rate)
    ckp_callback = ModelCheckpoint(dirpath=args.ckp_dir,
                                   filename="od-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
                                   monitor="val_loss",
                                   save_weights_only=True)
    callbacks = [pb_callback, ckp_callback]

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("data_path", type=str)
    parser.add_argument("--default_root_dir", default="od", type=str)
    parser.add_argument("--ckp_dir", default="od/models", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--pb_refresh_rate", default=1, type=int)
    main(parser.parse_args())
