from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint

from models import LeNet5


def main(args: Namespace) -> None:
    model = LeNet5(batch_size=args.batch_size, learning_rate=args.learning_rate)

    pb_callback = TQDMProgressBar(refresh_rate=args.pb_refresh_rate)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3)
    ckp_callback = ModelCheckpoint(dirpath=args.ckp_dir,
                                   filename=args.ckp_filename,
                                   monitor="val_loss",
                                   verbose=True,
                                   save_top_k=1)
    callbacks = [pb_callback, early_stop_callback, ckp_callback]

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--ckp_dir", default="..", type=str)
    parser.add_argument("--ckp_filename", default="ocr", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--pb_refresh_rate", default=1, type=int)
    raise SystemExit(main(parser.parse_args()))
