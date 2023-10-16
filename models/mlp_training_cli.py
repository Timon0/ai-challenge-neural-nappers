import sys
import json
import wandb

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI
from watermark import watermark

from data import CustomDataModule
from mlp import LightningModel

if __name__ == "__main__":

    print(watermark(packages="torch,lightning"))

    print(f"The provided arguments are {sys.argv[1:]}", end="\n\n")

    # Params
    model_name = "nn"
    batch_size = 500

    # Logger
    with open("../config/config.json", "r") as jsonfile:
        data = json.load(jsonfile)
        subscription_key = data["wandb"]["subscription_key"]
    wandb.login(key=subscription_key)
    wandb_logger = WandbLogger(project="neural-nappers", log_model=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    early_stopping_callback = EarlyStopping(mode="min", monitor="val_loss", patience=1)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    cli = LightningCLI(
        model_class=LightningModel,
        datamodule_class=CustomDataModule,
        run=False,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=42,
        trainer_defaults={
            "callbacks": [checkpoint_callback, early_stopping_callback, lr_monitor],
            "logger": wandb_logger,
            "max_epochs": 5
        },
    )

    cli.trainer.fit(cli.model, cli.datamodule)