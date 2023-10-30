from encoder import LightningModel
from data import CustomDataModule
import json

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
import wandb

if __name__ == "__main__":
    # Logger
    with open("../../config/config.json", "r") as jsonfile:
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
            #"logger": wandb_logger,
            "callbacks": [checkpoint_callback, early_stopping_callback, lr_monitor],
            "max_epochs": 5
        },
    )

    cli.trainer.fit(cli.model, cli.datamodule)
