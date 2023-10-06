import json

import torch
import wandb
from lightning.pytorch import loggers as pl_loggers, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from data import DataModule, FEATURES
from mlp import Model


if __name__ == "__main__":
    torch.cuda.empty_cache()
    seed_everything(42, workers=True)

    # Hyperparams
    model_name = "neural-network"
    batch_size = 200
    learning_rate = 1e-5

    # Logger
    with open("../config/config.json", "r") as jsonfile:
        data = json.load(jsonfile)
        subscription_key = data["wandb"]["subscription_key"]
    wandb.login(key=subscription_key)
    wandb_logger = pl_loggers.WandbLogger(project="neural-nappers")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{model_name}-batch{batch_size}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Training
    data_module = DataModule(batch_size)
    model = Model(model_name, n_feature=data_module.get_n_features(), n_labels=data_module.get_n_labels(), hidden_dim=64, n_hidden_layers=4, learning_rate=learning_rate)
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=5,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor]
    )
    trainer.fit(model, datamodule=data_module)
