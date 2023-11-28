import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from rnn import PyTorchRNN

class ConvBlock(nn.Module):

    def __init__(self, conv_in_channels: int, conv_out_channels: int, conv_kernel_size: int, conv_stride: int, pool_kernel_size: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=conv_in_channels,
                      out_channels=conv_out_channels,
                      kernel_size=conv_kernel_size,
                      stride=conv_stride),
            nn.ReLU(),
            # nn.MaxPool1d(pool_kernel_size),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x)


class PyTorchFCN(torch.nn.Module):

    def __init__(self, in_channels: int = 2, num_pred_classes: int = 2, rnn_hidden_dim: int = 128, rnn_hidden_layer_num: int = 1) -> None:
        super(PyTorchFCN, self).__init__()

        self.conv_layers = nn.Sequential(*[
            ConvBlock(in_channels, 128, 8, 1, 3),
            ConvBlock(128, 256, 5, 1, 3),
            ConvBlock(256, 128, 3, 1, 3),
        ])

        self.flatten = nn.Flatten()

        self.classifier = PyTorchRNN(4608, rnn_hidden_dim, rnn_hidden_layer_num, num_pred_classes, 49)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.classifier(x)


class LightningModel(L.LightningModule):

    def __init__(self, model=None, learning_rate=None):
        super(LightningModel, self).__init__()

        self.save_hyperparameters()

        self.num_features = 2
        self.num_classes = 2
        self.learning_rate = learning_rate
        self.seq_len = 49

        # model
        if model is None:
            self.model = PyTorchFCN(self.num_features, self.num_classes, 1024, 1)

        # metrics
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes=self.num_classes),
            MulticlassPrecision(num_classes=self.num_classes),
            MulticlassRecall(num_classes=self.num_classes),
            MulticlassF1Score(num_classes=self.num_classes)
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        loss, true_labels, logits = self._shared_step(batch)

        self.log('train_loss', loss)
        self.train_metrics(logits, true_labels)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)
        return loss

    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].sum()}

    def validation_step(self, batch, batch_nb):
        loss, true_labels, logits = self._shared_step(batch)

        self.log('val_loss', loss)
        self.val_metrics(logits, true_labels)
        self.log_dict(self.val_metrics)

        return loss

    def _shared_step(self, batch):
        features, true_labels = batch

        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        return loss, true_labels, logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=self._num_steps())
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def _num_steps(self) -> int:
        """Get number of steps"""
        train_dataloader = self.trainer.datamodule.train_dataloader()
        dataset_size = len(train_dataloader.dataset)
        num_steps = dataset_size * self.trainer.max_epochs // self.trainer.datamodule.batch_size
        return num_steps