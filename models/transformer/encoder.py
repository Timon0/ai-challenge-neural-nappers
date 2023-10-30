import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import time

class TransformerEncoderClassifier(nn.Module):

    def __init__(self, encoder_layer_dim=2, encoder_layer_nhead=8, num_features=2, num_layers=6, num_classes=2,
                 sequence_length=49):
        super().__init__()

        self.encoder_layer_dim = encoder_layer_dim
        self.encoder_layer_nhead = encoder_layer_nhead
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.encoder_layer_dim,
                                                        nhead=self.encoder_layer_nhead,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.feature_linear = nn.Linear(self.num_features, 1)
        self.sequence_linear = nn.Linear(self.sequence_length, self.num_classes)

        self.init_weights()

    def forward(self, x):
        output = self.transformer_encoder(x)
        output = self.feature_linear(output)
        output = torch.squeeze(output, dim=2)
        output = self.sequence_linear(output)
        return output

    def init_weights(self) -> None:
        init_range = 0.1
        self.feature_linear.bias.data.zero_()
        self.feature_linear.weight.data.uniform_(-init_range, init_range)

        self.sequence_linear.bias.data.zero_()
        self.sequence_linear.weight.data.uniform_(-init_range, init_range)


class LightningModel(L.LightningModule):

    def __init__(self, model=None, encoder_layer_dim=64, encoder_layer_nhead=1, num_layers=1, learning_rate=None):
        super().__init__()

        self.save_hyperparameters()

        self.sequence_length = 49
        self.num_features = 2
        self.num_classes = 2
        self.encoder_layer_dim = encoder_layer_dim
        self.encoder_layer_nhead = encoder_layer_nhead
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        if model is None:
            self.model = TransformerEncoderClassifier(self.num_features,
                                                      self.encoder_layer_nhead,
                                                      self.num_features,
                                                      self.num_layers,
                                                      self.num_classes,
                                                      self.sequence_length)
        else:
            self.model = model

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

    def training_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch)

        self.log('train_loss', loss)
        self.train_metrics(logits, true_labels)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_nb):
        loss, true_labels, logits = self._shared_step(batch)

        self.log('val_loss', loss)
        self.val_metrics(logits, true_labels)
        self.log_dict(self.val_metrics)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=self._num_steps())
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        return loss, true_labels, logits

    def _num_steps(self):
        train_dataloader = self.trainer.datamodule.train_dataloader()
        dataset_size = len(train_dataloader.dataset)
        num_steps = dataset_size * self.trainer.max_epochs // self.trainer.datamodule.batch_size
        return num_steps
