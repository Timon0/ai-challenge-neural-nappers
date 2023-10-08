import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score


class Model(L.LightningModule):

    def __init__(self, model_name, n_feature, n_labels, hidden_dim, n_hidden_layers, learning_rate):
        super().__init__()

        self.save_hyperparameters()

        self.n_feature = n_feature
        self.n_labels = n_labels
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate

        # model Layers
        self.up_projection = nn.Linear(n_feature, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(0, n_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.down_projection = nn.Linear(hidden_dim, n_labels)

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # metrics
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes=n_labels), MulticlassPrecision(num_classes=n_labels), MulticlassRecall(num_classes=n_labels), MulticlassF1Score(num_classes=n_labels)
        ])
        self.valid_metrics = metrics.clone(prefix='val_')

    def forward(self, x):
        x = self.up_projection(x)
        x = F.relu(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)

        x = self.down_projection(x)
        return x

    def training_step(self, batch, batch_nb):
        X, y = batch
        logits = self(X)

        loss = self.loss_fn(logits, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        X, y = batch
        logits = self(X)

        loss = self.loss_fn(logits, y)
        valid_metrics = self.valid_metrics(logits, y)

        self.log('val_loss', loss)
        self.log_dict(valid_metrics)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer
