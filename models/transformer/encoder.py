import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import math


class ClassificationHead(nn.Module):
    def __init__(self, d_model, seq_len, n_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.seq(x)
        return x


class TransformerEncoderClassifier(nn.Module):

    def __init__(self, num_features=2, encoder_layer_nhead=4, num_layers=2, dim_model=64, num_classes=2,
                 sequence_length=49, dropout: float = 0.1):
        super().__init__()

        self.model_type = 'Transformer'

        self.num_features = num_features
        self.encoder_layer_nhead = encoder_layer_nhead
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        self.embedding = nn.Linear(self.num_features, self.dim_model)

        self.pos_encoder = PositionalEncoding(self.dim_model, dropout, self.sequence_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_model,
                                                   nhead=self.encoder_layer_nhead,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.classifier = ClassificationHead(seq_len=sequence_length, d_model=self.dim_model, n_classes=num_classes)

    def forward(self, src):
        output = self.embedding(src)
        output = self.pos_encoder(output)
        output = self.encoder(output)
        return self.classifier(output)


class LightningModel(L.LightningModule):

    def __init__(self, model=None, encoder_layer_nhead=4, num_layers=2, dim_model=64, learning_rate=None):
        super().__init__()

        self.save_hyperparameters()

        self.sequence_length = 49
        self.num_features = 2
        self.num_classes = 2
        self.encoder_layer_nhead = encoder_layer_nhead
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.learning_rate = learning_rate

        if model is None:
            self.model = TransformerEncoderClassifier(self.num_features,
                                                      self.encoder_layer_nhead,
                                                      self.num_layers,
                                                      self.dim_model,
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      randomly zeroes-out some of the input
          max_length:   max sequence length
        """
        # inherit from Module
        super().__init__()

        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)

        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)

        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0)

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
        Args:
          x:        embeddings (batch_size, seq_length, d_model)

        Returns:
                    embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        # perform dropout
        return self.dropout(x)
