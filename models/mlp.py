import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW


class Model(L.LightningModule):

    def __init__(self, model_name, n_feature, n_labels, hidden_dim, n_hidden_layers, learning_rate=1e-5):
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
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y = batch
        logits = self(X)

        # Apart from the validation loss, we also want to track validation accuracy to get an idea, what the
        # model training has achieved "in real terms".
        loss = self.loss_fn(logits, y)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (y == predictions).float().mean()

        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def test_step(self, batch, batch_nb):
        X, y = batch
        logits = self(X)

        # Apart from the test loss, we also want to track test accuracy to get an idea, what the
        # model training has achieved "in real terms".
        loss = self.loss_fn(logits, y)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (y == predictions).float().mean()

        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        return {'test_loss': loss, 'test_accuracy': accuracy}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer
