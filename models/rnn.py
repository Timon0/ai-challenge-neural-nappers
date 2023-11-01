import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.autograd import Variable


class PyTorchRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, num_classes, seq_len):
        super(PyTorchRNN, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = num_hidden_layers
        self.seq_len = seq_len
        
        # RNN
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_hidden_layers, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        
        out, _ = self.rnn(x)
            
        out = self.fc(out)
        return out


class LightningModel(L.LightningModule):

    def __init__(self, model=None, hidden_dim=None, num_hidden_layers=None, learning_rate=None, seq_len=None):
        super().__init__()

        self.save_hyperparameters()

        self.num_features = 98
        self.num_classes = 2
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        self.seq_len = seq_len

        # model
        if model is None:
            self.model = PyTorchRNN(self.num_features, self.hidden_dim, self.num_hidden_layers, self.num_classes, self.seq_len)

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