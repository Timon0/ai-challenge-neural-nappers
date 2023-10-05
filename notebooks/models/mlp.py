import torch.nn as nn


class Model(nn.Module):

    def __init__(self, n_features):
        super(Model, self).__init__()

        self.linear = nn.Linear(n_features, 2)

    def forward(self, x):
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_nb):
        outputs = self(batch)

        self.log('train_loss', outputs.loss)

        return {'loss': outputs.loss}
