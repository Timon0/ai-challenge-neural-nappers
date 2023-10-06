import os

import lightning as L
from torch.nn import functional as F
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ['anglez', 'enmo',
            'hour',
            'anglez_abs', 'anglez_diff', 'enmo_diff', 'anglez_x_enmo',
            'anglez_rolling_mean', 'enmo_rolling_mean', 'anglez_rolling_max', 'enmo_rolling_max', 'anglez_rolling_min',
            'anglez_rolling_std', 'enmo_rolling_std']

LABEL = ['awake']


class DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 100):
        super().__init__()
        self.n_labels = 2
        self.features = FEATURES
        self.validation_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size

    def prepare_data(self):
        # load data
        dirname = os.path.dirname(__file__)
        df_train = pd.read_parquet(os.path.join(dirname, '../data/processed/train_split.parquet'))
        df_validation = pd.read_parquet(os.path.join(dirname, '../data/processed/validation_split.parquet'))

        # split into x and y
        X_train = df_train[FEATURES].astype('float32')
        y_train = df_train[LABEL].astype('int64')
        X_validation = df_validation[FEATURES].astype('float32')
        y_validation = df_validation[LABEL].astype('int64')

        # scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_validation)

        # convert to tensor
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train.to_numpy()).squeeze(1)
        X_validation = torch.from_numpy(X_validation)
        y_validation = torch.from_numpy(y_validation.to_numpy()).squeeze(1)

        # create dataset
        self.train_dataset = TensorDataset(X_train, y_train)
        self.validation_dataset = TensorDataset(X_validation, y_validation)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def get_n_features(self):
        return len(self.features)

    def get_n_labels(self):
        return self.n_labels