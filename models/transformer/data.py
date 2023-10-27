import os

import pandas as pd
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

FEATURES = ['anglez', 'enmo',
            'hour',
            'anglez_abs', 'anglez_diff', 'enmo_diff', 'anglez_x_enmo',
            'anglez_rolling_mean', 'enmo_rolling_mean', 'anglez_rolling_max', 'enmo_rolling_max', 'anglez_rolling_min',
            'anglez_rolling_std', 'enmo_rolling_std']

LABEL = ['awake']


class CustomDataSet(Dataset):
    def __init__(self, overview, root_dir):
        self.overview = overview
        self.root_dir = root_dir

        self.cache_series_id = None
        self.cache_series = None

    def __len__(self):
        return len(self.overview)

    def __getitem__(self, idx):
        series_id, step = self.overview.iloc[idx]

        if self.cache_series_id != series_id:
            path = os.path.join(self.root_dir, series_id + ".parquet")
            series = pd.read_parquet(path)
            self.cache_series_id = series_id
            self.cache_series = series

        step = self.cache_series[self.cache_series.step == step]

        X = torch.from_numpy(step[FEATURES].values).squeeze(0)
        y = torch.from_numpy(step[LABEL].astype('int64').to_numpy()).squeeze(0)

        return X, y


class CustomDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.num_classes = 2
        self.validation_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size

    def prepare_data(self):
        # load data
        dirname = os.path.dirname(__file__)
        train_root_dir = os.path.join(dirname, "../data/processed")
        validation_root_dir = os.path.join(dirname, "../data/processed")

        train_data = pd.read_parquet(os.path.join(train_root_dir, 'train_series_split.parquet'))
        validation_data = pd.read_parquet(os.path.join(validation_root_dir, 'validation_series_split.parquet'))

        self.train_dataset = CustomDataSet(train_data)
        self.validation_dataset = CustomDataSet(validation_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def get_num_classes(self):
        return self.num_classes