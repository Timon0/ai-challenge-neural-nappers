import os

import lightning as L
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset

ANGLEZ_LAGS_FUTURE = [f"anglez_future-lag_{i}" for i in range(-1, -25, -1)]
ENMO_LAGS_FUTURE = [f"enmo_future-lag_{i}" for i in range(-1, -25, -1)]
ANGLEZ_LAGS_PAST = [f"anglez_past-lag_{i}" for i in range(1, 25)]
ENMO_LAGS_PAST = [f"enmo_past-lag_{i}" for i in range(1, 25)]
LAGS_FUTURE = [*ANGLEZ_LAGS_FUTURE, *ENMO_LAGS_FUTURE]
LAGS_PAST = [*ANGLEZ_LAGS_PAST, *ANGLEZ_LAGS_PAST]
FEATURES = ['anglez', 'enmo', *LAGS_PAST, *LAGS_FUTURE]

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
        y = torch.from_numpy(step[LABEL].astype('int64').to_numpy()).squeeze(0, 1)

        return X, y


class CustomDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 100):
        super().__init__()
        self.num_classes = 2
        self.features = FEATURES
        self.validation_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size

    def prepare_data(self):
        # load data
        dirname = os.path.dirname(__file__)
        train_root_dir = os.path.join(dirname, "../data/processed/v2/train")
        validation_root_dir = os.path.join(dirname, "../data/processed/v2/validation")

        train_overview = pd.read_parquet(os.path.join(train_root_dir, 'overview.parquet'))
        validation_overview = pd.read_parquet(os.path.join(validation_root_dir, 'overview.parquet'))

        self.train_dataset = CustomDataSet(train_overview, train_root_dir)
        self.validation_dataset = CustomDataSet(validation_overview, validation_root_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def get_num_features(self):
        return len(self.features)

    def get_num_classes(self):
        return self.num_classes