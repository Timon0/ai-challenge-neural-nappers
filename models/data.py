import os

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ANGLEZ_LAGS_FUTURE = [f"anglez_future-lag_{i}" for i in range(-1, -25, -1)]
ENMO_LAGS_FUTURE = [f"enmo_future-lag_{i}" for i in range(-1, -25, -1)]
ANGLEZ_LAGS_PAST = [f"anglez_past-lag_{i}" for i in range(1, 25)]
ENMO_LAGS_PAST = [f"enmo_past-lag_{i}" for i in range(1, 25)]
LAGS_FUTURE = [*ANGLEZ_LAGS_FUTURE, *ENMO_LAGS_FUTURE]
LAGS_PAST = [*ANGLEZ_LAGS_PAST, *ENMO_LAGS_PAST]
FEATURES = ['anglez', 'enmo', *LAGS_PAST, *LAGS_FUTURE]

LABEL = ['awake']


class CustomDataSet(Dataset):
    def __init__(self, overview, root_dir):
        self.overview = overview
        self.root_dir = root_dir

        self.cache_series_id = None
        self.cache_series_X = None
        self.cache_series_y = None
        self.counter = 0

    def __len__(self):
        return len(self.overview)

    def __getitem__(self, idx):
        series_id, index_in_series = self.overview[idx]

        if self.cache_series_id != series_id:
            path = os.path.join(self.root_dir, str(series_id.item()) + ".feather")
            series = pd.read_feather(path)

            self.cache_series_id = series_id
            self.cache_series_X = torch.from_numpy(series[FEATURES].astype('float32').to_numpy())
            self.cache_series_y = torch.from_numpy(series[LABEL].astype('int64').to_numpy()).squeeze(1)

        return self.cache_series_X[index_in_series], self.cache_series_y[index_in_series]


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
        train_root_dir = os.path.join(dirname, "../data/processed/lag-features/train")
        validation_root_dir = os.path.join(dirname, "../data/processed/lag-features/validation")

        train_overview = pd.read_parquet(os.path.join(train_root_dir, 'overview.parquet'), columns=['num_series_id', 'series_index'])
        train_overview = train_overview.astype('int64')
        validation_overview = pd.read_parquet(os.path.join(validation_root_dir, 'overview.parquet'), columns=['num_series_id', 'series_index'])
        validation_overview = validation_overview.astype('int64')

        self.train_dataset = CustomDataSet(torch.from_numpy(train_overview.values), train_root_dir)
        self.validation_dataset = CustomDataSet(torch.from_numpy(validation_overview.values), validation_root_dir)

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