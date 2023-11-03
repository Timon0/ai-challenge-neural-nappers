import os

import lightning as L
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset

LAGS_FUTURE = [f"t_lag_{i}" for i in range(-1, -25, -1)]
LAGS_PAST = [f"t_lag_{i}" for i in range(1, 25)]
FEATURES = ['t_0', *LAGS_PAST, *LAGS_FUTURE]

LABEL = ['awake']

class CustomTimeSeriesDataSet(Dataset):
    def __init__(self, overview, root_dir):
        self.overview = overview
        self.root_dir = root_dir
        
        self.cache_series_id = None
        self.cache_series_X = None
        self.cache_series_y = None

    def __len__(self):
        return len(self.overview)

    def __getitem__(self, idx):
        series_id, label, index_in_series = self.overview[idx]

        if self.cache_series_id != series_id:
            self.cache_series_id = series_id

            path = os.path.join(self.root_dir, str(series_id.item()) + ".pt")
            self.cache_series_X = torch.load(path)

        return self.cache_series_X[index_in_series], label

class CustomTimeSeriesDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 100, seq_len: int = 24):
        super().__init__()
        self.num_classes = 2
        self.features = FEATURES
        self.validation_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.seq_len = seq_len

    def prepare_data(self):
        # load data
        dirname = os.path.dirname(__file__)
        train_root_dir = os.path.join(dirname, "../data/transformer/train")
        validation_root_dir = os.path.join(dirname, "../data/transformer/validation")

        train_overview = pd.read_parquet(os.path.join(train_root_dir, 'overview.parquet'), columns=['num_series_id', 'awake', 'series_index'])
        train_overview = train_overview.astype('int64')
        validation_overview = pd.read_parquet(os.path.join(validation_root_dir, 'overview.parquet'), columns=['num_series_id', 'awake', 'series_index'])
        validation_overview = validation_overview.astype('int64')

        self.train_dataset = CustomTimeSeriesDataSet(torch.from_numpy(train_overview.values), train_root_dir)
        self.validation_dataset = CustomTimeSeriesDataSet(torch.from_numpy(validation_overview.values), validation_root_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def get_num_features(self):
        return len(self.features)

    def get_num_classes(self):
        return self.num_classes