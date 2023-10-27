import pandas as pd
import numpy as np
import joblib
import gc
import torch
import os
import time
from torch.utils.data import DataLoader, TensorDataset, Dataset

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


if __name__ == "__main__":
    dirname = os.path.abspath('')
    train_root_dir = os.path.join(dirname, "../data/processed/lag-features/train")

    train_overview = pd.read_parquet(os.path.join(train_root_dir, 'overview.parquet'), columns=['num_series_id', 'series_index'])[0:1_000_000].copy()
    train_overview['num_series_id'] = train_overview['num_series_id'].astype('int64')
    train_overview['series_index'] = train_overview['series_index'].astype('int64')

    train_dataset = CustomDataSet(torch.from_numpy(train_overview.values), train_root_dir)
    train_data_loader = DataLoader(train_dataset, batch_size=200, num_workers=4)

    print('Start')
    start_time = time.time()

    counter = 0
    for i, batch in enumerate(train_data_loader):
        counter += 1

    print(counter)

    print(f'Took {time.time() - start_time} seconds')

