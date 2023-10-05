import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES = ['anglez', 'enmo',
            'hour',
            'anglez_abs', 'anglez_diff', 'enmo_diff', 'anglez_x_enmo',
            'anglez_rolling_mean', 'enmo_rolling_mean', 'anglez_rolling_max', 'enmo_rolling_max', 'anglez_rolling_min',
            'anglez_rolling_std', 'enmo_rolling_std']
TARGET = ['awake', 'sleep']

class MyDataset(Dataset):

    def __init__(self, df):
        print(df[FEATURES].shape)

        self.X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
        self.y = torch.tensor(df[TARGET].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]