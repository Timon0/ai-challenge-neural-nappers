import os

import lightning as L
import pandas as pd
import torch
import time
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ['anglez', 'enmo',
            'hour',
            'anglez_abs', 'anglez_diff', 'enmo_diff', 'anglez_x_enmo',
            'anglez_rolling_mean', 'enmo_rolling_mean', 'anglez_rolling_max', 'enmo_rolling_max', 'anglez_rolling_min',
            'anglez_rolling_std', 'enmo_rolling_std']

LABEL = ['awake']

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    df_train = pd.read_parquet(os.path.join(dirname, '../data/processed/validation_series_split.parquet'))[0:100_000]

    # split into x and y
    X_train = df_train[FEATURES].astype('float32')
    y_train = df_train[LABEL].astype('int64')

    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # convert to tensor
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train.to_numpy()).squeeze(1)

    # create dataset
    train_dataset = TensorDataset(X_train, y_train)


    train_data_loader = DataLoader(train_dataset, batch_size=200)

    print('Start')
    start_time = time.time()

    counter = 0
    for i, batch in enumerate(train_data_loader):
        counter += 1

    print(counter)

    print(f'Took {time.time() - start_time} seconds')
