from typing import Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from mlp import Model
from data import MyDataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Params
RANDOM_STATE = 42
features = ['anglez', 'enmo',
            'hour',
            'anglez_abs', 'anglez_diff', 'enmo_diff', 'anglez_x_enmo',
            'anglez_rolling_mean', 'enmo_rolling_mean', 'anglez_rolling_max', 'enmo_rolling_max', 'anglez_rolling_min',
            'anglez_rolling_std', 'enmo_rolling_std']
target_column = ['awake', 'sleep']

# Hyperparams
## None yet

# read data
data = pd.read_parquet('../../data/processed/train_with_features_lightweight_10.parquet')

data['sleep'] = data.awake.replace({0:1, 1:0})

# train/test split
def group_test_train_split(samples: pd.DataFrame, group: str, test_size, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = samples[group].drop_duplicates()
    groups_train, groups_test = train_test_split(groups, test_size=test_size, random_state=random_state)

    samples_test = samples.loc[lambda d: d[group].isin(groups_test)]
    samples_train = samples.loc[lambda d: d[group].isin(groups_train)]

    return samples_test, samples_train

test, train = group_test_train_split(data, 'series_id', test_size=0.2, random_state=RANDOM_STATE)

X_train = train[features]
y_train = train[target_column]
X_test = test[features]
y_test = test[target_column]


# setup model
model = Model(n_features=len(features))

# setup dataloader
dataset = MyDataset(data)
dataloader = DataLoader(dataset)

# train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')