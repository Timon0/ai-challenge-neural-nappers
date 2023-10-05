from datetime import datetime
import pandas as pd
import tqdm

class FixTimeRegressor():

    def __init__(self, wake_up_time, sleep_time):
        self.wake_up_time = wake_up_time
        self.sleep_time = sleep_time

    def fit(self, x, y):
        pass

    def predict(self, x):
        return_values = []
        # x['timestamp'] = pd.to_datetime(x['timestamp'][:19].replace('T', ' '))
        x['time'] = x['timestamp'].str[11:19]
        need_rows = x[x['time'] == self.wake_up_time]
        need_rows_2 = x[x['time'] == self.sleep_time]
        dataframe = pd.concat([need_rows, need_rows_2], axis=0)
        dataframe = dataframe.sort_values(['series_id', 'timestamp'])
        with tqdm.tqdm(total=dataframe.shape[0]) as p_bar:
            for index, row in dataframe.iterrows():
                if row['time'] == self.wake_up_time:
                    return_values.append({'row_id': len(return_values), 'series_id': row['series_id'], 'step': row['step'], 'event': 'wakeup', 'score': 0.5, 'timestamp': row['timestamp']})
                elif row['time'] == self.sleep_time:
                    return_values.append({'row_id': len(return_values), 'series_id': row['series_id'], 'step': row['step'], 'event': 'onset', 'score': 0.5, 'timestamp': row['timestamp']})

                p_bar.update(1)
        return pd.DataFrame(return_values)
    