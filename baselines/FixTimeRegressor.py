from datetime import datetime
import pandas as pd
import tqdm

class FixTimeRegressor():

    def __init__(self, wake_up_time, sleep_time):
        self.wake_up_time = datetime.strptime(wake_up_time, '%H:%M:%S')
        self.sleep_time = datetime.strptime(sleep_time, '%H:%M:%S')

    def fit(self, x, y):
        pass

    def predict(self, x):
        return_values = []
        with tqdm.tqdm(total=x.shape[0]) as p_bar:
            for index, row in x.iterrows():
                datetime_object = datetime.strptime(row['timestamp'][:19].replace('T', ' '), '%Y-%m-%d %H:%M:%S')
                if datetime_object.time() == self.wake_up_time.time():
                    return_values.append({'row_id': len(return_values), 'series_id': row['series_id'], 'step': row['step'], 'event': 'wakeup', 'score': 0.0})
                elif datetime_object.time() == self.sleep_time.time():
                    return_values.append({'row_id': len(return_values), 'series_id': row['series_id'], 'step': row['step'], 'event': 'onset', 'score': 0.0})

                p_bar.update(1)
        return pd.DataFrame(return_values)
    