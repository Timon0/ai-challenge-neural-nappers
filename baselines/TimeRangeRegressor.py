import pandas as pd
from datetime import datetime, timedelta
import random
import tqdm

class TimeRangeRegressor:

  def __init__(self, wake_up_range, sleep_range, seed=42):
    random.seed(seed)
    self.wake_up_range = []
    for time in self.datetime_range(datetime.strptime(wake_up_range[0], '%H:%M:%S'), datetime.strptime(wake_up_range[1], '%H:%M:%S'), timedelta(minutes=5)):
      self.wake_up_range.append(time)
    self.sleep_range = []
    for time in self.datetime_range(datetime.strptime(sleep_range[0], '%H:%M:%S'), datetime.strptime(sleep_range[1], '%H:%M:%S'), timedelta(minutes=5)):
      self.sleep_range.append(time)
  
  def datetime_range(self, start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

  def fit(self, x, y):
    pass

  def predict(self, x):
    x['time'] = x['timestamp'].str[11:19]
    needed_rows = x[(x['time'] >= str(self.wake_up_range[0].time())) & (x['time'] <= str(self.wake_up_range[-1].time()))]
    needed_rows2 = x[(x['time'] >= str(self.sleep_range[0].time())) & (x['time'] <= str(self.sleep_range[-1].time()))]
    df = pd.concat([needed_rows, needed_rows2], axis=0)
    first_date = df.iloc[0]
    
    current_wake_up = first_date['timestamp'][0:10] + 'T' + str(random.choice(self.wake_up_range))[11:19]
    current_sleep = first_date['timestamp'][0:10] + 'T' + str(random.choice(self.sleep_range))[11:19]
    return_values = []
    last_series_id = first_date['series_id']
    with tqdm.tqdm(total=df.shape[0]) as p_bar:
      for index, row in df.iterrows():
          
          if str(row['timestamp'])[:19] == current_wake_up:
              return_values.append({'row_id': len(return_values), 'series_id': row['series_id'], 'step': row['step'], 'event': 'wakeup', 'score': 0.5, 'timestamp': row['timestamp']})
              datetime_object = datetime.strptime(row['timestamp'][:10], '%Y-%m-%d') + timedelta(days=1)
              current_wake_up = str(datetime_object.date()) + 'T' + str(random.choice(self.wake_up_range))[11:19]
              
          elif str(row['timestamp'])[:19] == current_sleep:
              return_values.append({'row_id': len(return_values), 'series_id': row['series_id'], 'step': row['step'], 'event': 'onset', 'score': 0.5, 'timestamp': row['timestamp']})
              datetime_object = datetime.strptime(row['timestamp'][:10], '%Y-%m-%d') + timedelta(days=1)
              current_sleep = str(datetime_object.date()) + 'T' + str(random.choice(self.sleep_range))[11:19]

          if last_series_id != row['series_id']:
            datetime_object = datetime.strptime(row['timestamp'][:10], '%Y-%m-%d')
            current_sleep = str(datetime_object.date()) + 'T' + str(random.choice(self.sleep_range))[11:19]
            current_wake_up = str(datetime_object.date()) + 'T' + str(random.choice(self.wake_up_range))[11:19]
            last_series_id = row['series_id']
             
          p_bar.update(1)
    return pd.DataFrame(return_values)