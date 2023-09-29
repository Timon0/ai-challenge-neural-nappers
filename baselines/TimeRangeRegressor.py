import pandas as pd
from datetime import datetime, timedelta
import random
import tqdm

class TimeRangeRegressor:

  def daterange(self, start, end):
    print((end - start))
    for n in range(int((end - start))):
      yield start + timedelta(n)

  def datetime_range(self, start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

  def __init__(self, wake_up_range, sleep_range):
    self.wake_up_range = []
    for time in self.datetime_range(datetime.strptime(wake_up_range[0], '%H:%M:%S'), datetime.strptime(wake_up_range[1], '%H:%M:%S'), timedelta(minutes=5)):
      self.wake_up_range.append(time)
    self.sleep_range = []
    for time in self.datetime_range(datetime.strptime(sleep_range[0], '%H:%M:%S'), datetime.strptime(sleep_range[1], '%H:%M:%S'), timedelta(minutes=5)):
      self.sleep_range.append(time)
  
  def fit(self, x, y):
    pass

  def predict(self, x):
    x['time'] = x['timestamp'].str[11:19]
    needed_rows = x[(x['time'] >= str(self.wake_up_range[0].time())) & (x['time'] <= str(self.wake_up_range[-1].time()))]
    needed_rows2 = x[(x['time'] >= str(self.sleep_range[0].time())) & (x['time'] <= str(self.sleep_range[-1].time()))]
    df = pd.concat([needed_rows, needed_rows2], axis=0)

    current_wake_up = random.choice(self.wake_up_range)
    current_sleep = random.choice(self.sleep_range)
    return_values = []
    with tqdm.tqdm(total=df.shape[0]) as p_bar:
      for index, row in df.iterrows():
          if row['time'] == str(current_wake_up.time()):
              return_values.append({'row_id': len(return_values), 'series_id': row['series_id'], 'step': row['step'], 'event': 'wakeup', 'score': 0.5})
              current_wake_up = random.choice(self.wake_up_range)
          elif row['time'] == str(current_sleep.time()):
              return_values.append({'row_id': len(return_values), 'series_id': row['series_id'], 'step': row['step'], 'event': 'onset', 'score': 0.5})
              current_sleep = random.choice(self.sleep_range)

          p_bar.update(1)
    return pd.DataFrame(return_values)