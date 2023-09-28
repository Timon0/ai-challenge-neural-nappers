# source: https://www.kaggle.com/code/metric/event-detection-ap/comments#2432522

import sys
sys.path.append('../')

from scoring.event_detection_ap import score
import pandas as pd

tolerances = {
    "onset" : [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
}

column_names = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}


def competition_score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    return score(solution, submission, tolerances,**column_names)