import os

import pandas as pd

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    train_root_dir = os.path.join(dirname, "../../data/processed/lag-features/train")
    train_overview = pd.read_parquet(os.path.join(train_root_dir, 'overview.parquet'), columns=['num_series_id'])

    series_ids = train_overview.num_series_id.unique()

    for series_id in series_ids:
        path = os.path.join(train_root_dir, str(series_id.item()) + ".feather")
        series = troch.load(path)
        print(f"loaded {series_id}")

