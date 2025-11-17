import pandas as pd
from pathlib import Path

ratingcsv_path = '/Users/gioli/projects/stateswitch/data/rec/svf_annotated/'

for csv_file in Path(ratingcsv_path).glob("*.csv"):
    df = pd.read_csv(csv_file)
    df["switch_flag"] = 0
    new_csv_file = ratingcsv_path + csv_file.name.replace('desc-wordtimestampsrated.csv', 'desc-wordtimestampswithswitch.csv')
    df.to_csv(new_csv_file, index=False)
