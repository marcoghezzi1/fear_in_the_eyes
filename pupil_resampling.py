import pandas as pd
import os
from scipy import signal

# tmp absolute path
dir_pupil = '/Users/marcoghezzi/PycharmProjects/pythonProject/fear_gen/data/tmp_data/tmp_pupil'

for file in sorted(os.listdir(dir_pupil)):

    complete_path = dir_pupil + '/' + file
    df_ = pd.read_csv(complete_path).drop(columns='trial')

    resampled = []
    for i, rows in df_.iterrows():
        values = rows.to_numpy()
        resampled_values = signal.resample(values, 1000)
        resampled.append(resampled_values)
    df_resampled = pd.DataFrame(resampled)
    # df_resampled.to_csv('output path del server')

