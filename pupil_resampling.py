import pandas as pd
import os
from scipy import signal
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-S", "--server", action="store_true", help='path from server')

args = parser.parse_args()
config = vars(args)

check = config['server']

# tmp absolute path
if check:
    dir_pupil = '/var/data/FEAR_GEN/tmp_data/tmp_pupil'
    data_path = '/var/data/FEAR_GEN/'
else:
    print("Enter your local directory for pupil csv: ")
    dir_pupil = input()
    data_path = '.'


for file in sorted(os.listdir(dir_pupil)):

    complete_path = dir_pupil + '/' + file
    df_ = pd.read_csv(complete_path).drop(columns='trial')

    resampled = []
    for i, rows in df_.iterrows():
        values = rows.to_numpy()
        resampled_values = signal.resample(values, 1000)
        resampled.append(resampled_values)
    df_resampled = pd.DataFrame(resampled)
    if check:
        df_resampled.to_csv(data_path+'output/resampled/resampled_'+file)

