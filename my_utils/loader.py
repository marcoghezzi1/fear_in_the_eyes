import numpy as np
import scipy.io as sio
from os.path import join
from os import listdir, remove
from zipfile import ZipFile
import pandas as pd
from my_utils.gaze import dva2pixels
import re

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)


def load_event_features(ou_file, gazetime_file, pupil_file ):
    
    ou_and_classic_features = np.load(ou_file, allow_pickle=True)
    timestamps = np.load(gazetime_file, allow_pickle=True)
    pupil_dilation = pd.read_csv(pupil_file)
    
    pupil_dilation.head()
    pupil_dilation_ts = list()
    
    n_trials = len(ou_and_classic_features)

    feat_fixs = []
    feat_sacs = []
    stim_fix = []
    stim_sac = []
    pupil_fix = []
    pupil_sac = []

    c_fix = 0
    c_sac = 0
    
    for t in range(n_trials):
        curr_pupil = list()
        curr_data_dict = ou_and_classic_features[t]
        
        try:
            feat_fix = curr_data_dict['feat_fix']
            feat_sac = curr_data_dict['sacc_fix']
            
            first_ts_fix = timestamps[t]['fix'][0][0][2] 
            first_ts_sac = timestamps[t]['sac'][0][0][2]
            
            first_ts = min(first_ts_fix, first_ts_sac) # Computing the first timestamp of the trial
        
            for index, x in enumerate(pupil_dilation.iloc[t]):
                curr_pupil.append((first_ts + index, x)) 
            
            if feat_fix is None or feat_sac is None:
                continue
            
        except:
            continue
        
        feat_fix_new = []
        for index, curr_fix in enumerate(timestamps[t]['fix']):
            start_ts = curr_fix[0][2]
            end_ts = curr_fix[-1][2]
            
            pupil_list = [] # Computing the pupil_list for the current fixation
            for pupil_data in curr_pupil:
                if start_ts <= pupil_data[0] < end_ts:
                    pupil_list.append(pupil_data[1])
            
            if len(pupil_list) > 0:
                curr_mean = np.mean(pupil_list) # Computing mean and std for the 
                curr_std = np.std(pupil_list)
                pupil_fix.append((curr_mean, curr_std))
                feat_fix_new.append( list(feat_fix[index]) + [curr_mean, curr_std] )
            else:
                c_fix += 1
        
        feat_sac_new = []
        for index, curr_sac in enumerate(timestamps[t]['sac']):
            start_ts = curr_sac[0][2]
            end_ts = curr_sac[-1][2]
            
            pupil_list = []
            for pupil_data in curr_pupil:
                if start_ts <= pupil_data[0] < end_ts:
                    pupil_list.append(pupil_data[1])
            
            if len(pupil_list) > 0:
                curr_mean = np.mean(pupil_list)
                curr_std = np.std(pupil_list)
                pupil_sac.append((curr_mean, curr_std))
                feat_sac_new.append( list(feat_sac[index]) + [curr_mean, curr_std] )
            else:
                c_sac += 1
        
        feat_fixs.append(feat_fix_new)
        stim_fix.append(np.repeat(curr_data_dict['stimulus'], len(feat_fix_new))[:,np.newaxis])
        feat_sacs.append(feat_sac_new)
        stim_sac.append(np.repeat(curr_data_dict['stimulus'], len(feat_sac_new))[:,np.newaxis])
        pupil_dilation_ts.append(curr_pupil)

    feat_fixs = np.vstack(feat_fixs)
    feat_sacs = np.vstack(feat_sacs)
    stim_fix = np.vstack(stim_fix)
    stim_sac = np.vstack(stim_sac)

    return feat_fixs, feat_sacs, stim_fix, stim_sac


def load_reutter(path):
    scanpath = []
    paths = sorted_nicely(listdir(path))

    for subject in paths:
        sub_scan = load_reutter_sub(path + subject)
        scanpath.append(sub_scan)
    parameters = {
        'distance': 0.53,
        'width': 0.531,
        'height': 0.298,
        'x_res': 1920,
        'y_res': 1080,
        'fs': 1000.}
    scanpath = np.asarray(scanpath)
    return scanpath, parameters


def load_reutter_sub(sub_path):
    # read the whole file into variable `events` (list with one entry per line)
    with open(sub_path) as f:
        events = f.readlines()

    events = [event for event in events if 'SFIX' and 'SSACC' and 'MSG' and 'ESACC' not in event]
    trial_start_indices = np.where(["START" in ev for ev in events])[0]
    trial_end_indices = np.where(["END" in ev for ev in events])[0]

    sub_scan = []
    for i in range(len(trial_start_indices)):  # for each trial
        start = trial_start_indices[i] + 7
        if i != len(trial_start_indices) - 1:
            end = trial_start_indices[i + 1] - 9
        else:
            end = trial_end_indices[-1] - 4

        current_trial = events[ start : end ]  # Removing starting and ending info
        trial_coor = []
        for event in current_trial:

            if 'EFIX' not in event:
                try:
                    l = event.split('\t')
                    ts = float( l[0][-8:] )
                    x = float( l[1] )
                    y = float( l[2] )
                    trial_coor.append([x, y, ts])
                except:
                    pass
            else:
                l = event.split('\t')
                ts = float( l[0].split('   ')[1] )
                x = float( l[4] )  # in case of fixation end x and y coordinates are in positions 4,5
                y = float( l[5] )
                trial_coor.append([x, y, ts])


        trial_coor = np.asarray(trial_coor)
        sub_scan.append(trial_coor)
    return np.asarray(sub_scan)


def load_dataset(name, path, round='Round_9', session='S1', task='Video_1'):
    return load_reutter(path)


if __name__ == '__main__':
    data_cerf = load_reutter('../datasets/Reutter')
