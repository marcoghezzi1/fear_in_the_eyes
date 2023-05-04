from os.path import join
import scipy
import os
import numpy as np
from my_utils.loader import load_event_features
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
import numpy_indexed as npi
from sklearn.svm import LinearSVR, LinearSVC, OneClassSVM, SVR, SVC
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier, EasyEnsembleClassifier, BalancedBaggingClassifier
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_approximation import Nystroem
from scipy.stats import uniform
import pandas as pd
import re
from my_utils.plotter import build_roc_curve
import GPy
from sklearn.model_selection import KFold

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)

def load_dataset(path_ou, path_gazetime, path_pupil):
    global_data_fix = []
    global_data_sac = []

    paths_ou = sorted_nicely(os.listdir(path_ou))
    paths_gazetime = sorted_nicely(os.listdir(path_gazetime))
    paths_pupil = sorted_nicely(os.listdir(path_pupil))

    sias_df = pd.read_excel('./osf/Questionnaires.xlsx')
    sias_df = sias_df.drop(sias_df.index[24]) # Deleting three subjects because not suitable
    sias_df = sias_df.drop(sias_df.index[19]) 
    sias_df = sias_df.drop(sias_df.index[8])
    sias_df['UCS_pain_post']= pd.to_numeric(sias_df['UCS_pain_post'])
    sias_df = sias_df[sias_df['UCS_pain_post']>2] # Deleting subjects with UCS_pain_post less than 2

    sias_score = sias_df.iloc[:, 1:21]
    sias_score = sias_score.apply(pd.to_numeric)
    sias_score['SIAS05'] = [4]*43 - sias_score['SIAS05']
    sias_score['SIAS09'] = [4]*43 - sias_score['SIAS09']
    sias_score['SIAS11'] = [4]*43 - sias_score['SIAS11']
    sias_score = sias_score.sum(axis=1)
    sias_df['score'] = sias_score
    sias_df['anxiety'] = sias_score >= 30
    sias_df = sias_df[['VP', 'score', 'anxiety']]
    
    # Mapping subject ids to their label   
    map_sias = { x[1]['VP'] : x[1]['score']  for x in sias_df.iterrows() } 
    map_is_anxious = { x[1]['VP'] : x[1]['anxiety']  for x in sias_df.iterrows() }

    subs_considered = 0
    paths_ou.remove('event_features_11.npy')
    paths_ou.remove('event_features_20.npy')
    paths_ou.remove('event_features_25.npy')
    paths_ou.remove('event_features_42.npy')

    paths_gazetime.remove('gaze_data_timeLook011.asc.npy')
    paths_gazetime.remove('gaze_data_timeLook020.asc.npy')
    paths_gazetime.remove('gaze_data_timeLook025.asc.npy')
    paths_gazetime.remove('gaze_data_timeLook042.asc.npy')

    for index, file in enumerate(paths_ou):

        fix_data, sac_data, stim_fix, stim_sac = load_event_features( join(path_ou, file), 
                                                                      join(path_gazetime, paths_gazetime[index]), 
                                                                      join(path_pupil, paths_pupil[index]) )

        subject_id = int(file.split("_")[2].split(".")[0])

        label = map_sias.get(subject_id, -1)
        
        curr_label_f = np.ones([fix_data.shape[0], 1]) * label
        curr_label_s = np.ones([sac_data.shape[0], 1]) * label
        
        curr_subject_id_f = np.ones([fix_data.shape[0], 1]) * subject_id
        curr_subject_id_s = np.ones([sac_data.shape[0], 1]) * subject_id
        
        fix_data = np.hstack([curr_subject_id_f, curr_label_f, stim_fix, fix_data])
        sac_data = np.hstack([curr_subject_id_s, curr_label_s, stim_sac, sac_data])

        if label != -1: # I consider the subject only if it is in the selected 44
            global_data_fix.append(fix_data)
            global_data_sac.append(sac_data)
            subs_considered += 1
        else:
            print('Subject '+str(subject_id)+' not suitable')

    data_fix = np.vstack(global_data_fix)
    data_sac = np.vstack(global_data_sac)
    print('\nLoaded ' + str(subs_considered) + ' subjects...')
    return data_fix, data_sac

def get_features(data, config='all_features', typ='sac'):
    n_classic_features = 1 if typ=='fix' else 3
    n_pupil_features = 2
    n_tot_features = 15 if typ=='fix' else 17

    if config == 'all_features':
        return data[:, 3: ]
    elif config == 'classic_features':
        return data[:, 3 + n_tot_features - n_classic_features - n_pupil_features : 3 + n_tot_features - n_pupil_features ]
    elif config == 'pupil_features':
        return data[:, 3 + n_tot_features - n_pupil_features: ]
    elif config == 'ou_features':
        return data[:, 3: 3 + n_tot_features - n_classic_features - n_pupil_features]


# MAIN ---------------------------------------------------------------------

dataset_name = 'Reutter_OU_posterior_VI'
models_regression = [ #'GPR',
                      #SVR( C=1000, kernel='rbf', gamma=0.002),
                      RandomForestRegressor(),
                      MLPRegressor(hidden_layer_sizes=(100, 50, 25))
                    ]

directory_ou = join(join('features', dataset_name), 'train')
directory_gazetime = './osf/gazetime/'
directory_pupil = './osf/pupil_data/'

data_fix, data_sac = load_dataset(directory_ou, directory_gazetime, directory_pupil)
map_ss_sias = {} # Mapping (subject, stimulus) to Social Anxiety of the subject

X_fix = get_features( data_fix , typ='fix')
ids_f = data_fix[:, 0] # Subjects' ids (fixations)
yf = data_fix[:, 1] # Labels (fixations)
stim_f = data_fix[:, 2]  # Stimulus' ids (fixations)

X_sac = get_features( data_sac, typ='sac')
ids_s = data_sac[:, 0] # Subjects' ids (saccades)
ys = data_sac[:, 1] # Labels (saccades)
stim_s = data_sac[:, 2] # Stimulus' ids (saccades)

