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
from sklearn.model_selection import KFold, LeaveOneOut

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)


def train_sklearn(X, y, model, fold, config, model_type='fix'):
    from sklearn.metrics import make_scorer

    print('Regression using ', model ) 
    scorer = make_scorer(r2_score)
    pipe_reg = make_pipeline( RobustScaler(),
                              clone(model)
                            )

    pipe_reg = pipe_reg.fit(X, y)

    # if the model computes features importance
    if type(pipe_reg[1]) == RandomForestRegressor: 
        with open('./results/loo_subject/features_importance/RF_'+model_type+'_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, pipe_reg[1].feature_importances_)

    y_pred = pipe_reg.predict(X)
    mse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print('\tRMSE train: ', mse)
    print('\tMAE train: ', mae)
    print('\tR2 train: ', r2)

    return pipe_reg

def normalize_data_and_train_gpr(train_X, train_y, test_X, fold, config):
    # Normalization
    scaler = RobustScaler()
    train_X = scaler.fit_transform(train_X.copy())
    test_X = scaler.transform(test_X.copy())

    # Optimization
    kernel_fix = GPy.kern.RBF(input_dim = len(train_X[0]), ARD=True)
    #kernel_fix = GPy.kern.Matern32(input_dim = len(train_Xf[0]), variance=1.0, lengthscale=0.5, ARD=True)
    #kernel_fix = GPy.kern.Linear(input_dim = len(train_Xf[0]), variance=1.0, ARD=True)
    reg = GPy.models.SparseGPRegression(train_X, train_y.reshape(-1, 1), kernel_fix, num_inducing=100)
    reg.optimize()

    # Saving Feature importance
    with open('./results/loo_subject/features_importance/GPR_'+config+'_'+str(fold)+'.npy', 'wb') as f:
        np.save(f, reg.kern.lengthscale.values)

    y_pred, y_pred_var  = reg.predict(train_X)
    mse = mean_squared_error(train_y, y_pred, squared=False)
    mae = mean_absolute_error(train_y, y_pred)
    r2 = r2_score(train_y, y_pred)
    print('\tRMSE train: ', mse)
    print('\tMAE train: ', mae)
    print('\tR2 train: ', r2)

    # Return the trained model and normalized test_X
    return reg, test_X

def compute_pred(reg, X_test, y_test, fold, config):

    if type(reg) ==  GPy.models.SparseGPRegression:
        y_pred, y_pred_var  = reg.predict(X_test)
        #Saving subject based results
        with open('./results/loo_subject/subject_based/'+type(reg).__name__+'_y_pred_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_pred)
        with open('./results/loo_subject/subject_based/'+type(reg).__name__+'_y_pred_var_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_pred_var)
        with open('./results/loo_subject/subject_based/'+type(reg).__name__+'_y_test_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_test)

        return y_pred[0][0], y_test[0]
    else:
        y_pred = reg.predict(X_test)
        #Saving subject based results
        with open('./results/loo_subject/subject_based/'+type(reg[1]).__name__+'_y_pred_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_pred)
        with open('./results/loo_subject/subject_based/'+type(reg[1]).__name__+'_y_test_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_test)
    
        return y_pred[0], y_test[0]


def load_dataset(path_ou, path_gazetime, path_pupil):
    if os.path.exists('./pre_loaded/data_sac.npy'):
        data_sac = np.load('./pre_loaded/data_sac.npy', allow_pickle=True)
        data_fix = np.load('./pre_loaded/data_fix.npy', allow_pickle=True)
        return data_sac, data_fix

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
    
    plt.hist( list(map_sias) )

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
    # Saving npy files for a faster reloading
    with open('./pre_loaded/data_sac.npy', 'wb') as f:
        np.save(f, data_sac)
    with open('./pre_loaded/data_fix.npy', 'wb') as f:
        np.save(f, data_fix)
    print('\nLoaded ' + str(subs_considered) + ' subjects...')
    
    return data_fix, data_sac


def get_results_loo(X, y, model, config):
    
    global_y = []
    global_pred = []

    loo = LeaveOneOut()

    for fold, (train_index, test_index) in enumerate(loo.split(X)):

        print('\nFold ' + str(fold+1))

        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]
        test_y = y[test_index]
        
        if model == 'GPR':
            print('\nTraining')
            reg, test_X = normalize_data_and_train_gpr(train_X, train_y, test_X, fold=fold, config=config)
        else:
            print('\nTraining')
            reg = train_sklearn(train_X, train_y, model=model, 
                                fold=fold, config=config)

        y_pred, y_test = compute_pred(reg, test_X, test_y, fold, config)

        global_y.append(y_test)
        global_pred.append(y_pred)

    rmse = mean_squared_error(global_y, global_pred, squared=False)
    mae = mean_absolute_error(global_y, global_pred)
    r2 = r2_score(global_y, global_pred)

    print('Regression Evaluation')
    print('\tRMSE test', rmse)
    print('\tMAE test', mae)
    print('\tR2 test', r2)

    return {'rmse': rmse, 'mae': mae, 'r2': r2}

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

def fusion(X_fix, ids_f, X_sac, ids_s, map_ss_sias):
    ids_f = ids_f.astype('int')
    ids_s = ids_s.astype('int')
    unique_ids = np.unique(ids_f)

    # Compute the mean of fix and sac data for each unique id
    mean_fix = []
    mean_sac = []
    y_sias = []

    for curr_id in unique_ids:
        fix_rows = X_fix[ids_f == curr_id]
        sac_rows = X_sac[ids_s == curr_id]
        fix_mean_rows = np.mean(fix_rows, axis=0)
        sac_mean_rows = np.mean(sac_rows, axis=0)
        mean_fix.append(fix_mean_rows)
        mean_sac.append(sac_mean_rows)
        y_sias.append(map_ss_sias.get(curr_id))

    # Combine the mean data for each id
    mean_fix = np.vstack(mean_fix)
    mean_sac = np.vstack(mean_sac)

    # Concatenate the two arrays along the column axis
    return np.concatenate((mean_fix, mean_sac), axis=1), np.array(y_sias)



# MAIN ---------------------------------------------------------------------

dataset_name = 'Reutter_OU_posterior_VI'
models_regression = [ 'GPR',
                      SVR( C=1000, kernel='rbf', gamma=0.002), 
                      RandomForestRegressor(), 
                      MLPRegressor(hidden_layer_sizes=(100, 50, 25))
                    ]

directory_ou = join(join('features', dataset_name), 'train')
directory_gazetime = './osf/gazetime/'
directory_pupil = './osf/pupil_data/'

data_fix, data_sac = load_dataset(directory_ou, directory_gazetime, directory_pupil)
map_ss_sias = {} # Mapping subject to Social Anxiety of the subject

for x in data_fix[:, :3]:
    map_ss_sias[int(x[0])] = x[1]


for models, configuration in  [ (models_regression, 'all_features'), 
                                (models_regression, 'classic_features'),
                                (models_regression, 'pupil_features'), 
                                (models_regression, 'ou_features') ]:

    print('\nTraining models with ', configuration.replace('_', ' '))

    map_ss_sias = {} # Mapping subject to Social Anxiety of the subject
    for x in data_fix[:, :3]:
        map_ss_sias[int(x[0])] = x[1]

    X_fix = get_features( data_fix, configuration , typ='fix')
    ids_f = data_fix[:, 0] # Subjects' ids (fixations)

    X_sac = get_features( data_sac, configuration, typ='sac')
    ids_s = data_sac[:, 0] # Subjects' ids (saccades)


    X, y = fusion(X_fix, ids_f, X_sac, ids_s, map_ss_sias)

    print('Standard Deviation of labels', np.std(list(y)) )

    unique_f, counts_f = np.unique(ids_f, return_counts=True)
    cf = dict(zip(unique_f.astype(int), counts_f))

    unique_s, counts_s = np.unique(ids_s, return_counts=True)
    cs = dict(zip(unique_s.astype(int), counts_s))

    print('\n-------------------------------')
    print('\nFixations Counts per Subject: \n' + str(cf))
    print(' ')
    print('Saccades Counts per Subject: \n' + str(cs))
    print('\n-------------------------------')

    
    for model in models:
        cv_summary = get_results_loo(X, y, model=model, config=configuration)

