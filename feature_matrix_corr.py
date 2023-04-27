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
        with open('./results/features_importance/RF_'+model_type+'_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, pipe_reg[1].feature_importances_)

    y_pred = pipe_reg.predict(X)
    mse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print('\tRMSE train: ', mse)
    print('\tMAE train: ', mae)
    print('\tR2 train: ', r2)

    return pipe_reg

def normalize_data_and_train_gpr(train_X, train_y, test_X, fold, config, model_type):
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
    with open('./results/features_importance/GPR_'+model_type+'_'+config+'_'+str(fold)+'.npy', 'wb') as f:
        np.save(f, reg.kern.lengthscale.values)

    # Return the normalized test_X
    return reg, test_X

def evaluate(reg_fix, reg_sac, X_fix_test, y_f_test, stim_f_test, sub_f_test, X_sac_test, y_s_test, stim_s_test, sub_s_test, fold, config):

    metrics_regression = {}

    #Fixations -------
    ss = np.zeros_like(sub_f_test).astype('str')
    for i in range(len(sub_f_test)):
        ss[i] = str(int(sub_f_test[i])) + '-' + str(int(stim_f_test[i]))

    if type(reg_fix) ==  GPy.models.SparseGPRegression:
        ppred_fix, ppred_fix_var  = reg_fix.predict(X_fix_test)
        #Saving datapoint based results
        with open('./results/datapoint_based/'+type(reg_fix).__name__+'_ppred_fix_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, ppred_fix)
        with open('./results/datapoint_based/'+type(reg_fix).__name__+'_ppred_fix_var_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, ppred_fix_var)
        with open('./results/datapoint_based/'+type(reg_fix).__name__+'_y_fix_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_f_test)
    else:
        ppred_fix = reg_fix.predict(X_fix_test)
        #Saving datapoint based results
        with open('./results/datapoint_based/'+type(reg_fix[1]).__name__+'_ppred_fix_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, ppred_fix)
        with open('./results/datapoint_based/'+type(reg_fix[1]).__name__+'_y_fix_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_f_test)

    key_fix, ppred_fix_comb = npi.group_by(ss).mean(ppred_fix)
    
    #Saccades -------
    ss = np.zeros_like(sub_s_test).astype('str')
    for i in range(len(sub_s_test)):
        ss[i] = str(int(sub_s_test[i])) + '-' + str(int(stim_s_test[i]))

    if type(reg_sac) ==  GPy.models.SparseGPRegression:
        ppred_sac, ppred_sac_var = reg_sac.predict(X_sac_test) # Gaussian processes return both prediction and uncertainty
        #Saving datapoint based results
        with open('./results/datapoint_based/'+type(reg_sac).__name__+'_ppred_sac_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, ppred_sac)
        with open('./results/datapoint_based/'+type(reg_sac).__name__+'_ppred_sac_var_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, ppred_sac_var)
        with open('./results/datapoint_based/'+type(reg_sac).__name__+'_y_sac_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_s_test)
    else:
        ppred_sac = reg_sac.predict(X_sac_test) # Gaussian processes return both prediction and uncertainty
        #Saving datapoint based results
        with open('./results/datapoint_based/'+type(reg_sac[1]).__name__+'_ppred_sac_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, ppred_sac)
        with open('./results/datapoint_based/'+type(reg_sac[1]).__name__+'_y_sac_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_s_test)
    
    key_sac, ppred_sac_comb = npi.group_by(ss).mean(ppred_sac)

    # Extracting labels
    y_test = np.zeros(key_fix.shape)
    for i,k in enumerate(key_fix):
        subject = int(k.split('-')[0])
        stimulus = int(k.split('-')[1])
        y_test[i] = map_ss_sias[(subject, stimulus)]

    #Fusion --------
    y_pred = (np.array(ppred_fix_comb) + np.array(ppred_sac_comb)) / 2.
    
    if type(reg_sac) ==  GPy.models.SparseGPRegression:
        #Saving trial based results
        with open('./results/trial_based/'+type(reg_sac).__name__+'_y_pred_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_pred)
        with open('./results/trial_based/'+type(reg_sac).__name__+'_y_test_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_test)
    else:
        #Saving trial based results
        with open('./results/trial_based/'+type(reg_sac[1]).__name__+'_y_pred_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_pred)
        with open('./results/trial_based/'+type(reg_sac[1]).__name__+'_y_test_'+config+'_'+str(fold)+'.npy', 'wb') as f:
            np.save(f, y_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Regression Evaluation')
    print('\tRMSE test', rmse)
    print('\tMAE test', mae)
    print('\tR2 test', r2)

    metrics_regression['rmse'] = rmse
    metrics_regression['mae'] = mae
    metrics_regression['r2'] = r2

    return metrics_regression


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

def get_CV_splits(stim_f, ids_f, k):
    kf = KFold(n_splits=k)
    subs_splits = []
    sub_ids = np.unique(ids_f)
    for s in sub_ids:
        curr_stims = np.unique(stim_f[ids_f==s])[:,np.newaxis]
        subs_splits.append(kf.split(curr_stims))
    return subs_splits, sub_ids

def get_results_kfold(X_fix, ids_f, yf, stim_f, X_sac, ids_s, ys, stim_s, k, model, config):
    sub_splits_gen, sub_ids = get_CV_splits(stim_f, ids_f, k=k)
    
    sub_splits = {}
    for i,ss in enumerate(sub_splits_gen):
        curr_splits = []
        for train_index, test_index in ss:
            curr_splits.append((train_index, test_index))
        sub_splits[sub_ids[i]] = curr_splits
    
    cv_metrics = []

    for fold in range(k):
        print('\nFold ' + str(fold+1) + ' of ' + str(k))
        train_Xf = []
        train_yf = []
        train_Xs = []
        train_ys = []
        test_Xf = []
        test_yf = []
        test_Xs = []
        test_ys = []

        train_ids_f = []
        train_stf = []
        train_ids_s = []
        train_sts = []
        test_ids_f = []
        test_stf = []
        test_ids_s = []
        test_sts = []

        for s in sub_splits.keys():
            curr_Xf = X_fix[ids_f==s,:]
            curr_stf = stim_f[ids_f==s]
            curr_yf = yf[ids_f==s]

            curr_Xs = X_sac[ids_s==s,:]
            curr_sts = stim_s[ids_s==s]
            curr_ys = ys[ids_s==s]            

            train_index = sub_splits[s][fold][0]
            test_index = sub_splits[s][fold][1]
            for ti in train_index:
                train_Xf.append(curr_Xf[curr_stf==ti])
                train_stf.append(curr_stf[curr_stf==ti])
                train_yf.append(curr_yf[curr_stf==ti])
                train_ids_f.append(np.repeat(s, len(train_stf[-1]))) # train subject ids fixations
                
                train_Xs.append(curr_Xs[curr_sts==ti])
                train_sts.append(curr_sts[curr_sts==ti])
                train_ys.append(curr_ys[curr_sts==ti])
                train_ids_s.append(np.repeat(s, len(train_sts[-1]))) # train subject ids saccades

            for ti in test_index:
                test_Xf.append(curr_Xf[curr_stf==ti])
                test_stf.append(curr_stf[curr_stf==ti])
                test_yf.append(curr_yf[curr_stf==ti])
                test_ids_f.append(np.repeat(s, len(test_stf[-1]))) # test subject ids fixations

                test_Xs.append(curr_Xs[curr_sts==ti])
                test_sts.append(curr_sts[curr_sts==ti])
                test_ys.append(curr_ys[curr_sts==ti])
                test_ids_s.append(np.repeat(s, len(test_sts[-1]))) # test subject ids saccades

        train_Xf = np.vstack(train_Xf)
        train_yf = np.concatenate(train_yf)
        train_stf = np.concatenate(train_stf)
        train_ids_f = np.concatenate(train_ids_f)
        train_Xs = np.vstack(train_Xs)
        train_ys = np.concatenate(train_ys)
        train_sts = np.concatenate(train_sts)
        train_ids_s = np.concatenate(train_ids_s)

        test_Xf = np.vstack(test_Xf)
        test_yf = np.concatenate(test_yf)
        test_stf = np.concatenate(test_stf)
        test_ids_f = np.concatenate(test_ids_f)
        test_Xs = np.vstack(test_Xs)
        test_ys = np.concatenate(test_ys)
        test_sts = np.concatenate(test_sts)
        test_ids_s = np.concatenate(test_ids_s)

        if model == 'GPR':
            print('\nTraining Fixations')
            reg_fix, test_Xf = normalize_data_and_train_gpr(train_Xf, train_yf, test_Xf, fold=fold, config=config, model_type='fix')

            print('Training Saccades')
            reg_sac, test_Xs = normalize_data_and_train_gpr(train_Xs, train_ys, test_Xs, fold=fold, config=config, model_type='sac')
            
        else:
            print('\nTraining Fixations')
            reg_fix = train_sklearn(train_Xf, train_yf, model=model, 
                                    fold=fold, config=config, model_type='fix')
            
            print('Training Saccades')
            reg_sac = train_sklearn(train_Xs, train_ys, model=model, 
                                    fold=fold, config=config, model_type='sac')

        current_fold_metrics = evaluate(reg_fix, reg_sac,
                                        test_Xf, test_yf, test_stf, test_ids_f,
                                        test_Xs, test_ys, test_sts, test_ids_s, fold, config)

        cv_metrics.append(current_fold_metrics)

    #Returning means and stds of the metrics
    rmses = [fold['rmse'] for fold in cv_metrics]
    maes = [fold['mae'] for fold in cv_metrics]
    r2s = [fold['r2'] for fold in cv_metrics]
    return {'rmse_mean': np.mean(rmses),'rmse_std': np.std(rmses),'mae_mean': np.mean(maes),
            'mae_std': np.std(maes), 'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s) }

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

for x in data_fix[:, :3]:

    map_ss_sias[(int(x[0]), int(x[2]))] = x[1]

    map_ss_sias = {} # Mapping (subject, stimulus) to Social Anxiety of the subject
    for x in data_fix[:, :3]:
        map_ss_sias[(int(x[0]), int(x[2]))] = x[1]

    X_fix = get_features( data_fix , typ='fix')
    ids_f = data_fix[:, 0] # Subjects' ids (fixations)
    yf = data_fix[:, 1] # Labels (fixations)
    stim_f = data_fix[:, 2]  # Stimulus' ids (fixations)

    X_sac = get_features( data_sac, typ='sac')
    ids_s = data_sac[:, 0] # Subjects' ids (saccades)
    ys = data_sac[:, 1] # Labels (saccades)
    stim_s = data_sac[:, 2] # Stimulus' ids (saccades)


    print('Standard Deviation of labels', np.std(list(yf) + list(ys)) )

    n_sub_f = len(np.unique(ids_f))
    n_sub_s = len(np.unique(ids_s))
    assert n_sub_f == n_sub_s

    unique_f, counts_f = np.unique(ids_f, return_counts=True)
    cf = dict(zip(unique_f.astype(int), counts_f))

    unique_s, counts_s = np.unique(ids_s, return_counts=True)
    cs = dict(zip(unique_s.astype(int), counts_s))

    print('\n-------------------------------')
    print('\nFixations Counts per Subject: \n' + str(cf))
    print(' ')
    print('Saccades Counts per Subject: \n' + str(cs))
    print('\n-------------------------------')

    print('\n-----------corr matrix------')
    print(np.shape(X_fix))
    print(type(X_fix))
