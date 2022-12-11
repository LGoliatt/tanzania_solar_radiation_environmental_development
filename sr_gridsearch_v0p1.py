#!/usr/bin/python
# -*- coding: utf-8 -*-    

import numpy as np
import pandas as pd
import time as t
import datetime as dt
import warnings
import sys
# import glob as gl
import pylab as pl
import os
# import re
# import getopt
from sklearn.model_selection import (GridSearchCV, 
                                     KFold, 
                                     )
from sklearn.metrics import (r2_score,
                             )

from sklearn.svm import (SVR,)
from sklearn.linear_model import (
                                  Ridge,
                                  LinearRegression,
                                  )

from sklearn.neural_network import MLPRegressor
from pyearth import Earth as MARS

from xgboost import  XGBRegressor

from read_data_tanzania import *


from util.ELM import  ELMRegressor
from scipy import stats
from hydroeval import (kge,
                       nse)

warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" 

tStart = t.time()

#%%----------------------------------------------------------------------------
pd.options.display.float_format = '{:.3f}'.format

program_name = sys.argv[0]
arguments    = sys.argv[1:]
count        = len(arguments)

if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0
else:
    run0, n_runs = 0, 30
    
#%%----------------------------------------------------------------------------   
def accuracy_log(y_true, y_pred):
    y_true = np.abs(np.array(y_true))
    y_pred = np.abs(np.array(y_pred))
    
    return (np.abs(np.log10(y_true / y_pred)) < 0.3).sum() / len(y_true) * 100

def rms(y_true, y_pred):
    y_true = np.abs(np.array(y_true))
    y_pred = np.abs(np.array(y_pred))
    
    return ((np.log10(y_pred / y_true) ** 2).sum() / len(y_true)) ** 0.5

#------------------------------------------------------------------------------   
def lhsu(xmin,xmax,nsample):
   nvar = len(xmin)
   ran  = np.random.rand(nsample,nvar)
   s    = np.zeros((nsample,nvar))
   
   for j in range(nvar):
       idx = np.random.permutation(nsample)
       P   = (idx.T-ran[:,j])/nsample
       s[:,j] = xmin[j] + P * (xmax[j] - xmin[j]);
       
   return s

#------------------------------------------------------------------------------   
def RMSE(y, y_pred):
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    error = y -  y_pred    
    return np.sqrt(np.mean(np.power(error, 2)))

#------------------------------------------------------------------------------   
def RRMSE(y, y_pred):
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    return RMSE(y, y_pred)*100/np.mean(y)

#------------------------------------------------------------------------------   
def MAPE(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  return np.mean(np.abs(y_pred - y_true)/np.abs(y_true))*100

basename='Join Data Basis'
basename='sr_gs_'


datasets = []
for i in range(1,64) :
    datasets.append(read_data_tanzania(station='daressalaam', period='daily', model=i))
    
#%%----------------------------------------------------------------------------   

pd.options.display.float_format = '{:.3f}'.format
n_splits    = 5
scoring     = 'neg_root_mean_squared_error'
for run in range(run0, n_runs):
    random_seed=run*10
    
    for dataset in datasets:#[:1]:
        # dr=dataset['name'].replace(' ','_').replace("'","").lower()
        dr = basename.replace(' ','_').replace("'","").lower()
        path='./pkl_'+dr+f'/run_{n_runs}/'
        path='./pkl_'+dr+f'/'
        os.system('mkdir  -p '+path)
        
        for tk, tn in enumerate(dataset['target_names']):
            print (tk, tn)
            dataset_name = dataset['name']+'-'+tn
            target                          = dataset['target_names'][tk]
            y_train, y_test                 = dataset['y_train'][tk], dataset['y_test'][tk]
            X_train, X_test                 = dataset['X_train'], dataset['X_test']
            n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
            task, normalize                 = dataset['task'], dataset['normalize']
            n_samples_test                  = len(y_test)
            
            s = ''+'\n'
            s += '='*80+'\n'
            s += 'Dataset                    : '+dataset_name+' -- '+target+'\n'
            s += 'Output                     : '+tn+'\n'
            s += 'Number of training samples : '+str(n_samples_train) +'\n'
            s += 'Number of testing  samples : '+str(n_samples_test) +'\n'
            s += 'Number of features         : '+str(n_features)+'\n'
            s += 'Normalization              : '+str(normalize)+'\n'
            s += 'Task                       : '+str(dataset['task'])+'\n'
            s += '='*80
            s += '\n'            
            feature_names = dataset['feature_names']
            print(s)    
            #------------------------------------------------------------------
            args = (X_train,
                    y_train,
                    X_test,
                    y_test,
                    'eval',
                    task,
                    n_splits, 
                    int(random_seed),
                    scoring,
                    target, 
                    n_samples_train,
                    n_samples_test,
                    n_features)
            #------------------------------------------------------------------         
            
            lr = LinearRegression()
            svr_lin = SVR(kernel = 'linear')
            
            params_str = {'lasso__alpha': [0.1, 1.0, 10.0],
                          'ridge__alpha': [0.1, 1.0, 10.0],
                          'svr__C': [0.1, 1.0, 10.0],
                          }

            cv = n_splits
            
            ridgecv = GridSearchCV(estimator  = Ridge(random_state = random_seed), 
                                   param_grid = {'alpha':[0., 0.1, 0.3, 0.5, 1] + [2, 5, 10, 50, 100, 500, 1000]}, 
                                   cv         = cv,
                                   scoring    = 'neg_root_mean_squared_error',
                                   n_jobs     = -1,
                                   refit      = True)
            
           
                       
            svr = GridSearchCV(SVR(max_iter = 1000), 
                               param_grid = {'C':[0.001, 0.01, 0.1, 10, 50, 100, 500, 1000, 10000],
                                             'kernel':['rbf','poly','sigmoid','linear'],
                                             'epsilon':[0.001,0.001,0.01,0.1, 0.5, 1, 10, 50, 100]
                                             },           
                               cv         = cv,
                               scoring    = 'neg_root_mean_squared_error',
                               n_jobs     = -1,
                               refit      = True)

            boost = GridSearchCV(estimator  = XGBRegressor(random_state = random_seed), 
                                 param_grid = {'max_depth' : [3,5,10], # Maximum depth of a tree
                                               'eta'       : [0.1, 0.3, 0.5], # learning_rate
                                               'n_estimators' : [10, 50, 100, 150, ]
                                               }, 
                                 cv         = cv,
                                 scoring    = 'neg_root_mean_squared_error',
                                 n_jobs     = -1,
                                 refit      = True)
            
            mars = GridSearchCV(estimator  = MARS(), 
                                 param_grid = {
                                               'max_degree' : [0,1,2,],
                                               'penalty'    : [1e-3,  1e-1, 1, 10, 100, 1000],
                                               'max_terms'  : [1, 10,  100,  500, ],
                                               }, 
                                 cv         = cv,
                                 scoring    = 'neg_root_mean_squared_error',
                                 n_jobs     = -1,
                                 refit      = True)
            
            optimizers=[             
                ('MARS', args, random_seed, mars),
                ('RR'  , args, random_seed, ridgecv),
                ('SVR' , args, random_seed, svr),
                ('XGB' , args, random_seed, boost)
            ]
                           
            for (clf_name, args, random_seed, clf) in optimizers:
                np.random.seed(random_seed)
                clf.set_params(**{'cv':KFold(n_splits=n_splits, random_state=random_seed, shuffle=True)})
                
                list_results=[]
                #--------------------------------------------------------------
                s = ''
                s = '-'*80+'\n'
                s += 'Estimator                  : '+ clf_name + '\n'
                s += 'Dataset                    : '+ dataset_name + ' -- '+ target + '\n'
                s += 'Output                     : '+ tn + '\n'
                s += 'Run                        : '+ str(run) + '\n'
                s += 'Random seed                : '+ str(random_seed) + '\n'
                s += '-' * 80 + '\n'
                print(s)
               
                
                if len(X_test)>1:
                    clf.fit(X_train, y_train)
                    y_test_pred = clf.predict(X_test).ravel()
                else:
                    kf = KFold(n_splits = n_splits,
                               shuffle = True, 
                               random_state = random_seed)
                    y_train_pred=y_train.copy()
                    for train_index, test_index in kf.split(X_train):
                         xx_train, xx_test = X_train[train_index], X_train[test_index]
                         yy_train, yy_test = y_train[train_index], y_train[test_index]
                         clf.fit(xx_train, yy_train)
                         y_train_pred[test_index] = clf.predict(xx_test).ravel()

                if n_samples_test == 0:
                    sim={
                         'Y_TRAIN_TRUE'     :y_train,
                         'Y_TRAIN_PRED'     :y_train_pred,
                         'EST_NAME'         :clf_name,
                         'ALGO'             :clf_name,
                         'EST_PARAMS'       :clf.best_estimator_.get_params(),
                         'OPT_PARAMS'       :clf.best_params_,
                         'OUTPUT'           :target,
                         'TARGET'           :target,
                         'SEED'             :random_seed,
                         'ACTIVE_VAR_NAMES' :dataset['feature_names'],
                         'ACTIVE_VAR'       :dataset['feature_names'],
                         'SCALER'           :None
                         }
                    
                 
                    
                    r2 = r2_score(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())
                    r = stats.pearsonr(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())[0]
                    rmse = RMSE(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())  
                   
                    scores = '-'*80+'\n'
                    scores += f"R2Score : {r2}" + '\n'
                    scores += f"Rscore  : {r}" + '\n' 
                    scores += f"RMSE    : {rmse}" + '\n' 
                    scores += '-'*80+'\n'

                    print(scores)
           
      
                if n_samples_test > 0: 
                    
                    sim={
                         'Y_TEST_TRUE'      :y_test,
                         'Y_TEST_PRED'      :y_test_pred,
                         'EST_NAME'         :clf_name,
                         'ALGO'             :clf_name,
                         'EST_PARAMS'       :clf.best_estimator_.get_params(),
                         'OPT_PARAMS'       :clf.best_params_,
                         'OUTPUT'           :target,
                         'TARGET'           :target,
                         'SEED'             :random_seed,
                         'ACTIVE_VAR_NAMES' :dataset['feature_names'],
                         'ACTIVE_VAR'       :dataset['feature_names'],
                         'SCALER'           :None,
                         }
                 
                    
                    pl.figure()
                    
                    r2   = r2_score(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())
                    r    = stats.pearsonr(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())[0]
                    rmse = RMSE(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())                
                    rmsl = rms(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())     
                    mape = MAPE(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())   
                    acc  = accuracy_log(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())  
                    kge_ = kge(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())[0][0]
                    nse_ = nse(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())
                    
                    scores = '-'*80+'\n'
                    scores += f"R2Score : {r2}" + '\n'
                    scores += f"Rscore  : {r}" + '\n' 
                    scores += f"RMSE    : {rmse}" + '\n' 
                    scores += f"RMSL    : {rmsl}" + '\n' 
                    scores += f"MAPE    : {mape}" + '\n' 
                    scores += f"ACC     : {acc}"  + '\n' 
                    scores += f"KGE     : {kge_}" + '\n' 
                    scores += f"NSE     : {nse_}" + '\n' 
                    scores += '-'*80+'\n'

                    print(scores)
                    
                    
                    if task == 'forecast' or task == 'regression':
                        pl.figure(figsize=(12,5)); 
                        
                        s = range(len(y_test))
                        
                        pl.plot(sim['Y_TEST_TRUE'][s].ravel(), 'r-o', label='Real data',)
                        
                        pl.plot(sim['Y_TEST_PRED'][s].ravel(), 'b-o', label='Predicted',)
                                                
                        acc = accuracy_log(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())                
                        
                        pl.title(dataset_name
                                 + ' -- '
                                 + target
                                 + '\nRMSE = '
                                 + str(rmse)
                                 + ', '
                                 + 'R$^2$ = '
                                 + str(r2)
                                 + ', '
                                 + 'R = '
                                 + str(r)
                                 + 'KGE = '
                                 + str(kge_))
                        
                        pl.ylabel(dataset_name)
                        
                        pl.title(sim['EST_NAME']
                                 + ': (Testing) R$^2$='
                                 + str('%1.3f' % r2)
                                 + '\t RMSE='
                                 + str('%1.3f' % rmse)
                                 + '\t MAPE ='
                                 + str('%1.3f' % mape)
                                 + '\t R ='
                                 + str('%1.3f' % r)
                                 + '\t NSE ='
                                 + str('%1.3f' % nse_)
                                 + '\t KGE ='
                                 + str('%1.3f' % kge_)
                                  )
                        pl.show()                                                        
                    
                sim['RUN']          = run;
                sim['DATASET_NAME'] = dataset_name; 
                
                list_results.append(sim) 
        
                data    = pd.DataFrame(list_results)
                ds_name = dataset_name.replace('/','_').replace("'","").lower()
                tg_name = target.replace('/','_').replace("'","").lower()
                algo    = sim['ALGO'].split(':')[0] 
                
                pk = (path
                      # +'_'
                      + basename
                      + '_'
                      + '_run_'
                      + str("{:02d}".format(run))
                      + '_'
                      + ("%15s"%ds_name         ).rjust(15).replace(' ','_')
                      # + '_'
                      + ("%9s"%sim['EST_NAME']  ).rjust( 9).replace(' ','_')
                      # + '_'
                      + ("%10s"%algo            ).rjust(10).replace(' ','_')
                      # + '_'
                      + ("%15s"%tg_name         ).rjust(25).replace(' ','_')
                      # + '_'
                      + '.pkl') 
                
                pk = pk.replace(' ','_').replace("'","").lower()
                pk = pk.replace('(','_').replace(")","_").lower()
                pk = pk.replace('[','_').replace("]","_").lower()
                pk = pk.replace('-','_').replace("_","_").lower()
                
                data.to_pickle(pk)
                
tEnd = t.time()
print("Total Time:", dt.timedelta(seconds = tEnd - tStart))


import platform, psutil

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

text_file = open(path + "Time.txt", "w")

t = "="*40 + " System Information "+ "="*40 + '\n'
text_file.write(t)
uname = platform.uname()
text_file.write(f"System: {uname.system}\n")
text_file.write(f"Release: {uname.release}\n")
text_file.write(f"Version: {uname.version}\n")
text_file.write(f"Machine: {uname.machine}\n")
text_file.write(f"Processor: {uname.processor}\n")

t = "="*40 + " CPU Info " + "="*40 + '\n'
text_file.write(t)
text_file.write(f"Physical cores: {psutil.cpu_count(logical=False)}\n")
text_file.write(f"Total cores: {psutil.cpu_count(logical=True)}\n")
cpufreq = psutil.cpu_freq()
text_file.write(f"Max Frequency: {cpufreq.max:.2f}Mhz\n")
text_file.write(f"Min Frequency: {cpufreq.min:.2f}Mhz\n")

t = "="*40 + " Memory Information "+ "="*40 + '\n'
text_file.write(t)
svmem = psutil.virtual_memory()
text_file.write(f"Total: {get_size(svmem.total)}\n\n")

text_file.write(f"Runs: {n_runs}\n")
text_file.write(f"Total Time: {dt.timedelta(seconds = tEnd - tStart)}")
text_file.close()
