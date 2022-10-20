#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import scipy as sp
import pylab as pl
import pandas as pd
import seaborn as sns
import glob

from matplotlib.dates import DateFormatter
from datetime import datetime
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict, 
                                     TimeSeriesSplit, cross_val_score, 
                                     LeaveOneOut, KFold, StratifiedKFold,
                                     cross_val_predict,train_test_split)
#-------------------------------------------------------------------------------
def read_data_tanzania(station='daressalaam', period='daily',
                   kind='ml', plot=True,
                   scale=False, degree=1, roll=False, window=7,
                   model=1,
            ):
    #%%
    filename='./data/data_tanzania/'+period+'_tanzania_'+station+'_wb-esmap_qc.csv'
    data = pd.read_csv(filename, index_col=0, delimiter=',')
   
    date_max = '2017-05-31'; data=data[data.index<=date_max]
    date_min = '2015-05-21'; data=data[data.index>=date_min]

    date_range = data.index
    data.columns = [x.replace('_', '-') for x in data.columns]
    for c in data.columns:
       data[c].interpolate(method='linear', inplace=True)

   
    
    from itertools  import combinations
    count=1
    var_names={}
    # for j in range(1,7):
    for j in range(7,0,-1):
        comb =combinations(['T-amb', 'RH', 'WS', 'WS-gst', 'WD', 'BP'],j)
        for i in list(comb):
            var_names[count]=[k for k in list(i)];        
            count+=1
    
    variable_names=var_names[model]
    target_names=['GHI']
    print(target_names+variable_names)
    
    data=data[target_names+variable_names]     
    var_to_plot=data.columns
    df=data[var_to_plot]
    # print(df.shape)
    n=int(df.shape[0]*0.7)
    df.index=range(df.shape[0])
    id0=df.index <= n
    id1=df.index >  n 
    if plot:
        pl.rc('text', usetex=True)
        pl.rc('font', family='serif',  serif='Times')
        fig=pl.figure(figsize=(6,8))
        for i,group in enumerate(df.columns):
            pl.subplot(len(df.columns), 1, i+1)
            df[group].iloc[id0].plot(marker='', label='Training')#,fontsize=16,)#pyplot.plot(dataset[group].values)
            df[group].iloc[id1].plot(marker='', label='Test')#,fontsize=16,)#pyplot.plot(dataset[group].values)
            data[data.columns[i]].plot(marker='', lw=0)
            pl.axvline(n, color='k', ls='-.')
            pl.ylabel(group)
        fig.autofmt_xdate(rotation=30,)            
        pl.show()
    
    X=data[variable_names]
    y=data[target_names]


    import seaborn as sns
    df=X.copy(); df[target_names]=y.values
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = pl.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap =  cmap="YlGnBu"
    sns.heatmap(corr, mask=mask, cmap=cmap, #vmax=.3, center=0, 
                annot=True, #fmt="d")
                square=True, linewidths=.5, #cbar_kws={"shrink": .5},
            )
    #pl.show()
 
    if scale:
        X_scaler=MinMaxScaler()
        X = pd.DataFrame(data=X_scaler.fit_transform(X), columns=X.columns)
    
    if roll:
         y_roll = pd.DataFrame(y).rolling(window=window, min_periods=1, win_type=None).mean().values
         y[target_names] = y_roll
         
    X_train, X_test = X.iloc[:n].values, X.iloc[n:].values    
    y_train, y_test = y.iloc[:n].values, y.iloc[n:].values    
    n_samples, n_features = X_train.shape 

    if kind=='lstm':
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    if degree>1:
        from sklearn.preprocessing import PolynomialFeatures
        X_train = PolynomialFeatures(degree=degree).fit_transform(X_train)
        X_test  = PolynomialFeatures(degree=degree).fit_transform(X_test)
        variable_names = ['v'+str("%2.2d" % (i+1)) for i in range(X_train.shape[1])]
        
       
    regression_data =  {
      'task'            : 'regression',
      'name'            : station + ' M'+str(model),
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train.reshape(1,-1),
      'y_test'          : y_test.reshape(1,-1),      
     'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'items'           : None,
      'reference'       : "https://energydata.info/dataset/vietnam-solar-radiation-measurement-data",
      'normalize'       : 'None',
      'date_range'      : date_range,
      }
    #%%
    return regression_data
    #%%
        
#%%-----------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = [                 
            read_data_tanzania(station='daressalaam', period='daily', model=i, plot=True,)
            ] 
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print(D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
