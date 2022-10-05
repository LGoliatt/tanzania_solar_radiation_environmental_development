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
# def clean_data():
#     #%%
#     fn='Solar-Measurements_Tanzania_DarEsSalaam_WB-ESMAP_Raw.csv'
#     df=pd.read_csv(fn, sep=';')    
#     y = df[['Year',]].values.ravel()
#     m = df[['Month',]].values.ravel()
#     d = df[['Day']].values.ravel()
#     dt=pd.to_datetime(pd.DataFrame(np.c_[y,m,d], columns=['year', 'month', 'day']), unit='D')
#     df['Date']=dt    
#     df = df[df['GHI']>0]
#     dg = df.groupby('Date').mean()
#     dg['Date']=dg.index
#     dg.drop(['Year', 'Month', 'Day', 'Date'], axis=1, inplace=True)
#     dg.to_csv('daily_tanzania_daressalaam_wb-esmap_qc.csv', )
    
    #%%

#%%    
def read_data_tanzania(station='daressalaam', period='daily',
                   kind='ml', plot=True,
                   scale=False, degree=1, roll=False, window=7,
                   model=1,
            ):
    #%%
    filename='./data/data_tanzania/'+period+'_tanzania_'+station+'_wb-esmap_qc.csv'
    #filename='./'+period+'_tanzania_'+station+'_wb-esmap_qc.csv'
    data = pd.read_csv(filename, index_col=0, delimiter=',')
   
    date_max = '2017-05-31'; data=data[data.index<=date_max]
    date_min = '2015-05-21'; data=data[data.index>=date_min]

    date_range = data.index
    data.columns = [x.replace('_', '-') for x in data.columns]
    # print(date_range)
    for c in data.columns:
       #print(c)
       data[c].interpolate(method='linear', inplace=True)
       #data[c].plot(); pl.show()

    #['T-amb', 'RH', 'WS', 'WS-gst', 'WD', 'WD-st-dev', 'BP']
    # var_names={
    #         #1: ['T-amb', 'RH', 'WS', 'WS-gst', 'WD', 'BP'],
    #         1: ['T-amb', 'RH', 'WS',           'WD', 'BP'],
    #         2: ['T-amb', 'RH', 'WS',           'WD',     ],
    #         3: ['T-amb', 'RH',                 'WD',     ],
    #         4: [         'RH',                 'WD',     ],
    #         5: ['T-amb', 'RH',                           ],
    #        #7: ['T-amb', 'RH',                 'WD',     ],
    #         }
    
    # var_names={
    #      1:  ['T-amb', 'RH', 'WS', 'WS-gst', 'WD', 'BP','DNI', 'DHI', 'WD-st-dev'],
    #      2:  ['T-amb', 'RH', 'WS',           'WD', 'BP','DNI'],
    #      3:  ['T-amb', 'RH', 'WS',           'WD',      'DNI'],
    #      4:  ['T-amb', 'RH',                 'WD',      'DNI'],
    #      5:  [         'RH',                 'WD',      'DNI'],
    #      6:  ['T-amb', 'RH',                            'DNI'],
    #      7:  ['T-amb', 'RH', 'WS', 'WS-gst', 'WD', 'BP','DNI',        'WD-st-dev'],
    #      8:  ['T-amb', 'RH', 'WS', 'WS-gst', 'WD', 'BP',              'WD-st-dev'],
    #      9:  ['T-amb', 'RH', 'WS',           'WD', 'BP'],
    #      10: ['T-amb', 'RH', 'WS',           'WD',     ],
    #      11: ['T-amb', 'RH',                 'WD',     ],
    #      12: [         'RH',                 'WD',     ],
    #      13: ['T-amb', 'RH',                           ],
    #      }
    
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
            #pl.title(group, y=0.5, loc='right')
            data[data.columns[i]].plot(marker='', lw=0)
            pl.axvline(n, color='k', ls='-.')
            # pl.xlabel('Year')
            #pl.legend(loc=(1.01,0.5))
            pl.ylabel(group)
        fig.autofmt_xdate(rotation=30,)            
        pl.show()
    #%%
    # variable_names, target_names = [#'DNI', 
    #                                 #'DHI',
    #                                 'T-amb', 'RH', 'WS', 'WS-gst', 'WD', 'WD-st-dev', 'BP'], ['GHI']
    
    X=data[variable_names]
    y=data[target_names]#/1e3


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
      #'y_min'           : y_scaler.data_min_[0],
      #'y_max'           : y_scaler.data_max_[0],
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
    
def read_tanzania_stations(period='daily',plot=False,):
    #%%
    datasets = [
        read_data_tanzania(station='daressalaam', period=period, scale=False),
    ]
    
    for ds in datasets:
        print('%19s'%ds['name'], ds['X_train'].shape, ds['X_test'].shape, ds['date_range'][0], ds['date_range'][-1])
       
        
    n_features, n_samples, n_datasets = ds['X_train'].shape[1],ds['X_train'].shape[0],len(datasets)
    X_train = np.zeros((n_samples, n_features, n_datasets,))
    y_train = np.zeros((n_samples, n_datasets,))
    
    stations=[]
    for i,ds in enumerate(datasets):
        n_features = len(ds['feature_names'])
        n_samples  = ds['X_train'].shape[0]
        #print(n_features,n_samples)
        X_train[:,:,i] =  ds['X_train']
        y_train[:,i] =  ds['y_train']
        stations.append([ds['name']])
    
    n_features, n_samples, n_datasets = ds['X_test'].shape[1],ds['X_test'].shape[0],len(datasets)
    X_test = np.zeros((n_samples, n_features, n_datasets,))
    y_test = np.zeros((n_samples, n_datasets,))
    
    for i,ds in enumerate(datasets):
        n_features = len(ds['feature_names'])
        n_samples  = ds['X_test'].shape[0]
        #print(n_features,n_samples)
        X_test[:,:,i] =  ds['X_test']
        y_test[:,i] =  ds['y_test']

    n_samples, n_features, _ = X_train.shape
    variable_names=['x_'+str(i) for i in range(n_features)] #np.array(X_train.columns)
    target_names=['y_'+str(i) for i in range(n_datasets)]#np.array(y_train.columns)
    data_description = ['var_'+str(i) for i in range(X_train.shape[1])]
    #sn = os.path.basename(filename).split('-')[0].split('/')[-1]
    dataset=  {
      #'task':'forecast',
      'task':'regression',
      'name':'Solar Radiation Vietnam',
      'feature_names':variable_names,'target_names':target_names,
      'n_samples':n_samples, 'n_features':n_features,
      'X_train':X_train,
      'X_test':X_test,
      'y_train':y_train,
      'y_test':y_test,
      'targets':target_names,
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'reference':"",      
      'stations':stations,
      'normalize': 'MinMax',
      }   
    #%%
    return dataset    
    
#%%-----------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = [                 
            read_tanzania_stations(period='daily',plot=True,)
            ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print(D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
