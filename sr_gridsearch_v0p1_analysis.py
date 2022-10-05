#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import pandas as pd
from pandas.core.base import DataError
import math
import matplotlib.pyplot as pl
import scipy as sp
import glob
import seaborn as sns
import re
import os, sys
import itertools
import hydroeval as he
from scipy import stats

from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error

from util.metrics import rrmse, agreementindex,  lognashsutcliffe,  nashsutcliffe

import skill_metrics as sm

#%%-----------------------------------------------------------------------------
pd.options.display.float_format = '{:.3f}'.format
palette_color="Set1"#"Blues_r"

def fmt(x): 
    if (type(x) == str or type(x) == tuple or type(x) == list):
        return str(x)
    else:
      if (abs(x)>0.001 and abs(x)<1e0):
        return '%1.3f' % x   
      else:
        return '%1.2f' % x #return '%1.3f' % x
  
def fstat(x):
  #m,s= '{:1.4g}'.format(np.mean(x)), '{:1.4g}'.format(np.std(x))
  #m,s, md= fmt(np.mean(x)), fmt(np.std(x)), fmt(np.median(x)) 
  m,s, md= np.mean(x), np.std(x), np.median(x) 
  #text=str(m)+'$\pm$'+str(s)
  s = '--' if s<1e-8 else s
  text=fmt(m)+' ('+fmt(s)+')'#+' ['+str(md)+']'
  return text
  
def mean_percentual_error(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  return np.mean(np.abs(y_pred - y_true)/np.abs(y_true))*100

def VAF(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  return ( 1 - np.var(y_true - y_pred)/np.var(y_true) )*100

def accuracy_log(y_true, y_pred):
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return (np.abs(np.log10(y_true/y_pred))<0.3).sum()/len(y_true)*100
    
def rmse_lower(y_t, y_p, typ='Kx'):
    
    if typ == 'Kx':
        value=100
        r=y_t.ravel()<=value
        return (he.rmse(y_t[r], y_p[r])[0])
    elif typ == 'Q':
        r1=[ True,  True,  True, False, False, False, False, False,  True,
           False,  True, False,  True,  True,  True, False,  True,  True,
            True, False,  True,  True, False, False,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True, False,
            True,  True,  True,  True,  True, False,  True, False,  True,
            True,  True, False,  True, False, False]
        r2=[ True, False, False,  True,  True,  True, False, False,  True,
            True, False, False, False,  True,  True,  True,  True,  True,
           False, False]
            
        r = r1 if len(y_t)==51 else r2
        return(he.rmse(y_t[r], y_p[r])[0])
    else:
        sys.exit('Type not defined')

# http://www.jesshamrick.com/2016/04/13/reproducible-plots/
def set_style():
    # This sets reasonable defaults for size for
    # a figure that will go in a paper
    sns.set_context("paper")
    #pl.style.use(['seaborn-white', 'seaborn-paper'])
    #matplotlib.rc("font", family="Times New Roman")
    #(_palette("Greys", 1, 0.99, )
    #sns.set_palette("Blues_r", 1, 0.99, )
    sns.set_palette(palette_color, )
    sns.set_context("paper", font_scale=1.8, 
        rc={"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
            'xtick.labelsize':16,'ytick.labelsize':16,
            'font.family':"Times New Roman", }
        ) 
    # Set the font to be serif, rather than sans
    #sns.set(font='serif', font_scale=1.4,)
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style(style="white", rc={
        #"font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    
    #os.system('rm -rf ~/.cache/matplotlib/tex.cache/')
    pl.rc('text', usetex=True)
    #pl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    pl.rc('font', family='serif',  serif='Times')

#sns.set(style="ticks", palette="Set1", color_codes=True, font_scale=1.4,)
#%%-----------------------------------------------------------------------------
#fn='./data/data_ldc_vijay/sahay_2011.csv'
#A = pd.read_csv(fn, delimiter=';')
#B = A.drop(labels=['Number', 'Stream', 'Observed',], axis=1)
#y_test = A[['Observed']].values
#for c in B:
#    y_pred=B[[c]].values
#    
#    rmse, r2 = mean_squared_error(y_test, y_pred)**.5, r2_score(y_test, y_pred)
#    r=stats.pearsonr(y_test.ravel(), y_pred.ravel())[0]
#    acc=accuracy_log(y_test.ravel(), y_pred.ravel())
#    #rmslkx=rmse_lower(y_test.ravel(), y_pred.ravel(), typ='Kx')
#    #rmslq=rmse_lower(y_test.ravel(), y_pred.ravel(), typ='Q')
#    print("%12s \t %8.2f %8.2f %8.2f %8.2f" % (c,rmse, acc, r2,r))
    
#%%-----------------------------------------------------------------------------
    
set_style()    

plotspath='plots/'    
basename='eml__'

#path='./pkl_energy_efficiency'
#path='./pkl_qsar_aquatic_toxicity'
#path='./pkl_cahora_bassa'
#path='./pkl_energy_appliances'
#path='./pkl_solar_radiation*'
#path='./pkl_longitudinal_dispersion_coefficient'
#path='./pkl_sri__daressalaam*'
path='./pkl_join_data_basis/'
path='./pkl_sr_gs_/'

#from pandas.compat.pickle_compat import _class_locations_map
#
#_class_locations_map.update({
#    ('pandas.core.internals.managers', 'BlockManager'): ('pandas.core.internals', 'BlockManager')
#}) 

pkl_list  = []
for (k,p) in enumerate(glob.glob(path)):
    pkl_list += glob.glob(p+'/'+'*.pkl')

#
pkl_list.sort()
#
# leitura dos dados
#
A=[]
for pkl in pkl_list:
    #print(pkl)
    df = pd.read_pickle(pkl)       
    A.append(df)
#
A = pd.concat(A, sort=False)

#%%
# remove/collect information

#A=A[A['ALGO']=='CMA-ES: Covariance Matrix Adaptation Evolutionary Strategy']
#A=A[A['RUN']<=50]


#%%
models_to_remove = ['FFNET', 'BR',  'DT',  'PA',  'VR', #'ELM', 
                    'ANN', 
                    'SVR', 'SVM', 'PR', 'EN', 'KRR' ]

models_to_remove = [
                    #'MARS', 
                    #'XGB', 'RBFNN', 
                    #'GPR', 'SVR',
                    #'GPR-FS', 'SVR-FS', 
                    #'XGB-FS'
                    'MLP', 'ELM',
                    ]
#models_to_remove = []
for m in models_to_remove:
    A = A[A['EST_NAME'] != m]    

A['DATASET_NAME'] = [d.split(' ')[1].split('-')[0] for d in A['DATASET_NAME'] ]
#datasets_to_remove = ['LDC case 8', 'LDC case 9',]
# datasets_to_remove = ['M1','M3','M7','M8','M10','M14','M15','M16','M17','M18','M19','M20','M21','M22','M23','M24','M25','M26','M27','M29','M30']
# datasets_to_remove = ['M1','M2','M3','M4','M5','M6','M7','M8','M10','M14','M15','M16','M17','M18','M19','M20','M21','M22','M23','M24','M25','M26','M27','M28','M29','M30']
datasets_to_remove = ["M"+str(i) for i in range(1,64)]
datasets_to_remove.remove('M43')
datasets_to_remove.remove('M45')
datasets_to_remove.remove('M57')
datasets_to_remove.remove('M59')
datasets_to_remove.remove('M60')
datasets_to_remove.remove('M63')
print(datasets_to_remove)
#datasets_to_remove = ['LDC case '+str(i) for i in range(8)]
# datasets_to_remove = []
for m in datasets_to_remove:
    A = A[A['DATASET_NAME'] != m]    

# Deixar comentadas as linhas abaixo
#if A['DATASET_NAME'].unique()[0] == 'Energy Efficiency':
#    A['DATASET_NAME'] = A['OUTPUT']; A['OUTPUT']='Load'         

# A['DATASET_NAME'] = [d.split(' ')[1].split('-')[0] for d in A['DATASET_NAME'] ]
#A['DATASET_NAME'] = [d.split(' ')[1] for d in A['DATASET_NAME'] ]
for i in range(1,64):
    A["DATASET_NAME"].replace({"M"+str(i): "S"+f"{i:02d}"}, inplace=True)#  MODIFICANDO O NOME DO SUBCONJUNTO
# A["DATASET_NAME"].replace({"S09": "S01","S11": "S02","S12": "S03","S13": "S04","S02": "S05","S04": "S06","S05": "S07","S06": "S08","S28": "S09"}, inplace=True)    

#%%
steps=['TRAIN', 'TEST'] if 'Y_TEST_PRED' in A.columns else ['TRAIN']


steps=[ 'TEST'] if 'Y_TEST_PRED' in A.columns else ['TRAIN']

C = []
for step in steps:
    for k in range(len(A)):
        df=A.iloc[k]
        y_true = pd.DataFrame(df['Y_'+step+'_TRUE'], columns=[df['OUTPUT']])#['0'])
        y_pred = pd.DataFrame(df['Y_'+step+'_PRED'], columns=[df['OUTPUT']])#['0'])
        #print (k, df['EST_PARAMS'])
        
        run = df['RUN']
        av = df['ACTIVE_VAR']
        ds_name = df['DATASET_NAME']
        s0 = ''.join([str(i) for i in av])
        s1 = ' '.join(['x_'+str(i) for i in av])
        s2 = '|'.join(['$x_'+str(i)+'$' for i in av])
        var_names = y_true.columns
        
        #df['EST_PARAMS']['scaler']=df['SCALER']
        
        if len(y_true)>0:
            for v in var_names:
                _mape    = abs((y_true[v] - y_pred[v])/y_true[v]).mean()*100
                _vaf     = VAF(y_true[v], y_pred[v])
                _r2      = r2_score(y_true[v], y_pred[v])
                _mae     = mean_absolute_error(y_true[v], y_pred[v])
                _mse     = mean_squared_error(y_true[v], y_pred[v])
                _rrmse   = rrmse(y_true[v], y_pred[v])
                _wi      = agreementindex(y_true[v], y_pred[v])
                _r       = stats.pearsonr(y_true[v], y_pred[v])[0]
                #_nse     = he.nse(y_true.values, y_pred.values)[0]
                _nse     = nashsutcliffe(y_true.values, y_pred.values)
                #_lnse    = lognashsutcliffe(y_true.values, y_pred.values)
                _rmse    = he.rmse(y_true.values, y_pred.values)[0]
                #_rmsekx  = rmse_lower(y_true.values, y_pred.values, 'Kx')
                #_rmseq   = rmse_lower(y_true.values, y_pred.values, 'Q')
                _kge     = he.kge(y_true.values, y_pred.values)[0][0]
                _mare    = he.mare(y_true.values, y_pred.values)[0]
                dic     = {'Run':run, 'Output':v, 'MAPE':_mape, 'R$^2$':_r2, 'MSE':_mse,
                          'Active Features':s2, 'Seed':df['SEED'], 
                          'Dataset':ds_name, 'Phase':step, 'SI':None,
                          'NSE': _nse, 'MARE': _mare, 'MAE': _mae, 'VAF': _vaf, 
                          'Active Variables': ', '.join(df['ACTIVE_VAR_NAMES']),
                          #'RMSELKX':rmsekx, 'RMSELQ':rmseq, 
                          #'Scaler': df['SCALER'], 
                          'KGE': _kge,
                          'RMSE':_rmse, 'R':_r, 'Parameters':df['EST_PARAMS'],
                          'NDEI':_rmse/np.std(y_true.values),
                          'WI':_wi, 'RRMSE':_rrmse,
                          'y_true':y_true.values.ravel(), 
                          'y_pred':y_pred.values.ravel(),
                          'Optimizer':df['ALGO'].split(':')[0], #A['ALGO'].iloc[0].split(':')[0],
                          'Accuracy':accuracy_log(y_true.values.ravel(), y_pred.values.ravel()),
                          'Estimator':df['EST_NAME']}
                C.append(dic)
    
#        if step=='TEST':
#            pl.plot(y_true.values,y_true.values,'r-',y_true.values, y_pred.values, 'b.', )
#            t=ds_name+' - '+df['EST_NAME']+': '+step+': '+str(fmt(r2))
#            pl.title(t)
#            pl.show()

#%%
#            
#df              = pd.read_csv('./references/reference_tayfur.csv', delimiter=';')
#ref_estimator   = df['Estimator'].unique()[0]
#df['Run']       = 30
#
#for i in range(len(df)):
#    aux = dic.copy()
#    for c in df:
#        aux[c] =  df.iloc[i][c]
#    
#    C.append(aux)
        
C = pd.DataFrame(C)
C = C.reindex(sorted(C.columns), axis=1)

#C[C['Run']<25]
#C['Output']='$K_x$(m$^2$/s)'

#C=C[C['Run']< 50]
#C=C[C['Run']>=6]; C=C[C['Run']< 36  ]
#C=C[C['Run']>=12]; C=C[C['Run']< 42]
C['Dataset'] = [i.replace('Naula model','Case') for i in C['Dataset']]
# C['Output']='$Q_t$'
C['Output']='$GHI$'


#C=C[C['Optimizer']=='SGA']
#C=C[C['Optimizer']=='PSO']
#C=C[C['Optimizer']=='DE']
#%%

metrics=[
        #'R', 
        'R$^2$', 
        #'WI',
        #'RRMSE',
        #'RMSELKX', 'RMSELQ', 
        #'RMSE$(K_x<100)$', 'RMSE$(B/H<50)$', 
        'RMSE', #'NDEI', 
        'MAE', #'Accuracy', 
        'MAPE',
        #'NSE', #'LNSE', 
        # 'KGE',
        #'MARE',
        #'VAF', 'MAE (MJ/m$^2$)', 'R',  'RMSE (MJ/m$^2$)',
        ]
    
metrics_max =  ['NSE', 'VAF', 'R', 'Accuracy','R$^2$', 'KGE', 'WI']    
#%%
#aux=A.iloc[0]
#D1=pd.read_csv('./references/deng_2002.csv', delimiter=';')
#D2=pd.read_csv('./data/data_ldc_vijay/tayfur_2005.csv', delimiter=';')
#
#D1.sort_values(by=['Measured'], axis=0, inplace=True)
#D2.sort_values(by=['Kx(m2/s)'], axis=0, inplace=True)


#%%
#idx_drop = C[['Dataset','Estimator', 'Run', 'Phase', 'Output']].drop_duplicates().index
#C=C.iloc[idx_drop]

#C1 = pd.read_csv('./references/reference_zaher_elm.csv')
#C1['Estimator']=ref_estimator
#C1 = C1.reindex(sorted(C1.columns), axis=1)

#C.sort_index(axis=0, level=['Dataset','Estimator'], inplace=True)
#C=C.append(C1,)# sort=True)
#C=C.append(C1, sort=True)
#%%
#S=[]   
#for (i,j), df in C.groupby(['Dataset','Active Variables']): 
#    S.append({' Dataset':i, 'Active Variables':j})
#    print('\t',i,'\t','\t',j)

#S=pd.DataFrame(S)
#print(S)
#sns.catplot(x='Dataset', y='R$^2$', data=C, hue='Phase', kind='bar', col='Estimator')
#%%
#for (p,e,o), df in C.groupby(['Phase','Estimator', 'Output']):
# if p=='TEST':
#  #if e!= ref_estimator:  
#    print ('='*80+'\n'+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
#       
#    df1 = df.groupby(['Active Variables'])
#    
#    active_variables=pd.DataFrame()
#    for m in metrics: 
#        grp  = list(df1.groups.keys())
#        mean = df1[m].agg(np.mean).values
#        std  = df1[m].agg(np.std).values
#        v    = [fmt(i)+' ('+fmt(j)+')' for (i,j) in zip(mean,std)]
#        
#        active_variables['Set']=grp
#        active_variables[m]=v
#    
#    active_variables.sort_values(by=['Accuracy'], axis=0, inplace=True, ascending=False)    
#    
#    print(active_variables)

#%%
anova=[]
for f,df in C.groupby(['Phase', ]):    
    #df1 = df[df['Estimator']!=ref_estimator]
    df1=df
    for (d,o),df2 in df1.groupby(['Estimator','Output', ]):    
            if f!='TRAIN':
                print('\n'+'='*80+'\n'+str(d)+' '+str(f)+' '+str(o)+'\n'+'='*80)
                nam = 'Dataset'
                groups=df2.groupby(nam,)
                print(df2[nam].unique())
                
                for m in metrics:
                    #-pl.figure()
                    dic={}
                    for g, dg in groups: 
                        #-h=sns.distplot(dg[m].values, label=g)
                        dic[g]=dg[m].values
                        
                    f_, p_ = stats.f_oneway(*dic.values())
                    #f_, p_ = stats.kruskal(*dic.values())
                    #-h.legend(); h.set_xlabel(m); 
                    #-h.set_title('Dataset: '+d+'\n F-statistic = '+fmt(f_)+', '+'$p$-value = '+fmt(p_));
                    #-h.set_title('Dataset: '+d+' ($p = $'+fmt(p_)+')');
                    #-pl.ylabel(m)
                    #-pl.show()     
                    anova.append({ #m:fmt(p_),
                                  'Phase':f, 
                                  'Output':o,'Metric':m, 'F-value':fmt(f_), 'p-value':fmt(p_),  
                                  'Dataset':d})


anova=pd.DataFrame(anova)
groups=anova.groupby('Dataset')
p_value_table=[]
for g, dg in groups:
    dic = dict(zip(dg['Metric'],dg['p-value']))
    dic['Dataset'] = g
    p_value_table.append(dic)

p_value_table = pd.DataFrame(p_value_table)    

fn = basename+'_comparison_p_values_anova_datasets'+'_table.tex'
fn = re.sub('-','_', re.sub('\/','',fn)).lower()
p_value_table = p_value_table.reindex(sorted(p_value_table.columns), axis=1)
p_value_table.to_latex(buf=plotspath+fn, index=False)
print(p_value_table)

#%%    
#anova=[]
#for (f,d,o,), df in C.groupby(['Phase', 'Dataset', 'Output', ]):
# if f!='TRAIN':
#    print('\n'+'='*80+'\n'+str(d)+' '+str(f)+' '+str(o)+'\n'+'='*80)
#    nam = 'Estimator'
#    groups=df.groupby(nam,)
#    print(df['Estimator'].unique())
#    
#    for m in metrics:
#        dic={}
#        for g, dg in groups: 
#            dic[g]=dg[m].values
#            
#        f_, p_ = stats.f_oneway(*dic.values())
#        anova.append({'Metric':m,  'F-value':fmt(f_), 'p-value':fmt(p_), 'Phase':f, 
#                      #'Output':o,
#                      'Dataset':d})
#    
#    anova=pd.DataFrame(anova)
#    print (anova)
#    #idx=anova.columns[[2,1,0,3]]
#    #print (anova[idx].to_latex(index=None))
    
#%%   
aux=[]
for (f,d,e,o,), df in C.groupby(['Phase', 'Dataset', 'Estimator','Output',]):
    #print(d,f,e,o,len(df))
    dic={}
    dic['Dataset']=d
    dic['Phase']=f
    dic['Output']=o
    dic['Estimator']=e
    for f in metrics:
        dic[f]= fstat(df[f])
    
    aux.append(dic)
    
tbl = pd.DataFrame(aux)
tbl = tbl[tbl['Phase']=='TEST']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.width=140
df_table=[]
for (f,d), df in tbl.groupby(['Phase', 'Dataset',]):
#for (d,f), df in tbl.groupby(['Dataset','Phase', ]):
    for m in metrics:
        x, s = [], []
        for v in df[m].values:
            x_,s_ = v.split(' ')
            x_    = float(x_)
            x.append(x_)
            s.append(s_)
        
        x_idx     = np.argmax(x) if m in metrics_max else np.argmin(x)
        x         =[fmt(i) for i in x]
        x[x_idx]  = '{ \\bf '+x[x_idx]+'}'
        
        df[m]     = [ i+' '+j for (i,j) in zip(x,s)]
        
        
    fn = basename+'_comparison_datasets'+'_table_'+d.lower()+'_'+f.lower()+'.tex'
    fn = re.sub('\^','', re.sub('\$','',fn))
    fn = re.sub('\(','', re.sub('\)','',fn))
    fn = re.sub(' ','_', re.sub('\/','',fn))
    
    print('\n'+'='*80+'\n'+str(d)+' '+str(f)+'\n'+'='*80)
    print(fn)
    df['Modeling Phase']=df['Phase']
    df.drop(['Phase',], axis=1)
    #df1=df[['Modeling Phase', 'Dataset', 'Estimator', 'R', 'VAF', 'RMSE (MJ/m$^2$)', 'MAE (MJ/m$^2$)', 'NSE']]
    df1=df[['Modeling Phase', 'Dataset', 'Estimator', ]+metrics]
    #df.drop(['Output', 'Dataset', 'Phase'], axis=1, inplace=True)
    print(df1)
    df_table.append(df1)

df_table=pd.concat(df_table)

cpt = 'Caption to be inserted.'
fn = basename+'_comparison_datasets'+'_table'
fn = re.sub('-','_', re.sub('\/','',fn)).lower()
df_table.drop(labels=['Modeling Phase'], axis=1, inplace=True)
df_table.to_latex(buf=plotspath+fn+'.tex', index=False, escape=False, label=fn, caption=cpt, column_format='r'*df_table.shape[1])
print(df_table)

os.system('cp '+fn+'.tex'+' ./latex/tables/')

#%%    
# radarchart
for (f,d,o,), df in C.groupby(['Phase', 'Estimator', 'Output',]):
    if f=='TEST':
        #print(df[metrics].columns)
        print(f,d,o)
        df_estimator=df.groupby(['Dataset'])[metrics].agg(np.mean)
        df_estimator.index=df_estimator.index.values

        categories=df_estimator.columns
        N = len(categories)
        
        for i in df_estimator.columns: 
            if i in metrics_max:
                df_estimator[i]=df_estimator[i]/df_estimator[i].max()*100
            else:
                df_estimator[i]=df_estimator[i].min()/df_estimator[i]*100


        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

    
        # initialise the spider plot
        pl.figure(figsize=(7,7), )#dpi=72)
        ax = pl.subplot(111, polar=True)
        
        # if you want the first axis to be on top:
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw one axe per variable + add labels labels yet
        pl.xticks(angles[:-1], categories, color='k')
    
        # Draw ylabels
        ax.set_rlabel_position(0)
        pl.ylim(0,108)
        ax.tick_params(axis='y', colors='grey',  grid_linestyle='--', size=7)
        ax.tick_params(axis='x', colors='grey',  grid_linestyle='--', size=7)
        
        for i in range(len(df_estimator)):
            values=df_estimator.iloc[i].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', marker='o', label=df_estimator.index[i])
            ax.fill(angles, values,alpha=0.02)# 'w', alpha=0)

        # Add legend
        #pl.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        #pl.title(d, y=1.1,)
        pl.legend(loc=0, bbox_to_anchor=(1.12, 0.7), title=r"\textbf{"+d+"}", fancybox=True)
        pl.legend(loc=0, bbox_to_anchor=(1.12, 0.8), title=r"{"+d+"}", fancybox=True)

        fn = basename+'300dpi_radarchart_'+str(f)+'_'+str(d)+'_'+str(o)+'.png'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn))
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
        pl.savefig(plotspath+fn,  bbox_inches='tight', dpi=300)

        pl.show()
        
#sys.exit()
#%%    
# https://github.com/pog87/PtitPrince/blob/master/RainCloud_Plot.ipynb
n_estimators = C['Estimator'].unique().shape[0]
ds=C['Dataset'].unique(); ds.sort()
hs=C['Estimator'].unique(); hs.sort(); #hs=np.concatenate([hs[hs!=ref_estimator],hs[hs==ref_estimator]])
#sns.set(font_scale=2.5)

for kind in ['bar',]:#'box', 'violin']:
    for m in metrics:
        kwargs={'edgecolor':"k", 'capsize':0.05, 'alpha':0.95, 'ci':'sd', 'errwidth':1.0, 'dodge':True, 'aspect':2.8, 'legend':None, } if kind=='bar' else {'notch':0, 'ci':'sd','aspect':1.0618,}
#        sns.catplot(x='Dataset', y='R$^2$', data=C, hue='Phase', kind='bar', col='Estimator')
#        g=sns.catplot(x='Dataset', y=m, col='Estimator', data=C, 
#                       kind=kind, sharey=False, hue='Phase', 
#                       **kwargs,);
#        g=sns.catplot(col='Dataset', y=m, hue='Estimator', data=C, 
#                       kind=kind, sharey=False, x='Phase', 
#                       **kwargs,);
        if kind=='bar':
            g=sns.catplot(x='Dataset', y=m, hue='Estimator', #row='Phase', 
                          data=C[C['Phase']=='TEST'], 
                          order=ds, hue_order=hs, kind=kind, sharey=False,  
                          #col_wrap=2, palette=palette_color_1,
                          **kwargs,)            
        elif kind=='box':
            g=sns.catplot(col='Dataset', y=m, hue='Estimator', x='Phase', data=C[C['Phase']=='TEST'], 
                          #order=ds, hue_order=hs,
                          kind=kind, sharey=False, #col_wrap=2,                       
                          **kwargs,)
        elif kind=='violin':
            g=sns.catplot(x='Dataset', y=m, hue='Estimator', col='Phase', data=C[C['Phase']=='TEST'], 
                          #order=ds, hue_order=hs,
                          scale="count", #inner="quartile", 
                          count=0, legend=False,
                          kind=kind, sharey=False,  #col_wrap=2,                       
                          **kwargs,)
        else:
            pass
        
        #g.despine(left=True)
        fmtx='%2.2f'
        for ax in g.axes.ravel():
            ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=00,)# fontdict={'fontsize':17})
            if kind=='bar':
                ax.set_ylim([0, 1.15*ax.get_ylim()[1]])
                ax.set_xlabel(None); #ax.set_ylabel(m);
                _h=[]
                for p in ax.patches:
                    _h.append(p.get_height())
                 
                _h=np.array(_h)
                _h=_h[~np.isnan(_h)]
                _h_max = np.max(_h)
                for p in ax.patches:
                    _h= 0 if np.isnan(p.get_height()) else p.get_height()
                    p.set_height(_h)                
                    ax.text(
                            x=p.get_x() + p.get_width()/2., 
                            #y=1.04*p.get_height(), 
                            y=0.02*_h_max+p.get_height(), 
                            s=fmtx % p.get_height() if p.get_height()>0 else None, 
                            #fontsize=16, 
                            color='black', ha='center', 
                            va='bottom', rotation=90, weight='bold',
                            )
            pl.legend(bbox_to_anchor=(1.05, 0.5), loc=10, borderaxespad=0., ncol=1, fontsize=16, ) 
            #pl.legend(loc=0, ncol=n_estimators, borderaxespad=0.,)
            
        fn = basename+'300dpi_comparison_datasets'+'_metric_'+m.lower()+'_'+kind+'.png'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn))
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('-','_', re.sub('\<','_',fn)).lower()
        #print(fn)
        pl.savefig(plotspath+fn,  bbox_inches='tight', dpi=300)
                
        pl.show()

#sys.exit()
#%%
def replace_names(s):
    sv = [
            ('gamma', '$\gamma$'), ('epsilon','$\\varepsilon$'), ('C', '$C$'),
            ('l1_ratio','$L_1$ ratio'), ('alpha','$\\alpha$'),
            ('l2_penalty','$C_2$'),
            ('thin_plate','T. Plate'),('cubic','Cubic'),
            ('inverse','Inverse'),('quintic','Quintic'),('linear','Linear'),
            ('penalty','$\gamma$'),('max_degree','$q$'),
            ('hidden_layer_sizes', 'HL'),
            ('learning_rate_init', 'LR'),
            ('rbf_width', '$\gamma$'), 
            ('activation_func', '$G$'),
            ('activation', '$\\varphi$'),
            ('n_hidden', 'HL'),
            ('sigmoid', 'Sigmoid'),
            ('inv_multiquadric', 'Inv. Multiquadric'),
            ('multiquadric', 'Multiquadric'),
            ('hardlim', 'HardLim'),('softlim', 'SoftLim'),
            ('tanh', 'Hyp. Tangent'),
            ('gaussian', 'Gaussian'),
            ('identity', 'Identity'),
            ('swish', 'Swish'),
            ('relu', 'ReLU'),
            ('Kappa', '$\kappa$'),
            ('criterion','Criterion'),
            ('learning_rate','LR'),
            ('friedman_mse','MSE'),
            ('reg_lambda','$\lambda$'),
            ('max_depth','Max. Depth'),
            ('min_samples_leaf','Min. Samples Leaf'),
            ('min_samples_split','Min. Samples Split'),
            ('min_weight_fraction_leaf', 'Min. Weig. Fract. Leaf'),
            ('n_estimators', 'No Estimators'),
            ('presort', 'Presort'),
            ('subsample', 'Subsample'),
            ('n_neighbors','$K$'),
            ('positive','Positive Weights'),
            ('max_terms','Max. Terms'),
            ('max_iter','Max. Iter.'),
            ('min_child_weight','Min. Child Weight'),
            ('colsample_bytree','Col. Sample'),
            ('thin_plate', 'thin-plate'),            
            ('interaction_only','Interaction Only'), 
            ('k1','$k_0$'),
            ('sigma', '$\sigma$'), ('beta', '$\\beta$'),
            ('U/u*','$U/u^*$'), 
            ('enable_categorical','Enable Categorical'), 
            #('B','$B$'),('H','$H$'),('U','$U$'),('u*','$u^*$'),
        ]  
    for s1,s2 in sv:
        r=s.replace(str(s1), s2)
        if(r!=s):
            #print r           
            return r
    return r    
        
#%%
#for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):
#    print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))

#%%
parameters=pd.DataFrame()
for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):
  #if e!= ref_estimator:  
    print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
    aux={}
    par = pd.DataFrame(list(df['Parameters']))

    if e=='RBFNN':
        #par['hidden_layer_sizes']=[len(j) for j in par['hidden_layer_sizes']]
        _t=['func',]
        for t in _t:
            par[t] = [replace_names(i) for i in par[t].values]
            
        #print(par); print('\n\n\n\n\n')

    if e=='ANN' or e=='MLP':
        drp=[ 'alpha', 'batch_size', 'beta_1', 'beta_2', 'early_stopping', 'epsilon', 'learning_rate',
               'learning_rate_init', 'max_fun', 'max_iter', 'momentum', 'n_iter_no_change', 'nesterovs_momentum', 'power_t', 'random_state',
               'shuffle', 'solver', 'tol', 'validation_fraction', 'verbose', 'warm_start']
        par.drop(drp, axis=1, inplace=True)
#        par['hidden_layer_sizes']=[len(j) for j in par['hidden_layer_sizes']]
        par['hidden_layer_sizes']=[str(j) for j in par['hidden_layer_sizes']]
        _t=['activation',]
        for t in _t:
            par[t] = [replace_names(i) for i in par[t].values]
    
    if  e=='ELM':
        par.drop(['regressor'], axis=1, inplace=True)
        drp=[ 'activation_args', 'alpha', 'random_state', 'rbf_width', 'user_components']
        par.drop(drp, axis=1, inplace=True)
        _t=['activation_func',]
        for t in _t:
            par[t] = [replace_names(i) for i in par[t].values]

    if  e=='MARS':
        drp=[ 'allow_linear', 'allow_missing', 'check_every', 'enable_pruning', 'endspan', 'endspan_alpha', 'fast_K', 'fast_h',
               'feature_importance_type',  'min_search_points', 'minspan', 'minspan_alpha',  'smooth',
               'thresh', 'use_fast', 'verbose', 'zero_tol']
        par.drop(drp, axis=1, inplace=True)
      

    if  e=='XGB':
        drp=['base_score', 'booster', 'colsample_bylevel', 'colsample_bynode', 'colsample_bytree', 'gamma', 'gpu_id', 'importance_type',
               'interaction_constraints', 'learning_rate', 'max_delta_step',  'min_child_weight', 'missing', 'monotone_constraints',
               'n_jobs', 'num_parallel_tree', 'random_state', 'reg_alpha', 'reg_lambda', 'scale_pos_weight', 'subsample',
               'tree_method', 'validate_parameters', 'verbosity',
              # 'n_estimators', 'max_depth',  'eta',
               ]
        
        par.drop(drp, axis=1, inplace=True)
        par.drop(['objective'], axis=1, inplace=True)
        print(par)

    if  e=='SVR' or e=='SVR-FS':
        drp=[ 'cache_size', 'coef0', 'degree', 'gamma',  'max_iter', 'shrinking', 'tol', 'verbose']
        par.drop(drp, axis=1, inplace=True)
        par__=par
        #par['gamma'] = [0 if a=='scale' else a for a in par['gamma']]
        print(par)
        #sys.exit()

    if  e=='RR' :
        drp=[ 'copy_X', 'fit_intercept', 'max_iter', 'normalize', 'random_state', 'solver', 'tol']
        par.drop(drp, axis=1, inplace=True)
        #par['gamma'] = [0 if a=='scale' else a for a in par['gamma']]
        print(par)
        #sys.exit()

    if  e=='GPR' or e=='GPR-FS':
        par['$\\nu$'] = [float(str(a).split('nu=')[1].split(')')[0]) for a in par['kernel']]
        par['$l$'] = [float(str(a).split('length_scale=')[1].split(', nu')[0]) for a in par['kernel']]
        par.drop(labels=['kernel'], axis=1, inplace=True)
        print(par)
        #sys.exit()

    par=par.melt()
    par['Estimator']=e
    par['Dataset']=d
    par['Phase']=p
    par['Output']=o
    par['variable'] = [replace_names(i) for i in par['variable'].values]

    parameters = parameters.append(par, sort=True)
        
parameters['Parameter']=parameters['variable']
parameters=parameters[parameters['Parameter']!='regressor'] 

#%%
sns.set(font_scale=2)
for (p,e,t,o), df in parameters.groupby(['Phase','Estimator', 'Parameter','Output']):
 if p=='TEST':
  #if e!= ref_estimator:
   #if '-FS' in e:
    print ('='*80+'\n'+t+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
    pl.figure()
    if df['value'].unique().shape[0]<= 10:
        df['value']=df['value'].astype(int,errors='ignore',)
        kwargs={"linewidth": 1, 'edgecolor':None,}
        g = sns.catplot(x='value', col='Dataset', kind='count', data=df, 
                                                col_wrap=6,
                        aspect=0.618, palette=palette_color, **kwargs)
        fmtx='%3d'
        g.set_ylabels('Frequency')#(e+': Parameter '+t)            
        g.fig.tight_layout()

        for ax in g.axes.ravel():
            ax.axes.set_xlabel(e+': Parameter '+t)
            ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90, fontsize=16,)                       
            ax.set_ylim([0, 1.05*ax.get_ylim()[1]])            
            ylabels = ['%3d'% x for x  in ax.get_yticks()]
            ax.set_yticklabels(ylabels,)# fontsize=16,)
            # ax.set_xlabel(e+': '+t, )#fontsize=16,)
            ax.set_xlabel(' ', )

            #ax.set_xlabel('Day'); #ax.set_ylabel(m);

        for ax in g.axes.ravel():
            _h=[]
            for pat in ax.patches:
                _h.append(pat.get_height())
             
            _h=np.array(_h)
            _h=_h[~np.isnan(_h)]
            _h_max = np.max(_h)
            for pat in ax.patches:
                _h= 0 if np.isnan(pat.get_height()) else pat.get_height()
                pat.set_height(_h)
                ax.text(
                        x=pat.get_x() + pat.get_width()/2., 
                        #y=1.04*p.get_height(), 
                        y=0.05*_h_max+pat.get_height(), 
                        s=fmtx % pat.get_height(), 
                        #fontsize=16, 
                        color='black', ha='center', 
                        va='bottom', rotation=0, weight='bold',
                       )
        #pl.legend( loc=10, borderaxespad=0., fontsize=16, ) 
        #pl.show()
    else:
        df['value']=df['value'].astype(float,errors='ignore',)    
        kwargs={"linewidth": 1, 'aspect':0.618,}
        g = sns.catplot(x='value', y='Dataset', kind='box', data=df, notch=0,
                        orient='h', palette=palette_color, **kwargs, )
        xmin, xmax = g.ax.get_xlim()
        g.ax.set_xlim(left=0, right=xmax)
        #g.ax.set_xlabel(d+' -- '+e+': Parameter '+t, fontsize=16,)
        g.ax.set_xlabel(e+': Parameter '+t, )#fontsize=16,)
        g.ax.set_ylabel(d, rotation=90)
        g.ax.set_ylabel(None)#fontsize=16,)
        g.fig.tight_layout()
        #g.fig.set_figheight(4.00)
        #pl.xticks(rotation=45)
        #g.ax.set_ylabel(e+': Parameter '+t)
        
#    min, xmax = g.ax.get_xlim()
#    g.ax.set_xlim(left=0, right=xmax)
#    g.fig.tight_layout()
#    g.fig.set_figheight(0.50)
#    pl.xticks(rotation=45)
    fn = basename+'300dpi_comparison_datasets'+'_parameters_'+'__'+e+'__'+t+'__'+p+'.png'
    fn = re.sub('\^','', re.sub('\$','',fn))
    fn = re.sub('\(','', re.sub('\)','',fn))
    fn = re.sub(' ','_', re.sub('\/','',fn))
    fn = re.sub('\\\\','', re.sub('x.','x',fn))
    fn = re.sub('-','_', re.sub('\/','',fn)).lower()
    fn = fn.lower()
    #print(fn)
    pl.savefig(plotspath+fn, transparent=True, optimize=True,
               bbox_inches='tight', 
               dpi=300)
    pl.show()
sns.set(font_scale=1)
#sys.exit()    
#%%

# sensitivity analysis
# https://github.com/SALib/SALib/blob/master/examples/morris/morris.py

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import  XGBRegressor
from util.ELM import  ELMRegressor, ELMRegressor
#from util.MLP import MLPRegressor as MLPR
#from util.RBFNN import RBFNNRegressor, RBFNN
#from util.LSSVR import LSSVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from pyearth import Earth as MARS

from read_data_tanzania import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MaxAbsScaler

dataset=read_data_tanzania(model = 1)

feature_names    = dataset['feature_names'   ]
X_train          = dataset['X_train'         ]
X_test           = dataset['X_test'          ]
y_train          = dataset['y_train'         ]
y_test           = dataset['y_test'          ]
n_features       = dataset['n_features'      ]


v_ref = 'RRMSE'
v_aux = 'KGE'
k = -1
uncertainty_tab=[]
for (e,d,o,p,), df in C.groupby(['Estimator','Dataset','Output','Phase',]):
 if p!='TRAIN':
  #if e!= ref_estimator:  
   #if '-FS' in e:
    #print ('='*80+'\n'+p+' - '+d+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
    
    k = df[v_ref].idxmin()
    aux = df.loc[k] 
    param = aux['Parameters'].copy()
    
    X_train_ = pd.DataFrame(data=X_train, columns=feature_names)
    X_test_  = pd.DataFrame(data=X_test, columns=feature_names)
    
    #if e=='ELM':
        #_alpha = param['l2_penalty']
        #param.pop('l2_penalty')
        #regressor = None if _alpha<1e-4 else Ridge(alpha=_alpha,random_state=aux['Seed'])
        #param['regressor']=regressor
    
    estimators={
        'SVR':SVR(),
        'ELM':ELMRegressor(random_state=aux['Seed']),
        'MLP':MLPRegressor(random_state=aux['Seed']),
        'RR':Ridge(random_state=aux['Seed']),
        #'EN':ElasticNet(),
        #'RBFNN':RBFNNRegressor(),
        #'LSSVR':LSSVR(),
        'XGB':XGBRegressor(),
        'MARS':MARS(),
        #'GPR':GaussianProcessRegressor(random_state=aux['Seed'], optimizer=None, normalize_y=True),
        }

    # pl.figure()
    # pl.plot(X_test_['$B$']/X_test_['$H$'], aux['y_pred'], 'ro')
    # pl.plot(X_test_['$B$']/X_test_['$H$'], y_test.T, 'bo')
    # pl.title(e+' - '+d)
    # pl.show()
    
    active_features = aux['Active Variables']
    active_features = [s.replace(' ','') for s in active_features.split(',')]
    
    
    X_train_ = X_train_[active_features].values
    X_test_  = X_test_[active_features].values   
    n_features = X_train_.shape[1]

    #scaler=MaxAbsScaler()
    #scaler.fit(X_train_)    
    #X_train_ = scaler.transform(X_train_)
    #X_test_  = scaler.transform(X_test_)
    
    
    #reg = SVR() if 'SVR' in e else GaussianProcessRegressor(optimizer=None)
    reg=estimators[e]

    for pr in ['scaler', 'k1']:        
        if pr in param.keys():
            param.pop(pr) 
    
    reg.set_params(**param)    
    reg.fit(X_train_, y_train.T.ravel())
    
    n_outcomes=250000
    data=np.random.uniform( low=X_test_.min(axis=0), high=X_test_.max(axis=0), size=(n_outcomes, X_test_.shape[1]) )
    #data=np.random.normal( loc=X_test_.mean(axis=0), scale=X_test_.std(axis=0), size=(n_outcomes, X_test_.shape[1]) )
    predict = reg.predict(data)
    median = np.median(predict)
    mad=np.abs(predict - median).mean()
    uncertainty = 100*mad/median
    print(e,d, median, mad, n_features, uncertainty/n_features, uncertainty)
    dc={'Model':e, 'Case':d, 'No. features':n_features, 'Median':median, 
        'MAD':mad, 'Uncertainty':uncertainty, v_ref:aux[v_ref]}
    uncertainty_tab.append(dc)


#%%

uncertainty_tab = pd.DataFrame(uncertainty_tab)
fn='uncertainty_table__mc'
cpt='Caption to be inserted.'
fig=pl.figure(figsize=(4,4))
#for tab, unc_tab in uncertainty_tab.groupby(('Case')):

unc_tab= uncertainty_tab
unc_tab.to_latex(buf=plotspath+fn+'.tex', index=False, escape=False, label=fn, caption=cpt, column_format='r'*df_table.shape[1], float_format="%.1f")
print(unc_tab)

unc_tab=unc_tab[unc_tab['Case']!='Case 1']
unc_tab.index=[i for i in range(unc_tab.shape[0])]

#unc_tab['case']=['FS' if 'FS' in t else 'C'+s.split(' ')[1] for s,t in zip(unc_tab['Case'],unc_tab['Model'])]
unc_tab['case']=['FS' if 'FS' in t else s for s,t in zip(unc_tab['Case'],unc_tab['Model'])]

p1=sns.relplot(x='Uncertainty', y=v_ref, hue='Model', size='No. features', 
               #style='Case',
               sizes=(50, 500),
               #size_norm=(1,len(unc_tab['No. features'].unique())),
               data=unc_tab, alpha=0.9)
for line in range(0,unc_tab.shape[0]):
     p1.ax.text(unc_tab['Uncertainty'][line]+0, unc_tab[v_ref][line], 
             unc_tab['case'][line], 
             horizontalalignment='center',
             verticalalignment='center',
             size='small', color='black', 
             weight='semibold', rotation=0)

         
fn = basename+'300dpi_comparison_uncertainty_rmse'+'.png'
fn = re.sub('\^','', re.sub('\$','',fn))
fn = re.sub('\(','', re.sub('\)','',fn))
fn = re.sub(' ','_', re.sub('\/','',fn))
fn = re.sub('\\\\','', re.sub('x.','x',fn))
fn = re.sub('-','_', re.sub('\/','',fn)).lower()
fn = fn.lower()
#print(fn)
pl.savefig(plotspath+fn, transparent=True, optimize=True,
           bbox_inches='tight', 
           dpi=300)     

pl.show()


#sys.exit()    
#%%
v_ref = 'RMSE'
v_aux = 'KGE'
k = -1
unc_tab=[]
for (e,d,o,p,), df in C.groupby(['Estimator','Dataset','Output','Phase',]):
 if p!='TRAIN':
  #if e!= ref_estimator:  
   #if '-FS' in e:
    #print ('='*80+'\n'+p+' - '+d+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
    
    k = df[v_ref].idxmin()
    aux = df.loc[k] 
    
    #ix=aux['y_pred']>0; unc_p,unc_t = aux['y_pred'][ix], aux['y_true'][ix]
    #unc_e = (np.log10(unc_p) - np.log10(unc_t))
    #unc_m=unc_e.mean()
    #unc_s = np.sqrt(sum((unc_e - unc_m)**2)/(len(unc_e)-1))
    #pei95=fmt(10**(-unc_m-1.96*unc_s))+' to '+fmt(10**(-unc_m+1.96*unc_s))
    #print(p+' - '+d+' - '+e+' - '+str(o), fmt(unc_m), fmt(unc_s), pei95 )
    
    
    #ix=aux['y_pred']>0; unc_p,unc_t = aux['y_pred'][ix], aux['y_true'][ix]
    #unc_e = (np.log10(unc_p) - np.log10(unc_t))
    unc_p,unc_t = aux['y_pred'], aux['y_true']
    unc_e = ((unc_p) - (unc_t))
    unc_m=unc_e.mean()
    unc_s = np.sqrt(sum((unc_e - unc_m)**2)/(len(unc_e)-1))
    pei95=fmt((unc_m-1.96*unc_s))+' to '+fmt((unc_m+1.96*unc_s))
    #pei95=fmt(10**(-unc_m-1.96*unc_s))+' to '+fmt(10**(-unc_m+1.96*unc_s))
    #print(p+' - '+d+' - '+e+' - '+str(o), fmt(unc_m), fmt(unc_s), pei95 )
    
    #pl.figure()
    #sns.distplot(unc_e, ); pl.axvline(unc_m); 
    #pl.axvline((unc_m-1.96*unc_s)); pl.axvline((+unc_m+1.96*unc_s))
    #tit=e+' ('+d+') Average error = '+fmt(unc_m)
    #pl.title(tit)
    #pl.show()
    
    sig = '+' if unc_m > 0 else ''
    dc={'Model':e, 'Case':d, 'MPE':sig+fmt(unc_m), 'WUB':'$\pm$'+fmt(unc_s), 'PEI95':pei95}
    unc_tab.append(dc)


unc_tab = pd.DataFrame(unc_tab)
fn='uncertainty_table__models'
cpt='Caption to be inserted.'
unc_tab.to_latex(buf=plotspath+fn+'.tex', index=False, escape=False, label=fn, caption=cpt, column_format='r'*df_table.shape[1])
print(unc_tab)
    
#sys.exit()
#%%
stations = C['Dataset'].unique()
stations.sort()
colors={}
# kolors=[]
# import colorsys

# for i in range(30):
#     kolors.append(list(np.random.choice(range(255), size=3)))
# # kolors=['r', 'darkgreen', 'b', 'm', 'c','y', 'olive',  'darkorange', 'brown', 'darkslategray', ]

kolors = sns.color_palette(None, 64)

for i, j in zip(stations,kolors): 
    colors[i]=j

for type_plot in ['taylor', 'target']:
    for (d,o,p,), df in C.groupby(['Dataset','Output','Phase',]):
     if p!='TRAIN':
      #if e!= ref_estimator:  
       #if '-FS' in e:
        print ('='*80+'\n'+p+' - '+d+' - '+str(o)+'\n'+'='*80+'\n')
        ref=df.iloc[0]['y_true']    
        est=df['Estimator'].unique();
        est = [x+' ('+d+')' for x in est]
        label=dict(zip(est,kolors[:len(est)]))
        
        k=0
        pl.figure(figsize=[7.5,7.5])
        for e, df1 in df.groupby('Estimator'):    
            overlay='on' if k>0 else 'off'
            taylor_stats=[{'sdev':np.std(ref), 'crmsd':0, 'ccoef':1,  
                           'label':'Observation', 'bias':1, 'rmsd':0}]
            
            for i in range(len(df1)):
                pred=df1.iloc[i]['y_pred']
                ts=sm.taylor_statistics(pred,ref,'data')
                taylor_stats.append({'sdev':ts['sdev'][1], 'crmsd':ts['crmsd'][1], 
                                     'ccoef':ts['ccoef'][1],  'label':e, 
                                     'bias':sm.bias(pred, ref),
                                     'rmsd':sm.rmsd(pred, ref),                                 
                                     })
    
            taylor_stats = pd.DataFrame(taylor_stats)
            if type_plot=='taylor':
                sm.taylor_diagram(taylor_stats['sdev'].values, 
                              taylor_stats['crmsd'].values, 
                              taylor_stats['ccoef'].values,
                              markercolor =kolors[k], alpha = 0.0,
                              markerSize = 12, 
                              colSTD='k', colRMS='darkgreen', colCOR='darkblue',
                              overlay = overlay, 
                              markerLabel = label)
            elif type_plot=='target':
                sm.target_diagram(taylor_stats['bias'].values, 
                              taylor_stats['crmsd'].values, 
                              taylor_stats['rmsd'].values,
                              markercolor =kolors[k], alpha = 0.0,
                              markerSize = 4, circleLineSpec = 'k--',
                              #circles = [1000, 2000, 3000],
                              overlay = overlay, markerLabel = label)
            else:
                sys.exit('Plot type '+type_plot+' uNdefined')
                
            k+=1
        
    
             
        fn = basename+'300dpi_'+type_plot+'_diagram'+'__'+d+'__'+p+'__'+o+'.png'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn))
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('\\\\','', re.sub('x.','x',fn))
        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
        fn = fn.lower()
        print(fn)
        pl.savefig(plotspath+fn, transparent=True, optimize=True, bbox_inches='tight', dpi=300)        
        pl.show()
    
#sys.exit()
#%%
stations = C['Dataset'].unique()
stations.sort()
# colors={}
# for i, j in zip(stations,['r', 'darkgreen', 'b', 'm', 'c','y', 'olive',  'darkorange', 'brown', 'darkslategray', 'g','deepskyblue','coral',]): 
#     colors[i]=j
   
v_ref = 'RMSE'
v_aux = 'KGE'
k = -1
for (e,d,o,p,), df in C.groupby(['Estimator','Dataset','Output','Phase',]):
 if p!='TRAIN':
  #if e!= ref_estimator:  
   #if '-FS' in e:
    print ('='*80+'\n'+p+' - '+d+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
    
    k = df[v_ref].idxmin()
    aux = df.loc[k] 
    pl.figure(figsize=(3,3))
    ax = sns.regplot(x="y_true", y="y_pred", data=aux, ci=0.95, 
                     line_kws={'color':'black'}, 
                     scatter_kws={'alpha':0.85, 'color':colors[d], 's':20},
                     label='WI'+' = '+fmt(aux['WI']),
                     #label='R'+' = '+fmt(aux['R']),
                     )
    #ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(left    = aux['y_true'].min(), right    = 1.1*aux['y_true'].max() )
    ax.set_ylim(bottom  = aux['y_true'].min(), top      = 1.1*aux['y_true'].max() )
    ax.set_title(d+' -- '+e+' ({\\bf '+p+'}) '+'\n'+v_ref+' = '+fmt(df[v_ref][k]))
    ax.set_title(d+' -- '+e+' ('+p+') '+'\n'+v_ref+' = '+fmt(df[v_ref][k])+', '+v_aux+' = '+fmt(df[v_aux][k]))
    ax.set_xlabel('Measured   '+aux['Output'])
    ax.set_ylabel('Predicted  '+aux['Output'])
    ax.set_yticklabels(labels=ax.get_yticks(), rotation=0)
    ax.set_xticklabels(labels=ax.get_xticks(), rotation=0)
    ax.set_aspect(1)
    ax.legend(frameon=False, markerscale=0, loc=0)
    fn = basename+'300dpi_scatter'+'_best_model_'+'__'+e+'__'+d+'__'+p+'.png'
    fn = re.sub('\^','', re.sub('\$','',fn))
    fn = re.sub('\(','', re.sub('\)','',fn)) 
    fn = re.sub(' ','_', re.sub('\/','',fn))
    fn = re.sub('\\\\','', re.sub('x.','x',fn))
    fn = re.sub('-','_', re.sub('\/','',fn)).lower()
    fn = fn.lower()
    #print(fn)
    pl.savefig(plotspath+fn, transparent=True, optimize=True, bbox_inches='tight', dpi=300)
    
    pl.show()
#%%
# variaton plots, hydrograms
# https://seaborn.pydata.org/generated/seaborn.lineplot.html
stations = C['Dataset'].unique()
stations.sort()
# colors={}
# for i, j in zip(stations,['r', 'darkgreen', 'b', 'm', 'c','y', 'olive',  'darkorange', 'brown', 'darkslategray','g','deepskyblue','coral', ]): 
#     colors[i]=j
    
v_ref = 'RRMSE'
v_aux = 'KGE'
k = -1

for (d,o,p,), df1 in C.groupby(['Dataset','Output','Phase',]):
 if p!='TRAIN':
  #if e!= ref_estimator:  
   #if '-FS' in e:
    print ('='*80+'\n'+p+' - '+d+' - '+' - '+str(o)+'\n'+'='*80+'\n')
    
    tab_best_models=pd.DataFrame()
    for e, df in df1.groupby('Estimator'):
        k = df[v_ref].idxmin()
        aux = df.loc[k] 
        date_range_test = dataset['date_range'][-y_test.shape[-1]:]
        xrange=pd.date_range(start=date_range_test[0], periods=len(aux['y_true']), freq="d", normalize=True)
        aux['Month']    = xrange
        aux['Observed'] = aux['y_true']
        aux['Predicted']= aux['y_pred']
        print(e)
        #tab_best_models['Case']=d
        tab_best_models['Month']=aux['Month']
        tab_best_models['Observed']=aux['y_true']
        tab_best_models[e]=aux['y_pred']
        tab_best_models.index=aux['Month']
        
        fn='table_best_models_'+'__'+d+'__'+p+'.csv'; fn=re.sub(' ','_',fn).lower()
        tab_best_models.to_csv(path_or_buf=plotspath+fn, sep=';', index=False)
        tab_best_models.to_excel(plotspath+fn.replace('.csv', '.xlsx'), sheet_name='Station 2 best results', index=False)
        
        aux3=pd.DataFrame(np.c_[aux['y_true'],aux['y_pred'],], columns=['Observed', e], 
                     index=xrange)   
        
        pl.figure(figsize=(12,4))
        id_var='Year'
        aux3[id_var]=aux3.index
        target='GHI'
    
        aux4=aux3.melt(id_vars=id_var)
        aux4['Estimator']=aux4['variable']
        aux4[target]=aux4['value']
        ax=sns.lineplot(x=id_var, y=target, hue='Estimator',style='Estimator', data=aux4, markers=True, dashes=False)
        ax.legend(frameon=True, markerscale=0, loc=2, bbox_to_anchor=(1.01, 1.0))
        ax.set_title(d+':'+e)
        ax.set_title(d+' -- '+e+' ('+p+') '+' -- '+v_ref+' = '+fmt(df[v_ref][k])+', '+v_aux+' = '+fmt(df[v_aux][k]))
        ax.grid()

        fn = basename+'300dpi_variation_'+'_best_model_'+e+'__'+d+'__'+p+'.png'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn)) 
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('\\\\','', re.sub('x.','x',fn))
        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
        fn = fn.lower()
        #print(fn)
        pl.savefig(plotspath+fn, transparent=False, optimize=True, bbox_inches='tight', dpi=300)
        
        pl.show()
#%%
for (d,o,p,), df1 in C.groupby(['Dataset','Output','Phase',]):
 if p!='TRAIN':
  #if e!= ref_estimator:  
   #if '-FS' in e:
    print ('='*80+'\n'+p+' - '+d+' - '+' - '+str(o)+'\n'+'='*80+'\n')
    
    for e, df in df1.groupby('Estimator'):
        tab_model=pd.DataFrame()
        for k in range(len(df)):
            aux = df.iloc[k]
            #xrange=pd.date_range(start="2001-01-01", periods=len(aux['y_true']), freq="m", normalize=True)
            aux['Month']    = xrange
            aux['Observed'] = aux['y_true']
            aux['Predicted']= aux['y_pred']
            #tab_best_models['Case']=d
            #tab_best_models['Month']=aux['Month']
            tab_model['Observed']=aux['y_true']
            tab_model['Run '+str(k)]=aux['y_pred']
            tab_model.index=aux['Month']
            
        fn='table_model_'+'__'+d+'__'+e+'.csv'; fn=re.sub(' ','_',fn).lower()
        #tab_model.to_csv(path_or_buf=fn, sep=';', index=False)
        tab_model.to_excel(plotspath+fn.replace('.csv', '.xlsx'), sheet_name='Station 2 best results', index=False)
        
#%% 
    # pl.figure(figsize=(12,4))
    # id_var='Year'

    # tab_best_models[id_var]=tab_best_models.index
    # # target='$Q_t$'
    # target='$GHI$'
    
    # #date_range_test = dataset['date_range'][-y_test.shape[-1]:]
    
    # aux2=tab_best_models.melt(id_vars=id_var)
    # aux2['Estimator']=aux2['variable']
    # aux2[target]=aux2['value']
    # # aux2[id_var]=pd.to_datetime(aux2[id_var])
    # # aux2[target]
    # # aux2.set_index(id_var)
    # ax=sns.lineplot(x=id_var, y=target, hue='Estimator', data=aux2,)
    # #ax.set_xlim(bottom  = aux2[id_var].min(), top      = 1.0*aux2[id_var].max() )
    # #ax.set_ylim(bottom  = aux2[target].min(), top      = 1.0*aux2[target].max() )
    # ax.legend(frameon=True, markerscale=0, loc=2, bbox_to_anchor=(1.01, 1.0))
    # ax.set_title(d)
    # ax.grid()

    # fn = basename+'300dpi_variation_comparison_'+'_best_model_'+'__'+d+'__'+p+'.png'
    # fn = re.sub('\^','', re.sub('\$','',fn))
    # fn = re.sub('\(','', re.sub('\)','',fn)) 
    # fn = re.sub(' ','_', re.sub('\/','',fn))
    # fn = re.sub('\\\\','', re.sub('x.','x',fn))
    # fn = re.sub('-','_', re.sub('\/','',fn)).lower()
    # fn = fn.lower()
    # #print(fn)
    # pl.savefig(plotspath+fn, transparent=False, optimize=True, bbox_inches='tight', dpi=300)
    
    # pl.show()
    # print('fim')
#%%
#for (p,e,d,o), df in C.groupby(['Phase','Estimator','Dataset','Output']):
# if p!='TRAIN':
#  if e!= ref_estimator:  
#    print ('='*80+'\n'+d+' - '+e+' - '+str(o)+'\n'+'='*80+'\n')
#    aux={}
#    par = pd.DataFrame(list(df['Parameters']))
#    if e=='ANN':
#        par['Layer Sizes']=[len(j) for j in par['hidden_layer_sizes']]
#        #g=sns.catplot(hue='activation', x='Layer Sizes', data=par, kind='count', aspect=0.618)
#        g=sns.catplot(x='activation', hue='Layer Sizes', data=par, kind='count', aspect=0.618)
#        for p in g.ax.patches:
#                g.ax.annotate('{:.0f}'.format(p.get_height()),
#                            (p.get_x()*1.0, p.get_height()+.1), fontsize=12)
#        
#    par.columns = [replace_names(i) for i in par.columns]
#    if e!= 'ANN':
#     for t in par: 
#        print(d,e,o,t,type(par[t]), par[t].dtype)       
#        if par[t].dtype=='float64' or par[t].dtype=='int64':
#            #pl.figure(figsize=(1,4))
#            g = sns.catplot(x=t, data=par, kind='box', orient='h', notch=0, )#palette='Blues_r', )# width=0.1)
#            xmin, xmax = g.ax.get_xlim()
#            g.ax.set_xlim(left=0, right=xmax)
#            g.ax.set_xlabel(d+' -- '+e+': Parameter '+t)
#            g.fig.tight_layout()
#            g.fig.set_figheight(0.50)
#            pl.xticks(rotation=45)
#            #g.ax.set_title(d+' - '+e)
#            #xlabels = ['{:,.2g}'.format(x) for x in g.ax.get_xticks()/1000]
#            #g.set_xticklabels(xlabels)
#            pl.show()
#        if par[t].dtype=='int64':
#            par[t] = [ str(i) if type(i)==list or type(i)==tuple or i==None else i for i in par[t] ]
#            #par[t] = [replace_names(j) for j in par[t]]
#            g = sns.catplot(x=t, data=par, kind='count', palette=palette_color, aspect=0.618)
#            ymin, ymax = g.ax.get_ylim()
#            g.ax.set_ylim(bottom=0, top=ymax*1.1)
#            pl.ylabel(u'Frequency')
#            #if t=='n_hidden' or 'activation_func':    
#            pl.xticks(rotation=90)               
#            for p in g.ax.patches:
#                g.ax.annotate('{:.0f}'.format(p.get_height()),
#                            (p.get_x()*1.0, p.get_height()+.1), fontsize=16)
#                
#            pl.show()
#            
##        elif type(par[t].values[0])==str: 
##            par[t] = [ str(i) if type(i)==list or type(i)==tuple or i==None else i for i in par[t] ]
##            par[t] = [replace_names(j) for j in par[t]]
##            g = sns.catplot(x=t, data=par, kind='count', palette=palette_color, aspect=0.618)
##            ymin, ymax = g.ax.get_ylim()
##            g.ax.set_ylim(bottom=0, top=ymax*1.1)
##            pl.ylabel(u'Frequency')
##            #if t=='n_hidden' or 'activation_func':    
##            pl.xticks(rotation=90)               
##            for p in g.ax.patches:
##                g.ax.annotate('{:.0f}'.format(p.get_height()),
##                            (p.get_x()*1.0, p.get_height()+.1), fontsize=12)
#        else:
#            pass
#
#        #pl.xlabel('')
#        #pl.title(e+''+': '+replace_names(t), fontsize=16)
#       # pl.show()
            
#%%
#for (e,o), df in C.groupby(['Estimator','Output']):
#  if e!=ref_estimator:  
#    print ('='*80+'\n'+e+' - '+o+'\n'+'='*80+'\n')
#    aux={}
#    par = pd.DataFrame(list(df['Parameters']))
#    par=par.melt()
#    par['variable'] = [replace_names(i) for i in par['variable'].values]
#    #print(par)     
#    for p1, df5 in par.groupby('variable'):
#        if type(df5['value'].values[0])!=str and type(df5['value'].values[0])!=bool:
#            kwargs={'capsize':0.05, 'ci':'sd', 'errwidth':1, 'dodge':True, 'aspect':2.5}
#            fig=sns.catplot(x='variable', y='value', data=df5, kind='bar', **kwargs)
#            fmt='%1.0f' if type(df5['value'].values[0])==int else '%2.3f'
#            #fmt='%1.0f' if p1=='HL' else fmt
#            for ax in fig.axes.ravel():
#                for p in ax.patches:
#                    ax.set_ylabel(p1); ax.set_xlabel('Day')
#                    ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
#                    ax.text(
#                                p.get_x() + p.get_width()/3., 
#                                1.001*p.get_height(), 
#                                fmt % p.get_height(), 
#                                fontsize=12, color='black', ha='center', 
#                                va='bottom', rotation=90, #weight='bold',
#                            )
#        else:
#            kwargs={'dodge':True, 'aspect':0.618}
##            fig=sns.catplot(data=df5,x='value', kind='count', **kwargs)   
##            for ax in fig.axes.ravel():
##                #ax.set_ylim([0, 1.06*ax.get_ylim()[1]])
##                #t=str(ax.get_title()); ax.set_ylabel(t)
##                #ax.set_title('')
##                s1,s2= ax.get_title().split('|')
##                ax.set_title(s2); ax.set_ylabel(s1) ; ax.set_xlabel('') 
##                for p in ax.patches:
##                    p.set_height( 0 if np.isnan(p.get_height()) else p.get_height() )
##                    ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
##                    ax.text(
##                                p.get_x() + p.get_width()/2., 
##                                1.001*p.get_height(), 
##                                '%1.0f' % p.get_height(), 
##                                fontsize=16, color='black', ha='center', 
##                                va='bottom', rotation=0, #weight='bold',
##                            )
#
#
##        pl.xlabel('Day'); pl.ylabel(p1) 
##        pl.title(s)
#        
##        fn = basename+'_parameter_'+str(p1)+'_estimator_'+reg.lower()+'_'+'_distribution'+'.png'
##        #fig = ax.get_figure()
##        pl.savefig(re.sub('\\\\','',re.sub('\^','', re.sub('\$','',fn) ) ),  bbox_inches='tight', dpi=300)
##
##        pl.show()
#    
#%%
n=[]
for a in C['Active Variables']:
    b=a.replace(' ','').split(',')
    b=[replace_names(i) for i in b]
    n.append(', '.join(b))

C['Active Variables']=n    
#%%    
#from itertools import combinations
#kind='count'
#for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):  
#  if p!= 'TRAIN':  
#    if e!= ref_estimator:
#      if '-FS' in e:         
#        print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
#        print('Number of sets: ','for ', e, ' = ', df['Active Variables'].unique().shape)
#        kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':1.618, 'legend':None, } if kind=='count' else None
#        n=[]
#        for a in df['Active Variables']:
#            for i in a.replace(' ','').split(','):
#                #v=i.replace(' ','')
#                #n.append(replace_names(v))
#                n.append(i)
#        
#        n=[]
#        for a in df['Active Variables']:
#            b=a.replace(' ','').split(',')
#            #b=[replace_names(i) for i in b]
#            for k in range(len(b)):
#                for i in combinations(b,k+1):
#                    n.append({'set':', '.join(i), 'order':k+1, 'count':1})
#
#        P=pd.DataFrame(data=n)
#        Q=P.groupby(['set']).agg(np.sum)
#        Q.sort_values(by='count', axis=0, ascending=False, inplace=True)
#
#        for order, dfo in P.groupby(['order']):
#            g=sns.catplot(x='set', data=dfo, kind='count',
#                          order = dfo['set'].value_counts().index,
#                          aspect=3,
#                          #**kwargs,
#                          )
#            g.ax.set_xticklabels(labels=g.ax.get_xticklabels(),rotation=90)
#            g.ax.set_title(e+': Order = '+str(order))
#            pl.show()
                    
#%%
kind='count'
for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):  
  if p!= 'TRAIN':  
    #if e!= ref_estimator:
      if '-FS' in e:
        print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
        kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':0.618, 'legend':None, } if kind=='count' else None
        n=[]
        for a in df['Active Variables']:
            for i in a.replace(' ','').split(','):
                #v=i.replace(' ','')
                #n.append(replace_names(v))
                n.append(i)
                
        #n = [replace_names(i) for i in n]
        P=pd.DataFrame(data=n, columns=['Variable'])
        P['count']=1
        g=sns.catplot(x='Variable', data=P, kind='count',
                      order = P['Variable'].value_counts().index,
                      **kwargs,
                      )
        #g.set_xticklabels(labels=g.ax.get_xticklabels(), rotation=90)
        g.ax.legend(title=e)#labels=[e])
        fmtx='%d'
        for ax in g.axes.ravel():
            ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
            ax.set_ylabel('Count')
            ax.set_xlabel('Feature')
            if kind=='count':
                #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
                #ax.set_xlabel('Day'); #ax.set_ylabel(m);
                _h=[patch.get_height() for patch in ax.patches]
                 
                _h=np.array(_h)
                _h=_h[~np.isnan(_h)]
                _h_max = np.max(_h)
                for patch in ax.patches:
                    _h= 0 if np.isnan(patch.get_height()) else patch.get_height()
                    patch.set_height(_h)                
                    ax.text(
                            x=patch.get_x() + patch.get_width()/2., 
                            #y=1.04*patch.get_height(), 
                            y=0.02*_h_max+patch.get_height(), 
                            s=fmtx % patch.get_height() if patch.get_height()>0 else None, 
                            #fontsize=16, 
                            color='black', ha='center', 
                            va='bottom', rotation=90, weight='bold',
                            )
            
        fn = basename+'300dpi_active_features_distribution_'+e+'__'+kind+'.png'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn))
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
        #print(fn)
        pl.savefig(plotspath+fn,  bbox_inches='tight', dpi=300)
                
        pl.show()
        #--            
#%%
#kind='count'
#for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):  
#  if p!= 'TRAIN':  
#    if e!= ref_estimator:
#      if '-FS' in e:
#        print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
#        kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':1.618, 'legend':None, } if kind=='count' else None
#        n=[]
#        for a in df['Active Variables']:
#            for i in a.replace(' ','').split(','):
#                #v=i.replace(' ','')
#                #n.append(replace_names(v))
#                n.append(i)
#                
#        #n = [replace_names(i) for i in n]
#        P=pd.DataFrame(data=n, columns=['Variable'])
#        P['count']=1
#        g=sns.catplot(y='Variable', data=P, kind='count',
#                      order = P['Variable'].value_counts().index,
#                      **kwargs,
#                      )
#        #g.set_xticklabels(labels=g.ax.get_xticklabels(), rotation=90)
#        g.ax.legend(title=e)#labels=[e])
#        fmtx='%d'
#        for ax in g.axes.ravel():
#            #ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
#            ax.set_xlabel('Count')
#            ax.set_ylabel('Feature')
#            if kind=='count':
#                #ax.set_xlim([0, 1.21*ax.get_xlim()[1]])
#                #ax.set_xlabel('Day'); #ax.set_ylabel(m);
#                _h=[patch.get_width() for patch in ax.patches]
#                 
#                _h=np.array(_h)
#                _h=_h[~np.isnan(_h)]
#                _h_max = np.max(_h)
#                for patch in ax.patches:
#                    __h= 0 if np.isnan(patch.get_width()) else patch.get_width()
#                    patch.set_width(__h)                
#                    ax.text(
#                            y=patch.get_y() + 0.7*patch.get_height(), 
#                            #y=1.04*patch.get_height(), 
#                            x=_h_max*0.02+patch.get_width(), 
#                            s=fmtx % patch.get_width() if patch.get_width()>0 else None, 
#                            #fontsize=16, 
#                            color='black', ha='left', 
#                            va='bottom', rotation=0, weight='bold',
#                            )
#            
#        fn = basename+'300dpi_active_features_distribution_'+e+'__'+kind+'_h.png'
#        fn = re.sub('\^','', re.sub('\$','',fn))
#        fn = re.sub('\(','', re.sub('\)','',fn))
#        fn = re.sub(' ','_', re.sub('\/','',fn))
#        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
#        #print(fn)
#        pl.savefig(fn,  bbox_inches='tight', dpi=300)
#                
#        pl.show()
        #--            
#%%
kind='box'
for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):  
  if p!= 'TRAIN':  
    #if e!= ref_estimator:
      if '-FS' in e:
        print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
        kind='count'
        kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':1.2, 'legend':None, } if kind=='count' else None
        g=sns.catplot(x='Active Variables', col='Estimator', hue='Phase', data=df, 
                               #order=ds, hue_order=hs, 
                               kind=kind, sharey=False, 
                               order = df['Active Variables'].value_counts().index,
                               #col_wrap=2, palette=palette_color_1,
                               **kwargs,
                               )                    
        #g.despine(left=True)
        #g.ax.legend(title=e)#labels=[e])
        fmtx='%d'
        for ax in g.axes.ravel():
            ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
            ax.set_xlabel('Active Features')
            ax.set_ylabel('Count')
            if kind=='count':
                #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
                #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
                #ax.set_xlabel('Day'); #ax.set_ylabel(m);
                _h=[]
                for patch in ax.patches:
                    _h.append(patch.get_height())
                 
                _h=np.array(_h)
                _h=_h[~np.isnan(_h)]
                _h_max = np.max(_h)
                for patch in ax.patches:
                    _h= 0 if np.isnan(patch.get_height()) else patch.get_height()
                    patch.set_height(_h)                
                    ax.text(
                            x=patch.get_x() + patch.get_width()/2., 
                            #y=1.04*patch.get_height(), 
                            y=0.02*_h_max+patch.get_height(), 
                            s=fmtx % patch.get_height() if patch.get_height()>0 else None, 
                            #fontsize=16, 
                            color='black', ha='center', 
                            va='bottom', rotation=90, weight='bold',
                            )
            #pl.legend(bbox_to_anchor=(-0.00, 1.2), loc=10, borderaxespad=0., ncol=n_estimators, fontsize=16, ) 
            #pl.legend(bbox_to_anchor=(0.80, 0.8), loc=10, ncol=n_estimators, borderaxespad=0.,)
            
        fn = basename+'300dpi_active_features_sets'+'_'+e+'__'+kind+'.png'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn))
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
        #print(fn)
        pl.savefig(plotspath+fn,  bbox_inches='tight', dpi=300)
                
        pl.show()

#%%
kind='box'
for (p,d,o), df in C.groupby(['Phase','Dataset','Output']):
  #df = df[df['Estimator']!=ref_estimator]  
  if p!= 'TRAIN':
      if len(df)>0:
        print (p+'\t'+d+'\t\t'+'\t'+str(len(df)))
        kind='count'
        kwargs={'edgecolor':"k", 'alpha':0.95, 'dodge':True, 'aspect':1.2, 'legend':None, } if kind=='count' else None
        g=sns.catplot(x='Active Variables', col='Estimator', hue='Phase', data=df, 
                               #order=ds, hue_order=hs, 
                               kind=kind, sharey=False, 
                               order = df['Active Variables'].value_counts().index,
                               #col_wrap=2, palette=palette_color_1,
                               **kwargs,
                               )                    
        #g.despine(left=True)
        #g.ax.legend(title=e)#labels=[e])
        fmtx='%d'        
        for ax in g.axes.ravel():
            xticklabels=[]
            for xticklabel, patch in zip(ax.get_xticklabels(),ax.patches):
                print(patch.get_height(), xticklabel)
                if patch.get_height()>0:
                    xticklabels.append(xticklabel)
            
            ax.set_xticklabels(labels=xticklabels,rotation=90)
            ax.set_xlabel('Active Features')
            ax.set_ylabel('Count')
            if kind=='count':
                #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
                #ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
                #ax.set_xlabel('Day'); #ax.set_ylabel(m);
                _h=[]
                for patch in ax.patches:
                    _h.append(patch.get_height())
                 
                _h=np.array(_h)
                _h=_h[~np.isnan(_h)]
                _h_max = np.max(_h)
                for patch in ax.patches:
                    _h= 0 if np.isnan(patch.get_height()) else patch.get_height()
                    patch.set_height(_h)                
                    ax.text(
                            x=patch.get_x() + patch.get_width()/2., 
                            #y=1.04*patch.get_height(), 
                            y=0.02*_h_max+patch.get_height(), 
                            s=fmtx % patch.get_height() if patch.get_height()>0 else None, 
                            #fontsize=16, 
                            color='black', ha='center', 
                            va='bottom', rotation=90, weight='bold',
                            )
            #pl.legend(bbox_to_anchor=(-0.00, 1.2), loc=10, borderaxespad=0., ncol=n_estimators, fontsize=16, ) 
            #pl.legend(bbox_to_anchor=(0.80, 0.8), loc=10, ncol=n_estimators, borderaxespad=0.,)
            #pl.legend()
            
        fn = basename+'300dpi_active_features_sets'+'_'+e+'__'+kind+'.png'
        fn = re.sub('\^','', re.sub('\$','',fn))
        fn = re.sub('\(','', re.sub('\)','',fn))
        fn = re.sub(' ','_', re.sub('\/','',fn))
        fn = re.sub('-','_', re.sub('\/','',fn)).lower()
        #print(fn)
        pl.savefig(plotspath+fn,  bbox_inches='tight', dpi=300)
                
        pl.show()

#%%
for (p,d,e,o), df in C.groupby(['Phase','Dataset','Estimator','Output']):
 if p!= 'TRAIN':  
  #if e!= ref_estimator:
      if '-FS' in e:
        print (p+'\t'+d+'\t\t'+e+'\t'+str(len(df)))
        #aux={}
        #epar = pd.DataFrame(list(df['Parameters']))
        for kind in ['bar',]:
            for m in metrics:
                kwargs={'edgecolor':"k", 'capsize':0.05, 'alpha':0.95, 'ci':'sd', 'errwidth':1, 'dodge':True, 'aspect':3.0618, 'legend':None, } if kind=='bar' else {'notch':0, 'ci':'sd','aspect':1.0618,}
        #        sns.catplot(x='Dataset', y='R$^2$', data=C, hue='Phase', kind='bar', col='Estimator')
        #        g=sns.catplot(x='Dataset', y=m, col='Estimator', data=C, 
        #                       kind=kind, sharey=False, hue='Phase', 
        #                       **kwargs,);
        #        g=sns.catplot(col='Dataset', y=m, hue='Estimator', data=C, 
        #                       kind=kind, sharey=False, x='Phase', 
        #                       **kwargs,);
                if kind=='bar':
                    g=sns.catplot(x='Active Variables', y=m, hue='Estimator', row='Phase', data=df, 
                               #order=ds, hue_order=hs, 
                               kind=kind, sharey=False,  
                               #col_wrap=2, palette=palette_color_1,
                               **kwargs,)                    
                else:
                    pass
                
                #g.despine(left=True)
                fmtx='%2.2f'
                for ax in g.axes.ravel():
                    ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=90)
                    if kind=='bar':
                        ax.set_ylim([0, 1.21*ax.get_ylim()[1]])
                        #ax.set_xlabel('Day'); #ax.set_ylabel(m);
                        _h=[]
                        for p in ax.patches:
                            _h.append(p.get_height())
                         
                        _h=np.array(_h)
                        _h=_h[~np.isnan(_h)]
                        _h_max = np.max(_h)
                        for p in ax.patches:
                            _h= 0 if np.isnan(p.get_height()) else p.get_height()
                            p.set_height(_h)                
                            ax.text(
                                    x=p.get_x() + p.get_width()/4., 
                                    #y=1.04*p.get_height(), 
                                    y=0.02*_h_max+p.get_height(), 
                                    s=fmtx % p.get_height() if p.get_height()>0 else None, 
                                    #fontsize=16, 
                                    color='black', ha='center', 
                                    va='bottom', rotation=90, weight='bold',
                                    )
                    #pl.legend(bbox_to_anchor=(-0.00, 1.2), loc=10, borderaxespad=0., ncol=n_estimators, fontsize=16, ) 
                    pl.legend(bbox_to_anchor=(0.80, 1.0), loc=10, ncol=n_estimators, borderaxespad=0.,)
                    
                fn = basename+'300dpi_active_features'+'_metric_'+m.lower()+'_'+kind+'.png'
                fn = re.sub('\^','', re.sub('\$','',fn))
                fn = re.sub('\(','', re.sub('\)','',fn))
                fn = re.sub(' ','_', re.sub('\/','',fn))
                fn = re.sub('-','_', re.sub('\/','',fn)).lower()
                #print(fn)
                pl.savefig(plotspath+fn,  bbox_inches='tight', dpi=300)
                        
                pl.show()

print('terminou com sucesso')
#%%
