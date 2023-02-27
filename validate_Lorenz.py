#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 13:23:07 2022

@author: danilocoutodsouza
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from sklearn.metrics import mean_squared_error
import itertools
import seaborn as sns
import cmocean.cm as cmo
from matplotlib.colors import LinearSegmentedColormap

era_exp_name = 'ERA5-ERA5'

terms = ['Az', 'Ae', 'Kz', 'Ke',
         'Cz', 'Ca', 'Ck', 'Ce',
         'BAz', 'BAe', 'BKz', 'BKe',
         'Gz', 'Ge',
         '∂Az/∂t (finite diff.)', '∂Ae/∂t (finite diff.)',
         '∂Kz/∂t (finite diff.)','∂Ke/∂t (finite diff.)',
         'RGz', 'RGe', 'RKz', 'RKe']
energy_terms = ['Az','Ae','Kz','Ke']
budget_diff_terms = ['∂Az/∂t (finite diff.)', '∂Ae/∂t (finite diff.)',
                 '∂Kz/∂t (finite diff.)', '∂Ke/∂t (finite diff.)']
budget_diff_renamed = ['∂Az/∂t', '∂Ae/∂t',
                 '∂Kz/∂t', '∂Ke/∂t']

main_terms = ['Az', 'Ae', 'Kz', 'Ke', 'Ca', 'Ce', 'Ck', 'Ge']


results_ERA = pd.read_csv('./LEC_Results_48h/ERA5-ERA5/ERA5-ERA5.csv')

DiscreteColors = ['#58A8D6', '#74D669', '#D6BF5A', '#D6713A', '#D63631']
    
def get_rsme_r(results, results_ERA):

    df_exp_dict = {}
    rmse_dict, corrcoef_dict = {}, {}
    for exp in exps:
        dirname = exp.split('/')[-1]
        outfile = glob.glob(
            './LEC_Results_48h/'+str(dirname)+'*'+'/'+str(dirname)+'*csv')[0]
        df_exp = pd.read_csv(outfile, index_col=[0])
        df_exp['Datetime'] = pd.to_datetime(df_exp.Date) + pd.to_timedelta(
            df_exp.Hour, unit='h')
        time = df_exp.Datetime  
        df_exp.index = time
        df_exp = df_exp.drop(columns=['Datetime'])
        df_exp_dict[exp] = df_exp
        
        rmse = {}
        corrcoef = {}
        for term in terms:
            reference = results_ERA[term].values.ravel()
            predicted = df_exp[term].values.ravel()
            rmse[term] =  np.sqrt(mean_squared_error(reference, predicted))
            corrcoef[term] = np.corrcoef(reference, predicted)[0,1]
        rmse_dict[exp] = pd.DataFrame.from_dict([rmse])
        corrcoef_dict[exp] = pd.DataFrame.from_dict([corrcoef])
        
    df_rmse = pd.concat(rmse_dict)
    df_rmse.index = list(rmse_dict.keys())
        
    df_corrcoef =  pd.concat(corrcoef_dict)
    df_corrcoef.index = list(corrcoef_dict.keys())

    for term1, term2 in zip(budget_diff_terms,budget_diff_renamed):
        df_rmse = df_rmse.rename(columns={term1:term2})
        df_corrcoef = df_corrcoef.rename(columns={term1:term2})
        
    return df_rmse, df_corrcoef

def sns_heatmap(data,title):
     
    if 'RMSE_Normalised' in title:
        cmap = LinearSegmentedColormap.from_list('Custom',
                                    DiscreteColors, len(DiscreteColors))
    elif 'R_Normalised' in title:
        cmap = LinearSegmentedColormap.from_list('Custom',
                                    DiscreteColors[::-1], len(DiscreteColors))
    else:
        if 'RMSE' in title:
            cmap = cmo.matter
        else:
            cmap = cmo.matter_r
    
    plt.close('all')
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(data, annot=False, linewidths=.5, ax=ax, cmap=cmap)


    plt.title(title)
    plt.tight_layout()
    f.savefig('Figures_48h/stats_Lorenz/'+title)
   

results = glob.glob('LEC_Results_48h/*')
results_ERA = pd.read_csv('./LEC_Results_48h/ERA5-ERA5/ERA5-ERA5.csv')
exps = []
for exp in results:
    if 'ERA5-ERA5' not in exp:
        exps.append(exp.split('/')[1].split('_MPAS_track')[0])

df_rmse_all, df_corrcoef_all = get_rsme_r(results, results_ERA)

df_rmse_main = df_rmse_all[main_terms]
df_corrcoef_main = df_corrcoef_all[main_terms]


for df_rmse, df_corrcoef in zip([df_rmse_all, df_rmse_main],
                                [df_corrcoef_all, df_corrcoef_main]):
    
    print('-------------------------\n')
    print(df_rmse.columns.to_list())
    
    if len(df_rmse.columns) == 22:
        fname = 'all'
    else:
        fname = 'main'

    # Print which terms had the worse general performance, for each metric
    df_rmse_energy = df_rmse[energy_terms]
    rmse_energy_thres = round(df_rmse_energy.max().max()/4,-4)
    print('Greater RMSE ( >= '+str(rmse_energy_thres)+'):',
          df_rmse_energy.mean()[df_rmse_energy.mean() >=
                                       rmse_energy_thres].index.tolist())
    
    df_rmse_not_energy = df_rmse.drop(columns=energy_terms)
    rmse_thres = round(df_rmse_not_energy.max().max()/2)
    print('Greater RMSE ( >= '+str(rmse_thres)+'):',
         df_rmse_not_energy.mean()[df_rmse_not_energy.mean() >=
                                       rmse_thres].index.tolist())
    
    print('Smaller correlations ( <= 0.6):',
          df_corrcoef.mean()[df_corrcoef.mean() <= 0.6].index.tolist())
    
    # Normalize values for comparison
    rmse_norm = (df_rmse-df_rmse.min()
                      )/(df_rmse.max()-df_rmse.min())   
    corrcoef_norm =  (df_corrcoef-df_corrcoef.min()
                      )/(df_corrcoef.max()-df_corrcoef.min())   
    
    for data, title in zip([df_rmse_energy, df_rmse_not_energy,
                            rmse_norm, df_corrcoef, corrcoef_norm],
                           ['RMSE_energy', 'RMSE', 'RMSE_Normalised',
                            'R', 'R_Normalised']):
        sns_heatmap(data,title+'_'+fname)
    
    
    rmse_vals = np.arange(0,1,0.05)
    r_vals = np.arange(0,1,0.05)
    
    for rmse_val, r_val in itertools.product(rmse_vals, r_vals):
        
        rmse_val, r_val = round(rmse_val,2), round(r_val,2)
        
        approved_rmse = rmse_norm[rmse_norm <= rmse_val].dropna().index.to_list()
        approved_r = corrcoef_norm[corrcoef_norm >= r_val].dropna().index.to_list()
    
        if len(approved_rmse) > 0 and len(approved_r) > 0:
            approved = list(approved_rmse)
            approved.extend(x for x in approved_r if x not in approved)
        else:
            approved = []
        
        if len(approved) > 0 and len(approved) <=4:
            print('\nrmse:', rmse_val, 'r:', r_val)
            [print(i) for i in approved]