#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 13:23:07 2022

@author: danilocoutodsouza
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from sklearn.metrics import mean_squared_error
import itertools
import seaborn as sns
import cmocean.cm as cmo
from matplotlib.colors import LinearSegmentedColormap


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

DiscreteColors = ['#58A8D6', '#74D669', '#D6BF5A', '#D6713A', '#D63631']
    
def get_rsme_r(results_directory, results_ERA):

    df_exp_dict = {}
    rmse_dict, corrcoef_dict = {}, {}
    for exp in exps:
        dirname = exp.split('/')[-1]
        outfile = glob.glob(f'{results_directory}/{dirname}*/{dirname}*csv')[0]
        df_exp = pd.read_csv(outfile, index_col=[0])
        df_exp['Datetime'] = pd.to_datetime(df_exp.Date) + pd.to_timedelta(
            df_exp.Hour, unit='h')
        time = df_exp.Datetime  
        df_exp.index = time
        df_exp = df_exp.drop(columns=['Datetime'])
        df_exp_dict[exp] = df_exp
        
        results_ERA['Datetime'] = pd.to_datetime(results_ERA['Date']
                        ) + pd.to_timedelta(results_ERA['Hour'], unit='H')
        
        mask = (results_ERA['Datetime'] >= df_exp.index[0]) & (
            results_ERA['Datetime'] <= df_exp.index[-1])
        
        results_ERA = results_ERA.loc[mask]
        
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

def sns_heatmap(data,title, figures_directory):
     
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
    if '48h' in benchmarks:
        y = 6
    elif benchmarks in ['72h_sst', '2403-2903']:
        y = 2
    else:
        y = 10
    f, ax = plt.subplots(figsize=(10, y))
    sns.heatmap(data, annot=False, linewidths=.5, ax=ax, cmap=cmap)


    plt.title(title)
    plt.tight_layout()
    stats_figure_directory = f'{figures_directory}/stats_Lorenz/'
    if not os.path.exists(stats_figure_directory):
            os.makedirs(stats_figure_directory)
    f.savefig(f'{stats_figure_directory}/{title}')
    
def normalize_df(df):
    return (df-df.min())/(df.max()-df.min())
   
# results_directory = input("prompt path to tracks: ")
results_directory = '../experiments_48h/LEC_Results_48h_pbl/'
experiment_directory = ('/').join(os.path.dirname(results_directory).split('/')[:-1])
benchmarks = os.path.basename(os.path.dirname(results_directory)).split('LEC_Results_')[1]

stats_directory = os.path.join(experiment_directory, f'stats_{benchmarks}')
figures_directory = os.path.join(experiment_directory, f'Figures_{benchmarks}')
lec_directory = os.path.join(experiment_directory, f'LEC_Results_{benchmarks}')

for directory in [figures_directory, stats_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

results = glob.glob(f'{lec_directory}/*')
        
results_ERA = pd.read_csv(glob.glob(f'{results_directory}/*ERA5*/*ERA5*.csv')[0])

exps = []
for exp in results:
    if 'ERA5' not in exp:
        exps.append(os.path.basename(exp).split('_')[0])

df_rmse_all, df_corrcoef_all = get_rsme_r(results_directory, results_ERA)

df_rmse_main = df_rmse_all[main_terms]
df_corrcoef_main = df_corrcoef_all[main_terms]

# make a csv with normalised values for comparing with other terms
df_rmse_main_norm = normalize_df(df_rmse_main).sort_index(ascending=True)
df_rmse_main_norm.to_csv(f'{stats_directory}/Lorenz-main_RMSE_normalised.csv')

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
        sns_heatmap(data, f'{title}_{fname}', figures_directory)
    
    
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