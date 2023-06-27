#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:14:06 2023

@author: daniloceano
"""

from glob import glob
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt

DiscreteColors = ['#58A8D6', '#74D669', '#D6BF5A', '#D6713A', '#D63631']

def sns_heatmap(data,title):
     
    cmap = LinearSegmentedColormap.from_list('Custom',
                                    DiscreteColors, len(DiscreteColors))
    plt.close('all')
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(data, annot=False, linewidths=.5, ax=ax, cmap=cmap)


    plt.title(title)
    plt.tight_layout()
    f.savefig('Figures_'+benchmarks+'/'+title+'.png')

benchmarks = input("prompt experiments (24h, 48h, 48h_sst): ")

stat_files = glob('./stats-'+benchmarks+'/*normalised.csv')

stats = {}
for file in stat_files:
    variable = file.split('./stats-'+benchmarks+'/')[-1].split('_')[0]
    tmp = pd.read_csv(file, index_col=[0]).sort_index(ascending=True)
    indexes = tmp.index
    tmp.index = range(len(tmp))
    
    if 'rmse' in tmp.columns:
        stats[variable] = tmp['rmse']
    else:
        for col in tmp.columns:
            stats[col] = tmp[col]
    
            
df = pd.DataFrame.from_dict(stats)
df.index = indexes
df['mean'] = df.mean(axis=1)
# 
sns_heatmap(df,'scores')


df.to_csv('./stats-'+benchmarks+'/score-table.csv')