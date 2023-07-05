# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    score-table.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/03/14 18:14:06 by Danilo            #+#    #+#              #
#    Updated: 2023/07/05 19:40:06 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
from glob import glob
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt

DiscreteColors = ['#58A8D6', '#74D669', '#D6BF5A', '#D6713A', '#D63631']

def sns_heatmap(data):
     
    cmap = LinearSegmentedColormap.from_list('Custom',
                                    DiscreteColors, len(DiscreteColors))
    plt.close('all')
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(data, annot=False, linewidths=.5, ax=ax, cmap=cmap)


    plt.title('scores')
    plt.tight_layout()

    fname = f'{figures_directory}/scores.png'
    f.savefig(fname, dpi=500)
    print(f'Saved {fname}')

experiment_directory = '../experiments_48h'
benchmarks_name = '48h_pbl'

benchmarks_path = '/p1-nemo/danilocs/mpas/MPAS-BR/benchmarks/Catarina_physics-test/'
benchmarks_directory = f'{benchmarks_path}/Catarina_250-8km.physics-pbl_sst/'

stats_directory = os.path.join(experiment_directory, f'stats_{benchmarks_name}')
figures_directory = os.path.join(experiment_directory, f'Figures_{benchmarks_name}')

stats_files = glob(f'{stats_directory}/*csv')

stats = {}
for file in stats_files:
    variable = os.path.basename(file).split('_')[0].split('.csv')[0]
    print(variable)
    
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
sns_heatmap(df)
csv_filename = f'{stats_directory}/score-table.csv'
df.to_csv(csv_filename)
print('Saved',csv_filename)