#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:10:49 2023

@author: danilocoutodsouza
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = {'kessler':'whitesmoke','thompson':'gainsboro', 'wsm6':'darkgray',
          'off':'dimgray'}

edgecolors = {'fritsch':'tab:orange','tiedtke':'tab:red',
          'ntiedtke':'tab:purple', 'freitas':'tab:brown','off':'tab:green'}

posx = {'kessler':0,'thompson':2, 'wsm6':4, 'off':6}


def get_exp_name(bench):
    expname = bench.split('/')[-2].split('run.')[1]
    microp = expname.split('.')[0].split('_')[-1]
    cumulus = expname.split('.')[-1].split('_')[-1] 
    return microp+'_'+cumulus

run_duration = pd.read_csv('run_duration', delim_whitespace=True,
                           header=None, names=list(range(9)),
                           parse_dates=[[5,6,7]], infer_datetime_format=True)

starts = pd.DataFrame(run_duration.iloc[::2])
starts.index = range(len(starts))
ends = pd.DataFrame(run_duration.iloc[1::2])
ends.index = range(len(ends))

exps = []
for bench in starts[8]:
    exp = get_exp_name(bench)
    exps.append(exp)
    
exps = pd.DataFrame(exps)
exps['start'] = pd.to_datetime('2023 '+starts['5_6_7'])
exps['end'] = pd.to_datetime('2023 '+ends['5_6_7'])
exps['duration'] = (exps['end']-exps['start']).dt.total_seconds()

color, edgecolor = [], []
pos = []
i = 0
for exp in exps[0]:
    microp = exp.split('_')[0]
    cumulus = exp.split('_')[1]
    color.append(colors[microp])
    edgecolor.append(edgecolors[cumulus])
    pos.append(i+posx[microp])
    i += 2
    
exps['color'] = color
exps['edgecolor'] = edgecolor

plt.close('all')
plt.figure(figsize=(12,10))
plt.bar(pos,exps['duration'], color=exps['color'], edgecolor=exps['edgecolor'],
        linewidth=2)
plt.xticks(pos,exps[0],rotation=30, ha='right')
plt.yscale('log')
for l in exps['duration'][:8]:
    plt.axhline(l, alpha=0.2, linestyle='dashed',c='k',linewidth=1)
    
plt.savefig('run_duration.png',dpi=500)