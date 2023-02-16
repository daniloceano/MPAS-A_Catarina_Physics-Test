#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 13:23:07 2022

@author: danilocoutodsouza
"""

import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cmocean
import glob
import numpy as np
import seaborn as sns
import skill_metrics as sm
from matplotlib import rcParams

era_exp_name = 'ERA5-ERA5'

def plot_timeseries(data):
    plt.close('all')
    sns.set_theme(style="ticks", font_scale=2.2)
    sns.set(rc={'figure.figsize':(10,15)})
    g = sns.relplot(x="date", y="value", kind="line",
                     size="source", size_order=['ERA5','MPAS-A'],
                     style='microp', hue='cumulus', palette=palette,
                     markers=True,data=data, legend='full')
    g.set_xticklabels(rotation=30)
    plt.title(term)
    if term in budget_diff_terms:
        fname = term[:3]
    else:
        fname = term
    plt.savefig('./Figures_48h/timeseries/'+fname+'.png',dpi=300)
    print('Time series created for term: '+fname)
    
def get_stats(data):
    ccoef, crmsd, sdev, exps = [], [], [], []
    experiments = list(data['experiment'].unique())
    data.index = data['date']
    # Resample for 1H for standardization
    reference = data[
        data['experiment'] == era_exp_name].resample('1H').mean()['value']
    # get experiments time range
    min_time = data[data['experiment'] != era_exp_name].index.min()
    max_time = data[data['experiment'] != era_exp_name].index.max()
    # Slice data for using only times present in model data
    reference = reference[(reference.index >= min_time) &
                          (reference.index <= max_time)]
    for experiment in experiments:
        if experiment != era_exp_name:
            predicted = data[
                    data['experiment'] == experiment].resample('1H').mean()['value']
            # Just to make sure
            predicted = predicted[(predicted.index >= min_time) &
                                  (predicted.index <= max_time)]
            
            # Compute and store stats
            stats = sm.taylor_statistics(predicted,reference)
            ccoef.append(stats['ccoef'][1])
            crmsd.append(stats['crmsd'][1])
            sdev.append(stats['sdev'][1])
            exps.append(experiment)
    ccoef, crmsd, sdev = np.array(ccoef),np.array(crmsd),np.array(sdev)
    return sdev,crmsd,ccoef,exps
       
def taylor_diagram(sdevs,crmsds,ccoefs,experiments):
    '''
    Produce the Taylor diagram
    Label the points and change the axis options for SDEV, CRMSD, and CCOEF.
    Increase the upper limit for the SDEV axis and rotate the CRMSD contour 
    labels (counter-clockwise from x-axis). Exchange color and line style
    choices for SDEV, CRMSD, and CCOEFF variables to show effect. Increase
    the line width of all lines.
    For an exhaustive list of options to customize your diagram, 
    please call the function at a Python command line:
    >> taylor_diagram
    '''
    # Set the figure properties (optional)
    rcParams.update({'font.size': 14}) # font size of axes text
    STDmax = round(np.amax(sdevs))
    RMSmax = round(np.amax(crmsds))
    tickRMS = np.linspace(0,round(RMSmax*1.2,1),6)
    axismax = round(STDmax*1.2,1)
    sm.taylor_diagram(sdevs,crmsds,ccoefs,
                      markerLabelColor = 'b', 
                      markerLabel = experiments,
                      markerColor = 'r', markerLegend = 'on', markerSize = 15, 
                      titleRMS = 'off', widthRMS = 2.0,
                      colRMS = '#728B92', styleRMS = '--',  
                      widthSTD = 2, styleSTD = '--', colSTD = '#8A8A8A',
                      colCOR = 'k', styleCOR = '-',
                      widthCOR = 1.0, titleCOR = 'on',
                      colObs = 'k', markerObs = '^',
                      titleOBS = 'Obs.', styleObs =':',
                      axismax = axismax, alpha = 1)
    
def plot_taylor(data):
    plt.close('all')
    plt.figure(figsize=(12,10))
    stats = get_stats(data)
    sdev,crmsd,ccoef,expnames = stats[0],stats[1],stats[2],stats[3]
    taylor_diagram(sdev,crmsd,ccoef,expnames)
    if term in budget_diff_terms:
        fname = term[:3]
    else:
        fname = term
    plt.savefig('./Figures_48h/taylor/'+fname+'.png', dpi=300)
    print('Taylor diagram created for term: '+term)
    
    
# ----------------------------------
results = glob.glob('LEC_results_48h/*')
exps = []
for exp in results:
    exps.append(exp.split('/')[1].split('_MPAS_track')[0])
    
terms = ['Az', 'Ae', 'Kz', 'Ke',
         'Cz', 'Ca', 'Ck', 'Ce',
         'BAz', 'BAe', 'BKz', 'BKe',
         'Gz', 'Ge',
         '∂Az/∂t (finite diff.)', '∂Ae/∂t (finite diff.)',
         '∂Kz/∂t (finite diff.)','∂Ke/∂t (finite diff.)',
         'RGz', 'RGe', 'RKz', 'RKe']
energy_terms = ['Az','Ae','Kz','Ke']
conversion_terms = ['Cz','Ca','Ck','Ce']
boundary_terms = ['BAz', 'BAe', 'BKz', 'BKe']
budget_diff_terms = ['∂Az/∂t (finite diff.)', '∂Ae/∂t (finite diff.)',
                 '∂Kz/∂t (finite diff.)', '∂Ke/∂t (finite diff.)']
residuals_terms = ['RGz', 'RKz', 'RGe', 'RKe']

palette = ['#3B95BF','#87BF4B','#BFAB37','k','#BF3D3B','#C2847A']

i = 0
for exp in exps:
    print(exp)
    dirname = exp.split('/')[-1]
    outfile = glob.glob(
        './LEC_results_48h/'+str(dirname)+'*'+'/'+str(dirname)+'*csv')[0]
    df = pd.read_csv(outfile)
    df['Datetime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
    time = df.Datetime    
    
    for term in terms:
        tmp = pd.DataFrame(df[term]).rename(columns={term:'value'})
        tmp['date'] = time
        tmp['term'] = term
        tmp['microp'] = exp.split('-')[0]
        tmp['cumulus'] = exp.split('-')[-1]
        tmp['experiment'] = exp
        
        if 'ERA5'in exp:
            tmp['source'] = 'ERA5'
        else:
            tmp['source'] = 'MPAS-A'
        
        if term in energy_terms:
            tmp['type'] = 'energy'
        elif term in conversion_terms:
            tmp['type'] = 'conversion'
        elif term in boundary_terms :
            tmp['type'] = 'boundary'
        elif term in residuals_terms:
            tmp['type'] = 'residual'
        elif term in budget_diff_terms:
            tmp['type'] = 'budget'
        else: 
            pass
            
        if i == 0:
            i += 1 
            df_sns = tmp
        else:
            df_sns = pd.concat([df_sns, tmp])
            
df_sns.index = np.arange(0,len(df_sns))



for term in terms:
    data = df_sns[df_sns['term'] == term]
    
    plot_timeseries(data)
    plot_taylor(data)