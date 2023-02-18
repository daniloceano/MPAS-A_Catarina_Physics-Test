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
import seaborn as sns
import skill_metrics as sm
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import itertools

era_exp_name = 'ERA5-ERA5'

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
generation_terms = ['RGz', 'RGe', 'Gz', 'Ge']


palette = ['#3B95BF','#87BF4B','#BFAB37','k','#BF3D3B','#C2847A']
c = ['#3B95BF','#87BF4B','#BFAB37','#BF3D3B']

results_ERA = pd.read_csv('./LEC_Results_48h/ERA5-ERA5/ERA5-ERA5.csv')


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
                      titleRMS = 'off', widthRMS = 2.0,tickRMS = tickRMS,
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
      
def export_stats_by_term(stats):
    # Create data frames for each variable, contaning the metrics 
    # for all experiments as rows
    dict_stats_term = {}
    for term in terms:
        term_dfs = []
        for exp, df in stats.items():
            term_dfs.append(df.loc[term])
        df_term = pd.concat(term_dfs, axis=1)
        df_term.columns = stats.keys()
        df_term = df_term.T
        if '∂' in term:
            term = term.split('/')[0]
        df_term.to_csv('stats/terms/stats_Lorenz_'+term+'.csv')    
        dict_stats_term[term] = df_term
    return dict_stats_term


def terms_label(term):
    if term in energy_terms:
        label = 'energy'
    elif term in conversion_terms:
        label = 'conversion'
    elif term in boundary_terms :
        label = 'boundary'
    elif term in residuals_terms:
        label = 'residual'
    elif term in generation_terms:
        label = 'generation'
    elif (term in budget_diff_terms) or ('∂' in term):
        label = 'budget'
    else: 
        pass
    return label

def bar_plot_terms_metrics(stats):

    experiments = [i for i in exps if i != 'ERA5-ERA5']
    
    x = np.arange(0,len(experiments)*4,4)  # the label locations
    width = 0.5  # the width of the bars
    
    dict_stats_term = export_stats_by_term(stats)
    df_stats = stats[experiments[0]]
    
    for metric in list(df_stats.keys()):
        for term in terms:
                
            if '∂' in term:
                term = term.split('/')[0]
            data = dict_stats_term[term][metric].drop('ERA5-ERA5')
            data_norm =(data-data.min())/(data.max()-data.min())
            
            for d, n in zip([data,data_norm],['','_norm']):
                
                plt.close('all')
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111)
                multiplier = 0
            
                offset = width * multiplier
                ax.bar(x+offset,d, width, label=term, color=c[multiplier])
                multiplier += 1
                
                ax.set_ylabel(metric)
                ax.set_xticks(x, experiments, rotation=30, ha='right')
                ax.legend(loc='upper left')
                ax.grid(color='grey',linewidth=0.5, alpha=0.4, linestyle='dashed')
                fname = 'Figures_48h/Lorenz/'+term+'_'+metric+n+'.png'
                fig.savefig(fname, dpi=500)
                print(fname,'saved')
                
def test_for_rmse_r(stats,r_val,rmse_val):
            
    dict_stats_term = export_stats_by_term(stats)
    bad_schemes = {}
    
    for term in terms:
            
        if '∂' in term:
            term = term.split('/')[0]
        r = dict_stats_term[term]['r'].drop('ERA5-ERA5')
        r_norm =  (r-r.min())/(r.max()-r.min())
        rmse = dict_stats_term[term]['rmse'].drop('ERA5-ERA5')
        rmse_norm = (rmse-rmse.min())/(rmse.max()-rmse.min())
        
        df = pd.DataFrame(r_norm,columns=['r_norm'])
        df['rmse_norm'] = rmse_norm
        reproved = (df[(df.r_norm <= r_val) | (df.rmse_norm >= rmse_val)]).index.to_list()
        bad_schemes[term] = reproved
        
        # df_test = df.where(df.r >= 0.7).where(df.rmse_norm <= 0.2)
    return bad_schemes
    
def get_list_of_reproveds(bad_schemes):
    
    bad_schemes = pd.DataFrame(
        dict([ (k,pd.Series(v)) for k,v in bad_schemes.items() ]))
    
    reproved_schemes = []
    
    for col in bad_schemes.columns:
        
        p = bad_schemes[col].dropna().to_list()
        for i in p:
            reproved_schemes.append(i)
            
    reproved_schemes = pd.Series(reproved_schemes).unique()
    
    return reproved_schemes


# ----------------------------------
results = glob.glob('LEC_Results_48h/*')
exps = []
for exp in results:
    exps.append(exp.split('/')[1].split('_MPAS_track')[0])

i = 0
stats = {}
for exp in exps:
    print(exp)
    dirname = exp.split('/')[-1]
    outfile = glob.glob(
        './LEC_Results_48h/'+str(dirname)+'*'+'/'+str(dirname)+'*csv')[0]
    df = pd.read_csv(outfile)
    df['Datetime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
    time = df.Datetime    
    
    stats_exp = {}
    for term in terms:
        stats_exp[term] = {}
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
        
        tmp['type'] = terms_label(term)
        
        reference = results_ERA[term]
        predicted = tmp['value']
        
        stats_exp[term]['bias'] = np.mean(reference - predicted)        
        stats_exp[term]['mse'] = mean_squared_error(reference, predicted)
        stats_exp[term]['rmse'] = np.sqrt(stats_exp[term]['mse'])
        stats_exp[term]['mae'] = mean_absolute_error(reference, predicted) 
        stats_exp[term]['mape'] = mean_absolute_percentage_error(reference, predicted)
        stats_exp[term]['r'] = np.corrcoef(reference, predicted)[0,1]

        if i == 0:
            i += 1 
            df_sns = tmp
        else:
            df_sns = pd.concat([df_sns, tmp])
        
    # Create and export DataFrame containing the metrics for each experiment
    df_stats = pd.DataFrame(stats_exp).T
    df_stats.to_csv('stats/experiments/stats_Lorenz_'+exp+'.csv')
    # Create a dict containing df for each experiment, where columns are
    # metrics for each variable (rwos)
    stats[exp] = df_stats    
            
df_sns.index = np.arange(0,len(df_sns))

# ----------------------------------


# Plot timeseries and Taylor Diagrams for each term
for term in terms:
    data = df_sns[df_sns['term'] == term]
    # plot_timeseries(data)
    # plot_taylor(data)
    
# bar_plot_terms_metrics(stats)

# =============================================================================
# CHoosing best options
# =============================================================================
# r_vals, rmse_vals = [], []
# n = []
# for r_val, rmse_val  in itertools.product(np.arange(0.5,1,0.1),
#                                           np.arange(0,0.5,0.1)):
#     bad_schemes = test_for_rmse_r(stats,r_val,rmse_val)
#     reproved_schemes = get_list_of_reproveds(bad_schemes)
#     r_vals.append(r_val)
#     rmse_vals.append(rmse_val)
#     n.append(len(reproved_schemes))
#     print(r_val,rmse_val,len(reproved_schemes))
    
# df_reproved = pd.DataFrame(np.array([r_vals,rmse_vals,n]).T, columns=['r','rmse','n'])
# df_reproved = df_reproved.set_index(['r','rmse'])
# da_reproved = df_reproved.to_xarray()
# pc = plt.pcolormesh(da_reproved.r, da_reproved.rmse,da_reproved.n)
# plt.colorbar(pc)

## Choosen r > 0.8 and rmse < 0.4
bad_schemes = test_for_rmse_r(stats,0.8,0.4)
reproved_schemes = get_list_of_reproveds(bad_schemes)

for exp in exps:
    if exp not in reproved_schemes:
        print(exp)