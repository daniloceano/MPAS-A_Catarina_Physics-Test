#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 20:04:44 2023

@author: daniloceano
"""

import glob
import argparse
import f90nml
import datetime
import itertools
import imageio

import numpy as np
import pandas as pd
import xarray as xr
import cmocean.cm as cmo
import cartopy.crs as ccrs
import seaborn as sns

import scipy.stats as st
import skill_metrics as sm

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rcParams
from metpy.calc import wind_speed

class BenchData:
    
    def __init__(self, path, times):
        self.bench = path
        self.data = xr.open_dataset(path+'/latlon.nc').chunk({"Time": -1})
        self.data = self.data.assign_coords({"Time":times})
        

    def get_exp_name(self):
        expname = self.bench.split('/')[-1].split('run.')[-1]
        microp = expname.split('.')[0].split('_')[-1]
        cumulus = expname.split('.')[-1].split('_')[-1] 
        return microp+'_'+cumulus
        
def interpolate_mpas_data(variable_data, target, reference):
    if reference == 'quickscat':
        return variable_data.rename({'latitude':'lat','longitude':'lon'}
            ).interp(lat=target.lat,lon=target.lon,method='cubic',
                     assume_sorted=False)
    elif reference == 'ERA5':
        return variable_data.interp(latitude=target.latitude,
                     longitude=target.longitude,method='cubic',
                     assume_sorted=False)

def convert_lon(df,LonIndexer):
    df.coords[LonIndexer] = (df.coords[LonIndexer] + 180) % 360 - 180
    df = df.sortby(df[LonIndexer])
    return df

def anim(path,pattern):
    filenames = glob.glob(path+pattern+'*')
    ff=[]
    for i in range(len(filenames)):
    	ff.append(filenames[i].split('/')[-1].split('.')[0])
    
    with imageio.get_writer(path+'/wind.mp4',fps=5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

def get_times_nml(namelist,model_data):
    ## Identify time range of simulation using namelist ##
    # Get simulation start and end dates as strings
    start_date_str = namelist['nhyd_model']['config_start_time']
    run_duration_str = namelist['nhyd_model']['config_run_duration']
    # Convert strings to datetime object
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d_%H:%M:%S')
    
    run_duration = datetime.datetime.strptime(run_duration_str,'%d_%H:%M:%S')
    # Get simulation finish date as object and string
    finish_date  = start_date + datetime.timedelta(days=run_duration.day,
                                                   hours=run_duration.hour)
    ## Create a range of dates ##
    times = pd.date_range(start_date,finish_date,periods=len(model_data.Time)+1)[0:-1]
    return times

def plot_taylor(sdevs,crmsds,ccoefs,experiments):
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
                      tickRMS = tickRMS, titleRMS = 'off', widthRMS = 2.0,
                      colRMS = '#728B92', styleRMS = '--',  
                      widthSTD = 2, styleSTD = '--', colSTD = '#8A8A8A',
                      titleSTD = 'on',
                      colCOR = 'k', styleCOR = '-',
                      widthCOR = 1.0, titleCOR = 'off',
                      colObs = 'k', markerObs = '^',
                      titleOBS = 'IMERG', styleObs =':',
                      axismax = axismax, alpha = 1)

## Parser options ##
parser = argparse.ArgumentParser()

parser.add_argument('-bdir','--bench_directory', type=str, required=True,
                        help='''path to benchmark directory''')
parser.add_argument('-o','--output', type=str, default=None,
                        help='''output name to append file''')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-q','--quickscat', type=str, default=None,
                        help='''path to QUICKSCAT data''')
group.add_argument('-e','--ERA5', type=str, default=None,
                        help='''path to ERA5 data''')

args = parser.parse_args()

benchmarks_experiment = input("prompt experiments (24h, 48h, 48h_sst): ")

## Start the code ##
# benchs = glob.glob(args.bench_directory+'/run*')
benchs = glob.glob(args.bench_directory+'/run*')
# Dummy for getting model times
model_output = benchs[0]+'/latlon.nc'
namelist_path = benchs[0]+"/namelist.atmosphere"
# open data and namelist
model_data = xr.open_dataset(model_output)
namelist = f90nml.read(glob.glob(namelist_path)[0])
times = get_times_nml(namelist,model_data)

first_day = datetime.datetime.strftime(times[0], '%Y-%m-%d')
last_day = datetime.datetime.strftime(times[-2], '%Y-%m-%d')
if args.quickscat:
    da_reference = convert_lon(xr.open_dataset(args.quickscat),'lon').sel(
                        lat=slice(model_data.latitude[-1],model_data.latitude[0]),
                        lon=slice(model_data.longitude[0],model_data.longitude[-1])
                        ).sel(time=slice(first_day,last_day))
    u = da_reference.uwnd
    v = da_reference.vwnd
    reference = 'quickscat'
    
if args.ERA5:
    da_reference = convert_lon(xr.open_dataset(args.ERA5),'longitude').sel(
                        latitude=slice(
                            model_data.latitude[0],model_data.latitude[-1]),
                        longitude=slice(
                            model_data.longitude[0],model_data.longitude[-1])
                        ).sel(time=slice(first_day,last_day))
    u = da_reference.u10
    v = da_reference.v10
    reference = 'ERA5'                          

print('\nOpening all data and putting it into a dictionary...')
benchmarks = [BenchData(bench, times) for bench in benchs]
data = {}

for benchmark in benchmarks:
    exp_name = benchmark.get_exp_name()
    print('\n',exp_name)
    data[exp_name] = {}
    
    for model_var, reference_var in zip(['u10','v10','windspeed'],
                                        [u,v,'windspeed']):
        
        data[exp_name][model_var] = {}
        
        if model_var == 'windspeed':
            variable_experiment_data = wind_speed(benchmark.data['u10'],
                                               benchmark.data['v10'])
            reference_data = wind_speed(u,v)  
            
            
        else:
            variable_experiment_data = benchmark.data[model_var]
            reference_data = reference_var
            
        interpoled_data = interpolate_mpas_data(
            variable_experiment_data.compute(),reference_data, reference)
        
        data[exp_name][model_var]['data'] = variable_experiment_data
        data[exp_name][model_var]['interpoled_data'] = interpoled_data

        resampled_data = interpoled_data.sel(Time=reference_data.time)

        statistical_metrics = sm.taylor_statistics(
            resampled_data.values.ravel(), reference_data.values.ravel())
    
        data[exp_name][model_var].update(
            {metric: statistical_metrics[metric][1] 
             for metric in statistical_metrics.keys()})
    

# # =============================================================================
# # Make gif
# # =============================================================================
# print('\nPlotting maps...')
# datacrs = ccrs.PlateCarree()

# levels = np.arange(0,30,2)
# skip = (slice(None, None, 2), slice(None, None, 2))

# for t in times:
#     plt.close('all')
#     fig = plt.figure(figsize=(10, 12))
#     gs = gridspec.GridSpec(6, 3)
#     print('\n',t)
#     i = 0
#     for col in range(3):
#         for row in range(6):
#             bench = benchs[i]
#             benchmark_data = BenchData(bench)
#             experiment = benchmark_data.get_exp_name()
            
#             u = data[experiment]['u10']['data'].sel(Time=t)
#             v = data[experiment]['v10']['data'].sel(Time=t)
#             windspeed = data[experiment]['windspeed']['data'].sel(Time=t)
            
                
#             ax = fig.add_subplot(gs[row, col], projection=datacrs,frameon=True)
            
#             ax.set_extent([-55, -30, -20, -35], crs=datacrs) 
#             gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',
#                               alpha=0.8, color='#383838')
#             gl.xlabel_style = {'size': 12, 'color': '#383838'}
#             gl.ylabel_style = {'size': 12, 'color': '#383838'}
#             gl.right_labels = None
#             gl.top_labels = None
#             if row != 5:
#                 gl.bottom_labels = None
#             if col != 0:
#                 gl.left_labels = None
        
#             ax.text(-50,-19,experiment)
            
#             cf = ax.contourf(windspeed.longitude, windspeed.latitude, windspeed,
#                               cmap='rainbow', levels=levels)
        
#             # ax.quiver(u.longitude[::20], u.latitude[::20],
#             #           u[::20,::20], v[::20,::20], width=0.025,headaxislength=3.5)
#             ax.streamplot(u.longitude[::20].values, u.latitude[::20].values,
#                           u[::20,::20].values, v[::20,::20].values, color='k')
            
#             ax.coastlines(zorder = 1)
#         i += 1
    
#     cb_axes = fig.add_axes([0.85, 0.18, 0.04, 0.6])
#     fig.colorbar(cf, cax=cb_axes, orientation="vertical") 
#     fig.subplots_adjust(wspace=0.1,hspace=0, right=0.8)
    
#     if args.output is not None:
#         fname = args.output
#     else:
#         fname = (args.bench_directory).split('/')[-2].split('.nc')[0]
#     fname1 = fname+'_wind-maps'
    
#     time_string = t.strftime("%d-%HZ")
#     fname = './Figures_48h/wind/wind-test'+'_'+time_string+'.png'
    
#     fig.savefig(fname, dpi=500)
#     print(fname,'saved')
    
#     anim('./Figures_48h/wind/','wind-test')
    
# =============================================================================
# Plot Taylor Diagrams and do Statistics ##
# =============================================================================

for var in ['u10','v10','windspeed']:
    
    print('\n-------------------------------')
    print(var)
    
    if args.output is not None:
        fname = args.output
    else:
        fname = (args.bench_directory).split('/')[-2].split('.nc')[0]
    fname += '_'+var
    
    ccoef = [data[exp][var]['ccoef'] for exp in data.keys() if exp != 'IMERG']
    crmsd  = [data[exp][var]['crmsd'] for exp in data.keys() if exp != 'IMERG']
    sdev = [data[exp][var]['sdev'] for exp in data.keys() if exp != 'IMERG']
    ccoef, crmsd, sdev = np.array(ccoef),np.array(crmsd),np.array(sdev)
    print('plotting taylor diagrams..')
    fig = plt.figure(figsize=(10,10))
    plot_taylor(sdev,crmsd,ccoef,list(data.keys()))
    plt.tight_layout(w_pad=0.1)
    fig.savefig('Figures_'+benchmarks_experiment+'/stats_wind/'+fname+'-taylor.png', dpi=500)    
    print('stats_wind/'+fname+'-taylor created!')
    
    
    df_stats = pd.DataFrame(crmsd,
                           index=[exp for exp in data.keys() if exp != 'IMERG'],
                           columns=['rmse'])
    df_stats['ccoef'] = ccoef
    
    
    
    # Normalize values for comparison
    df_stats_norm = (df_stats-df_stats.min()
                      )/(df_stats.max()-df_stats.min()) 
    df_stats_norm.sort_index(ascending=True).to_csv(
        './stats-'+benchmarks_experiment+'/'+var+'_RMSE_normalised.csv')
    
    
    for df, title in zip([df_stats, df_stats_norm],
                           ['stats', 'stats normalised']):
        for col in df.columns:
            plt.close('all')
            f, ax = plt.subplots(figsize=(10, 10))
            ax.bar(df.index,df[col].values)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            f.savefig('Figures_'+benchmarks_experiment+'/stats_wind/'+title+'_'+col+'_'+var+'.png', dpi=500)
    
    rmse_vals = np.arange(0.6,-0.01,-0.05)
    r_vals = np.arange(0.6,1.01,0.05)
    
    for rmse_val, r_val in itertools.product(rmse_vals, r_vals):
        
        rmse_val, r_val = round(rmse_val,2), round(r_val,2)
        
        rmse_norm = df_stats_norm['rmse']
        corrcoef_norm = df_stats_norm['ccoef']
        
        approved_rmse = rmse_norm[rmse_norm <= rmse_val].dropna().index.to_list()
        approved_r = corrcoef_norm[corrcoef_norm >= r_val].dropna().index.to_list()
        
        if len(approved_rmse) > 0 and len(approved_r) > 0:
        
            approved = list(approved_rmse)
            approved.extend(x for x in approved_r if x not in approved)
            
        else:
            
            approved = []
        
        if len(approved) > 0 and len(approved) <= 4:
                    
            print('\nrmse:', rmse_val, 'r:', r_val)
            [print(i) for i in approved]