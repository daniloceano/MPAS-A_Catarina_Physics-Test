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
    
    def __init__(self, path):
        self.bench = path
        self.data = xr.open_dataset(path+'/latlon.nc').chunk({"Time": -1})
        self.data = self.data.assign_coords({"Time":times})
        

    def get_exp_name(self):
        expname = self.bench.split('/')[-1].split('run.')[-1]
        microp = expname.split('.')[0].split('_')[-1]
        cumulus = expname.split('.')[-1].split('_')[-1] 
        return microp+'_'+cumulus
        
def interpolate_mpas_data(variable_data, target):
    return variable_data.rename({'latitude':'lat','longitude':'lon'}
            ).interp(lat=target.lat,lon=target.lon,method='cubic',
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
    
    df = sorted(ff,key=str)
    
    # filenames =[]
    # for i in range(len(df)):
    # 	filenames.append(path+str(df[i])+'.png')
    
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


## Parser options ##
parser = argparse.ArgumentParser()

parser.add_argument('-bdir','--bench_directory', type=str, required=True,
                        help='''path to benchmark directory''')
parser.add_argument('-q','--quickscat', type=str, default=None, required=True,
                        help='''path to QUICKSCAT data''')
parser.add_argument('-o','--output', type=str, default=None,
                        help='''output name to append file''')

args = parser.parse_args()

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
# da_quickscat = xr.open_dataset(args.qs).sel(lat=slice(model_data.latitude[-1],
#                  model_data.latitude[0]),lon=slice(model_data.longitude[0],
#                 model_data.longitude[-1])).sel(time=slice(first_day,last_day))
da_quickscat = convert_lon(xr.open_dataset(args.quickscat),'lon').sel(
                    lat=slice(model_data.latitude[-1],model_data.latitude[0]),
                    lon=slice(model_data.longitude[0],model_data.longitude[-1])
                    ).sel(time=slice(first_day,last_day))
                                     

print('\nOpening all data and putting it into a dictionary...')
benchmarks = [BenchData(bench) for bench in benchs]
data = {}

for benchmark in benchmarks:
    exp_name = benchmark.get_exp_name()
    print('\n',exp_name)
    data[exp_name] = {}
    
    for model_var, quickscat_var in zip(['u10','v10','windspeed'],
                                        ['uwnd','vwnd','windspeed']):
        
        data[exp_name][model_var] = {}
        
        if model_var == 'windspeed':
            variable_experiment_data = wind_speed(benchmark.data['u10'],
                                               benchmark.data['v10'])
            reference_data = wind_speed(da_quickscat['uwnd'],
                                        da_quickscat['vwnd'])  
            
            
        else:
            variable_experiment_data = benchmark.data[model_var]
            reference_data = da_quickscat[quickscat_var]
            
        interpoled_data = interpolate_mpas_data(variable_experiment_data.compute(),
                                                    reference_data)
        
        data[exp_name][model_var]['data'] = variable_experiment_data
        data[exp_name][model_var]['interpoled_data'] = interpoled_data

        resampled_data = interpoled_data.sel(Time=reference_data.time)

        statistical_metrics = sm.taylor_statistics(
            resampled_data.values.ravel(), reference_data.values.ravel())
    
        data[exp_name][model_var].update({metric: statistical_metrics[metric][1] for metric in statistical_metrics.keys()})
    

# =============================================================================
# Make gif
# =============================================================================
print('\nPlotting maps...')
datacrs = ccrs.PlateCarree()

levels = np.arange(0,30,2)
skip = (slice(None, None, 2), slice(None, None, 2))

for t in times:
    plt.close('all')
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(6, 3)
    print('\n',t)
    i = 0
    for col in range(3):
        for row in range(6):
            bench = benchs[i]
            benchmark_data = BenchData(bench)
            experiment = benchmark_data.get_exp_name()
            
            u = data[experiment]['u10']['data'].sel(Time=t)
            v = data[experiment]['v10']['data'].sel(Time=t)
            windspeed = data[experiment]['windspeed']['data'].sel(Time=t)
            
                
            ax = fig.add_subplot(gs[row, col], projection=datacrs,frameon=True)
            
            ax.set_extent([-55, -30, -20, -35], crs=datacrs) 
            gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',
                              alpha=0.8, color='#383838')
            gl.xlabel_style = {'size': 12, 'color': '#383838'}
            gl.ylabel_style = {'size': 12, 'color': '#383838'}
            gl.right_labels = None
            gl.top_labels = None
            if row != 5:
                gl.bottom_labels = None
            if col != 0:
                gl.left_labels = None
        
            ax.text(-50,-19,experiment)
            
            cf = ax.contourf(windspeed.longitude, windspeed.latitude, windspeed,
                              cmap='rainbow', levels=levels)
        
            # ax.quiver(u.longitude[::20], u.latitude[::20],
            #           u[::20,::20], v[::20,::20], width=0.025,headaxislength=3.5)
            ax.streamplot(u.longitude[::20].values, u.latitude[::20].values,
                          u[::20,::20].values, v[::20,::20].values, color='k')
            
            ax.coastlines(zorder = 1)
        i += 1
    
    cb_axes = fig.add_axes([0.85, 0.18, 0.04, 0.6])
    fig.colorbar(cf, cax=cb_axes, orientation="vertical") 
    fig.subplots_adjust(wspace=0.1,hspace=0, right=0.8)
    
    # if 'args.output is not None:
    #     fname = args.output
    # else:
    #     fname = (args.'bench_directory).split('/')[-2].split('.nc')[0]
    # fname1 = fname+'_wind-maps'
    
    time_string = t.strftime("%d-%HZ")
    fname = './Figures_48h/wind/wind-test'+'_'+time_string+'.png'
    
    fig.savefig(fname, dpi=500)
    print(fname,'saved')
    
    anim('./Figures_48h/wind/','wind-test')