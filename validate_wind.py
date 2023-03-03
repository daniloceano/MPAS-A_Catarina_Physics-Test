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

def convert_lon(df,LonIndexer):
    df.coords[LonIndexer] = (df.coords[LonIndexer] + 180) % 360 - 180
    df = df.sortby(df[LonIndexer])
    return df

def anim(path,pattern):
    filenames = glob.glob(path+pattern)
    ff=[]
    for i in range(len(filenames)):
    	ff.append(int(filenames[i].split('/')[-1].split('.')[0]))
    
    df = sorted(ff,key=int)
    
    filenames =[]
    for i in range(len(df)):
    	filenames.append(path+str(df[i])+'.png')
    
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

def get_exp_name(bench):
    expname = bench.split('/')[-1].split('run.')[-1]
    microp = expname.split('.')[0].split('_')[-1]
    cumulus = expname.split('.')[-1].split('_')[-1] 
    return microp+'_'+cumulus

quickfile = '/home/daniloceano/Documents/MPAS/MPAS-BR/met_data/QUICKSCAT/Catarina_20040321-20040323_v11l30flk.nc'

benchdata = '/home/daniloceano/Documents/MPAS/MPAS-BR/benchmarks/Catarina_physics-test/Catarina_250-8km.microp_scheme.convection_scheme'


## Start the code ##
# benchs = glob.glob(args.bench_directory+'/run*')
benchs = glob.glob(benchdata+'/run*')
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
da_quickscat = convert_lon(xr.open_dataset(quickfile),'lon').sel(
                    lat=slice(model_data.latitude[-1],model_data.latitude[0]),
                    lon=slice(model_data.longitude[0],model_data.longitude[-1])
                    ).sel(time=slice(first_day,last_day))
print(da_quickscat)                                               
u_quickscat = da_quickscat.uwnd
v_quickscat = da_quickscat.vwnd  
windspeed_quickscat = wind_speed(u_quickscat,v_quickscat)                                          

print('\nOpening all data and putting it into a dictionary...')
data = {}
for bench in benchs:
    
    experiment = get_exp_name(bench)
    print('\n',experiment)
    
    model_data = xr.open_dataset(bench+'/latlon.nc').chunk({"Time": -1})
    model_data = model_data.assign_coords({"Time":times})


    u = model_data.u10
    u_interp = u.rename({'longitude':'lon','latitude':'lat'}).sel(
        Time=times[0:-1:6]).interp(lat=da_quickscat.lat, lon=da_quickscat.lon,
                                      method='cubic', assume_sorted=False)
    stats_u = sm.taylor_statistics(u_quickscat.values.ravel(),
                                  u_interp.values.ravel())
        
    v = model_data.v10
    v_interp = v.rename({'longitude':'lon','latitude':'lat'}).sel(
        Time=times[0:-1:6]).interp(lat=da_quickscat.lat, lon=da_quickscat.lon,
                                  method='cubic', assume_sorted=False)
    stats_v = sm.taylor_statistics(v_quickscat.values.ravel(),
                                  v_interp.values.ravel())
    
    windspeed = wind_speed(u,v)
    windspeed_interp = wind_speed(u_interp,v_interp)
    stats_ws = sm.taylor_statistics(windspeed_quickscat.values.ravel(),
                                  windspeed_interp.values.ravel())
    
    print('limits for data:',round(float(windspeed.min()),2),
          round(float(windspeed.max()),2))
    print('limits for interp data:',round(float(windspeed_interp.min()),2),
          round(float(windspeed_interp.max()),2))
    
    data[experiment] = {}
    data[experiment]['u'] = u
    data[experiment]['v'] = v
    data[experiment]['windspeed'] = windspeed
    
    data[experiment]['ccoef_u'] = (stats_u['ccoef'][1])
    data[experiment]['crmsd_u'] = (stats_u['crmsd'][1])
    data[experiment]['sdev_u'] = (stats_u['sdev'][1])
    
    data[experiment]['ccoef_v'] = (stats_v['ccoef'][1])
    data[experiment]['crmsd_v'] = (stats_v['crmsd'][1])
    data[experiment]['sdev_v'] = (stats_v['sdev'][1])
    
    data[experiment]['ccoef_ws'] = (stats_ws['ccoef'][1])
    data[experiment]['crmsd_ws'] = (stats_ws['crmsd'][1])
    data[experiment]['sdev_ws'] = (stats_ws['sdev'][1])
    
# =============================================================================
# Make gif
# =============================================================================
print('\nPlotting maps...')
datacrs = ccrs.PlateCarree()

levels = np.arange(0,30,2)

for t in times:
    plt.close('all')
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(6, 3)
    print('\n',t)
    i = 0
    for col in range(3):
        for row in range(6):
            
            try:
                bench = benchs[i]
                experiment = get_exp_name(bench)
                
                u = data[experiment]['u'].sel(Time=t)
                v = data[experiment]['v'].sel(Time=t)
                windspeed = data[experiment]['windspeed'].sel(Time=t)
                
                    
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
                
                ax.coastlines(zorder = 1)
            except:
                pass
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
    
    anim('./Figures_48h/','wind-test')