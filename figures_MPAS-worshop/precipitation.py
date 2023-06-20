# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    precipitation.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/06/20 11:48:29 by Danilo            #+#    #+#              #
#    Updated: 2023/06/20 16:34:45 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import glob
import argparse
import f90nml
import datetime
import os

import numpy as np
import pandas as pd
import xarray as xr
import cmocean.cm as cmo
import cartopy.crs as ccrs

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

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
    times = pd.date_range(start_date,finish_date,periods=len(model_data.Time)+1)[1:]
    return times

def get_exp_name(bench):
    expname = bench.split('/')[-1].split('run.')[-1]
    microp = expname.split('.')[0].split('_')[-1]
    cumulus = expname.split('.')[-1].split('_')[-1] 
    return microp+'_'+cumulus

def get_model_accprec(model_data):
    if ('rainnc' in model_data.variables
        ) and ('rainc' in model_data.variables):
        acc_prec = model_data['rainnc']+model_data['rainc']
    # Get only micrphysics precipitation
    elif ('rainnc' in model_data.variables
        ) and ('rainc' not in model_data.variables):
        acc_prec = model_data['rainnc']
    # Get convective precipitation
    elif ('rainnc' not in model_data.variables
        ) and ('rainc' in model_data.variables):
        acc_prec = model_data['rainc'] 
    elif ('rainnc' not in model_data.variables
        ) and ('rainc' not in model_data.variables):
        acc_prec = model_data.uReconstructMeridional[0]*0
    return acc_prec[-1]

## Parser options ##
parser = argparse.ArgumentParser()

parser.add_argument('-bdir','--bench_directory', type=str, required=True,
                        help='''path to benchmark directory''')
parser.add_argument('-i','--imerg', type=str, default=None, required=True,
                        help='''path to IMERG data''')
parser.add_argument('-o','--output', type=str, default=None,
                        help='''output name to append file''')

#args = parser.parse_args()

args = parser.parse_args(['-bdir', '/p1-nemo/danilocs/mpas/MPAS-BR/benchmarks/Catarina_physics-test/Catarina_250-8km.best-physics_sst/',
                           '-i', '/p1-nemo/danilocs/mpas/MPAS-BR/met_data/IMERG/IMERG_20040321-20040325.nc'])

benchmarks = input("prompt experiments (24h, 48h, 48h_sst, 72h_sst, '2403-2903'): ")

if (benchmarks == '48h_sst') or (benchmarks == '72h_sst'):
    ncol, nrow, imax = 2, 2, 3
elif benchmarks == '2403-2903':
    ncol, nrow, imax = 1, 1, 1
else:
    ncol, nrow, imax = 3, 6, 18
print('Figure will have ncols:', ncol, 'rows:', nrow, 'n:', imax)

## Start the code ##
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
imerg = xr.open_dataset(args.imerg).sel(lat=slice(model_data.latitude[-1],
                 model_data.latitude[0]),lon=slice(model_data.longitude[0],
                model_data.longitude[-1])).sel(time=slice(first_day,last_day))
print(imerg)                                               
                                                   
print('Using IMERG data from',first_day,'to',last_day)                                             
imerg_accprec = imerg.precipitationCal.cumsum(dim='time')[-1].transpose(
    'lat', 'lon')
print('Maximum acc prec:',float(imerg_accprec.max()))

print('\nOpening all data and putting it into a dictionary...')
data = {}
data['IMERG'] = imerg_accprec

for bench in benchs:
    
    experiment = get_exp_name(bench)
    print('\n',experiment)
    
    model_data = xr.open_dataset(bench+'/latlon.nc').chunk({"Time": -1})
    model_data = model_data.assign_coords({"Time":times})

    acc_prec = get_model_accprec(model_data)
    acc_prec = acc_prec.where(acc_prec >= 0, 0)
    acc_prec_interp = acc_prec.interp(latitude=imerg_accprec.lat,
                                      longitude=imerg_accprec.lon,
                                      method='cubic',assume_sorted=False)
    interp =  acc_prec_interp.where(acc_prec_interp >=0, 0).transpose(
        'lat', 'lon')
    
    print('limits for prec data:',float(acc_prec.min()),float(acc_prec.max()))
    print('limits for interp prec data:',float(acc_prec_interp.min()),
          float(acc_prec_interp.max()))
    
    data[experiment] = {}
    data[experiment]['data'] = acc_prec
    data[experiment]['interp'] = interp

# =============================================================================
# Plot acc prec maps and bias
# =============================================================================
print('\nPlotting maps...')
plt.close('all')
fig1 = plt.figure(figsize=(10, 12))
fig2 = plt.figure(figsize=(10, 12))
gs1 = gridspec.GridSpec(6, 3)
gs2 = gridspec.GridSpec(6, 3)
datacrs = ccrs.PlateCarree()

if (benchmarks == '48h_sst'):
    prec_levels = np.arange(0,375,25)
    bias_levels = np.arange(-200,225,25)
    bias_norm = colors.TwoSlopeNorm(vmin=-200, vcenter=0, vmax=200)
elif (benchmarks == '72h_sst'):
    prec_levels = np.arange(0,475,25)
    bias_levels = np.arange(-250,275,25)
    bias_norm = colors.TwoSlopeNorm(vmin=-250, vcenter=0, vmax=250)

i = 0
for col in range(ncol):
    for row in range(nrow):
        
        if i == imax:
            break
        
        bench = benchs[i]
        experiment = get_exp_name(bench)
        print('\n',experiment)
        
        prec = data[experiment]['data']
        prec_interp = data[experiment]['interp']
        
        for fig in [fig1,fig2]:
            
            ax = fig.add_subplot(gs1[row, col], projection=datacrs,frameon=True)
            
            ax.set_extent([-55, -30, -20, -35], crs=datacrs) 
            gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',
                              alpha=0.8, color='#383838')
            gl.xlabel_style = {'size': 16, 'color': '#383838'}
            gl.ylabel_style = {'size': 16, 'color': '#383838'}
            gl.right_labels = None
            gl.top_labels = None
            if row != 5:
                gl.bottom_labels = None
            if col != 0:
                gl.left_labels = None
        
            ax.text(-50,-19,experiment)
            
            if fig == fig1:
                print('Plotting accumulate prec..')
                cf1 = ax.contourf(prec.longitude, prec.latitude, prec,
                                  cmap=cmo.rain, levels=prec_levels)
                print('prec limits:',float(prec.min()), float(prec.max()))
            else:
                print('Plotting bias..')
                bias = prec_interp-imerg_accprec
                cf2 = ax.contourf(imerg_accprec.lon, imerg_accprec.lat,bias,
                                 cmap=cmo.balance_r,
                                 levels=bias_levels, norm=bias_norm)
                print('bias limits:',float(bias.min()), float(bias.max()))
            ax.coastlines(zorder = 1)
        i+=1

for fig, cf in zip([fig1, fig2], [cf1, cf2]):
    cb_axes = fig.add_axes([0.85, 0.18, 0.04, 0.6])
    fig.colorbar(cf, cax=cb_axes, orientation="vertical") 
    fig.subplots_adjust(wspace=0.1,hspace=0, right=0.8)
    
if args.output is not None:
    fname = args.output
else:
    fname = (args.bench_directory).split('/')[-2].split('.nc')[0]

directory = f'./precipitation_{benchmarks}'
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the figure 
fname1 = os.path.join(directory, f'{fname}_accprec.png')
fname2 = os.path.join(directory, f'{fname}_acc_prec_bias.png')
fig1.savefig(fname1, dpi=300)
print("Saved {}".format(fname1))
fig2.savefig(fname2, dpi=300)
print("Saved {}".format(fname2))

# =============================================================================
# Plot IMERG acc prec
# =============================================================================
print('\nPlotting IMERG data..')
plt.close('all')
fig = plt.figure(figsize=(10, 10))
datacrs = ccrs.PlateCarree()
ax = fig.add_subplot(111, projection=datacrs,frameon=True)
ax.set_extent([-55, -30, -20, -35], crs=datacrs) 
gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',
                  alpha=0.8, color='#383838')
gl.xlabel_style = {'size': 16, 'color': '#383838'}
gl.ylabel_style = {'size': 16, 'color': '#383838'}
gl.right_labels = None
gl.top_labels = None
cf = ax.contourf(imerg_accprec.lon, imerg_accprec.lat,
                 imerg_accprec, cmap=cmo.rain,
                 levels=prec_levels)
fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.1)
ax.coastlines(zorder = 1)

imergname = args.imerg.split('/')[-1].split('.nc')[0]
fname1 = os.path.join(directory, f'{imergname}_accprec.png')
plt.savefig(fname1, dpi=300)
print("Saved {}".format(fname1))
