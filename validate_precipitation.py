#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:52:10 2023

@author: daniloceano
"""
import glob
import argparse
import f90nml
import datetime
import itertools

import numpy as np
import pandas as pd
import xarray as xr
import cmocean.cm as cmo
import cartopy.crs as ccrs

import scipy.stats as st
import skill_metrics as sm

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

def willmot_d_index(observed, modeled):
    obs_mean = observed.mean()
    mod_mean = modeled.mean()
    
    numerator = np.sum(np.abs(observed - modeled))
    denominator = np.sum(np.abs(observed - obs_mean)) + np.sum(np.abs(modeled - mod_mean))
        
    return  1 - (numerator / denominator)

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
parser.add_argument('-i','--imerg', type=str, default=None, required=True,
                        help='''path to IMERG data''')
parser.add_argument('-o','--output', type=str, default=None,
                        help='''output name to append file''')

args = parser.parse_args()

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
    
    stats = sm.taylor_statistics(imerg_accprec.values.ravel(),
                                 interp.values.ravel())
    
    data[experiment] = {}
    data[experiment]['data'] = acc_prec
    data[experiment]['interp'] = interp
    data[experiment]['ccoef'] = (stats['ccoef'][1])
    data[experiment]['crmsd'] = (stats['crmsd'][1])
    data[experiment]['sdev'] = (stats['sdev'][1])
    data[experiment]['willmot_d_index'] = willmot_d_index(
        imerg_accprec.values.ravel(),interp.values.ravel())

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

prec_levels = np.arange(0,425,4)
bias_levels = np.arange(-700,411,10)
bias_norm = colors.TwoSlopeNorm(vmin=-700, vcenter=0, vmax=411)

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
            gl.xlabel_style = {'size': 12, 'color': '#383838'}
            gl.ylabel_style = {'size': 12, 'color': '#383838'}
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
fname1 = 'Figures_'+benchmarks+'/'+fname+'_acc_prec'
fname2 = 'Figures_'+benchmarks+'/'+fname+'_acc_prec_bias'
fig1.savefig(fname1+'.png', dpi=500)
fig2.savefig(fname2+'.png', dpi=500)
print(fname1,'and',fname1,'saved')

# =============================================================================
# Plot IMERG ac prec
# =============================================================================
print('\nPlotting IMERG data..')
plt.close('all')
fig = plt.figure(figsize=(10, 10))
datacrs = ccrs.PlateCarree()
ax = fig.add_subplot(111, projection=datacrs,frameon=True)
ax.set_extent([-55, -30, -20, -35], crs=datacrs) 
gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',
                  alpha=0.8, color='#383838')
gl.xlabel_style = {'size': 12, 'color': '#383838'}
gl.ylabel_style = {'size': 12, 'color': '#383838'}
gl.right_labels = None
gl.top_labels = None
cf = ax.contourf(imerg_accprec.lon, imerg_accprec.lat,
                 imerg_accprec, cmap=cmo.rain,
                 levels=np.arange(0,imerg_accprec.max()+2,2))
fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.1)
ax.coastlines(zorder = 1)

imergname = args.imerg.split('/')[-1].split('.nc')[0]
fig.savefig('Figures_'+benchmarks+'/'+imergname+'.png', dpi=500)
print('Figures_'+benchmarks+'/'+imergname,'saved')

# =============================================================================
# PDFs
# =============================================================================
print('\nPlotting PDFs..')
nbins = 100
params_imerg = st.weibull_min.fit(imerg_accprec.values.ravel())
x_imerg = np.linspace(st.weibull_min.ppf(0.01, *params_imerg),
                st.weibull_min.ppf(0.99, *params_imerg), nbins)
pdf_imerg = st.weibull_min.pdf(x_imerg, *params_imerg)

plt.close('all')
fig = plt.figure(figsize=(10, 16))
gs = gridspec.GridSpec(6, 3)


i = 0
for col in range(ncol):
    for row in range(nrow):
        
        if i == imax:
            break
    
        ax = fig.add_subplot(gs[row, col], frameon=True)
    
        bench = benchs[i]
        experiment = get_exp_name(bench)
        print('\n',experiment)
        
        reference = imerg_accprec.values.ravel()
        predicted =  data[experiment]['interp'].values.ravel()
        
        if experiment != 'off_off':
                 
            ax.hist(reference, bins=nbins, color='k', lw=1, alpha=0.3,
                    density=True, histtype='step',label='IMERG', zorder=1) 
            
            ax.hist(predicted, bins=nbins, color='tab:red',  lw=1, alpha=0.3,
                    density=True, histtype='step', label=experiment, zorder=100)

            # ax.set_xscale('log')
            ax.set_yscale('log')  
            ax.text
            ax.text(50, len(reference)*.1,experiment)
            
            i+=1
            
            
fig.subplots_adjust(hspace=0.25)
fig.savefig('Figures_'+benchmarks+'/'+fname+'_PDF.png', dpi=500)    
print('Figures_'+benchmarks+'/'+fname+'_PDF','saved')

# =============================================================================
# Plot Taylor Diagrams and do Statistics ##
# =============================================================================
ccoef = [data[exp]['ccoef'] for exp in data.keys() if exp != 'IMERG']
crmsd  = [data[exp]['crmsd'] for exp in data.keys() if exp != 'IMERG']
sdev = [data[exp]['sdev'] for exp in data.keys() if exp != 'IMERG']
d_index = [
    data[exp]['willmot_d_index'] for exp in data.keys() if exp != 'IMERG']
ccoef, crmsd, sdev = np.array(ccoef),np.array(crmsd),np.array(sdev)
print('plotting taylor diagrams..')
fig = plt.figure(figsize=(10,10))
plot_taylor(sdev,crmsd,ccoef,list(data.keys()))
plt.tight_layout(w_pad=0.1)
fig.savefig('Figures_'+benchmarks+'/'+fname+'_prec-taylor.png', dpi=500)    
print('Figures_'+benchmarks+'/'+fname+'_prec-taylor created!')


df_stats = pd.DataFrame(crmsd,
                       index=[exp for exp in data.keys() if exp != 'IMERG'],
                       columns=['rmse'])
df_stats['ccoef'] = ccoef
df_stats['d_index'] = d_index


# Normalize values for comparison
df_stats_norm = (df_stats-df_stats.min()
                  )/(df_stats.max()-df_stats.min()) 
df_stats_norm.sort_index(ascending=True).to_csv(
    './stats-'+benchmarks+'/precip_RMSE_normalised.csv')

for data, title in zip([df_stats, df_stats_norm],
                       ['stats', 'stats normalised']):
    for col in data.columns:
        plt.close('all')
        f, ax = plt.subplots(figsize=(10, 10))
        ax.bar(data.index,data[col].values)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        f.savefig('Figures_'+benchmarks+'/stats_prec/'+title+'_'+col+'.png', dpi=500)

rmse_vals = np.arange(0.6,-0.01,-0.05)
r_vals = np.arange(0.6,1.01,0.05)
d_vals = np.arange(0.6,1.01,0.05)

for rmse_val, r_val, d_val in itertools.product(rmse_vals, r_vals, d_vals):
    
    rmse_val, r_val, d_val = round(rmse_val,2), round(r_val,2), round(d_val,2)
    
    rmse_norm = df_stats_norm['rmse']
    corrcoef_norm = df_stats_norm['ccoef']
    d_norm = df_stats_norm['d_index']
    
    approved_rmse = rmse_norm[rmse_norm <= rmse_val].dropna().index.to_list()
    approved_r = corrcoef_norm[corrcoef_norm >= r_val].dropna().index.to_list()
    approved_d = d_norm[d_norm >= d_val].dropna().index.to_list()
    
    if len(approved_rmse) > 0 and len(approved_r) > 0 and len(approved_d) > 0:
    
        approved = list(approved_rmse)
        approved.extend(x for x in approved_r if x not in approved)
        approved.extend(x for x in approved_d if x not in approved)
        
    else:
        
        approved = []
    
    if len(approved) > 0 and len(approved) <= 4:
                
        print('\nrmse:', rmse_val, 'r:', r_val, 'd:', d_val)
        [print(i) for i in approved]