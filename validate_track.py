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

import pandas as pd
import numpy as np
import xarray as xr

from metpy.units import units

from wrf import interplevel

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.lines import Line2D
from matplotlib import pyplot

import cartopy.crs as ccrs
import cartopy

import dask
import dask.distributed

from geopy.distance import geodesic

colors = {'ERA':'k', 'fritsch':'tab:orange','tiedtke':'tab:red',
          'ntiedtke':'tab:purple', 'freitas':'tab:brown','off':'tab:green'}

lines = {'ERA':'solid', 'wsm6':'dashed','thompson':'dashdot',
         'kessler':(0, (3, 1, 1, 1)),'off':(0, (3, 1, 1, 1, 1, 1))}

markers = {'ERA':'o', 'wsm6':'x', 'thompson':'P','kessler':'D','off':'s'}



def initialize_map(ax, row, col, datacrs):
    ax.set_extent([-55, -30, -20, -35], crs=datacrs) 
    ax.coastlines(zorder = 1)
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
    gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',alpha=0.8,
                  color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.bottom_labels = None
    gl.right_labels = None
    if row != 0:
        gl.top_labels = None
    if col != 0:
        gl.left_labels = None

def make_legend(colors,markers,lines, ax):
    labels, handles = zip(*[(k, mpatches.Rectangle((0, 0), 1, 1, facecolor=v)) for k,v in colors.items()])
    legend1 = ax.legend(handles, labels, loc=4,
                            framealpha=1, bbox_to_anchor=(1.105, 0.27))
    custom_lines = []
    lebels = []
    for line, marker in zip(lines,markers):
        custom_lines.append(Line2D([0], [0], color='k', lw=1,
                                linestyle=lines[line], marker=markers[marker]))
        lebels.append(line)
    legend2 = ax.legend(custom_lines, lebels, loc=4, framealpha=1,
                            bbox_to_anchor=(1.11, 0.1))
    return legend1, legend2    


    

# =============================================================================
# Plot all tracks in one image
# =============================================================================

plt.close('all')
fig = plt.figure(figsize=(15, 12))
datacrs = ccrs.PlateCarree()
ax = fig.add_subplot(1, 1, 1, projection=datacrs)
initialize_map(ax, 0, 0, datacrs)

for exp in data:   
    
    track = data[exp]['track']
    lons, lats, min_slp = track['lon'], track['lat'], track['min']
    
    if exp == 'ERA':
        microp, cumulus = 'ERA', 'ERA'
        zorder=100
    else:
        microp, cumulus = exp.split('_')[0], exp.split('_')[1]
        zorder=1
    ls = lines[microp]
    marker = markers[microp]
    color = colors[cumulus]

    ax.plot(lons ,lats, markeredgecolor=color, marker=marker, zorder=zorder,
                markerfacecolor='None', linewidth=1.5, linestyle=ls,
                c=color, label=exp)
    ax.scatter(lons.iloc[0],lats.iloc[0], s=150, marker=marker, color='gray')
    ax.scatter(lons.iloc[-1],lats.iloc[-1], s=150, marker=marker,
                facecolor=color, zorder=100)
    
legend1, legend2 = make_legend(colors,markers,lines, ax)
ax.add_artist(legend1)
ax.add_artist(legend2)

if args.output is not None:
    fname = args.output
else:
    fname = (args.bench_directory).split('/')[-2].split('.nc')[0]
fname += '_track'
fig.savefig(fname+'.png', dpi=500)
print(fname+'.png created!')

# =============================================================================
# Plot multiple subplots with tracks
# =============================================================================

benchs = glob.glob(args.bench_directory+'/run*')
plt.close('all')
fig = plt.figure(figsize=(10, 13))
gs = gridspec.GridSpec(6, 3)
i = 0
for row in range(6):
    for col in range(3):

        bench = benchs[i]
        ax = fig.add_subplot(gs[row, col], projection=datacrs,frameon=True)
        initialize_map(ax, row, col, datacrs)
        
        expname = bench.split('/')[-1].split('run.')[-1]
        microp = expname.split('.')[0].split('_')[-1]
        cumulus = expname.split('.')[-1].split('_')[-1] 
        experiment = microp+'_'+cumulus
        print(experiment)
        
        ax.text(-50,-22,experiment,bbox=dict(facecolor='w', alpha=0.5))
        
        track = data[experiment]['track']
        
        lons, lats, min_slp = track['lon'], track['lat'], track['min']
        
        ls = lines[microp]
        marker = markers[microp]
        color = colors[cumulus]
            
        ax.plot(lons,lats,zorder=100,markeredgecolor=color,marker=marker,
                    markerfacecolor='None',linewidth=0.5, linestyle=ls,
                    c=color, label=expname)
        
        if args.ERA5:
            track_era = data['ERA']['track']
            lons_era, lats_era = track_era.lon, track_era.lat
            ax.plot(lons_era,lats_era,zorder=1,markeredgecolor='k',
                    marker='o',markerfacecolor='None',
                    linewidth=0.75, linestyle='solid',
                        c='gray', label=expname)
        i+=1
        
fname2 = fname+'_multipanel'
plt.savefig(fname2+'.png', dpi=500)
print(fname2+'.png created!')

# =============================================================================
# Plot minimum slp
# =============================================================================
print('plotting minimum SLP..')
plt.close('all')
fig1 = plt.figure(figsize=(15, 12))
fig2 = plt.figure(figsize=(15, 12))
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)

for exp in data:   
    
    print(exp)
    
    slp = data[exp]['slp']
    track = data[exp]['track']
    
    track.index = pd.DatetimeIndex(track.index)
    track = track.resample('1H').mean()
    
    time, min_slp = track.index, track['min']
        
    print('data range:',min_slp.min(),'to',min_slp.max())
    
    if exp == 'ERA':
        microp, cumulus = 'ERA', 'ERA'
        zorder=100
    else:
        microp, cumulus = exp.split('_')[0], exp.split('_')[1]
        distance = track['distance']
        
    ls = lines[microp]
    marker = markers[microp]
    color = colors[cumulus]
    
    ax1.plot(time,min_slp, markeredgecolor=color, marker=marker,
                markerfacecolor='None', linewidth=1.5, linestyle=ls,
                c=color, label=exp, zorder=zorder)
    if exp != 'ERA':
        ax2.plot(time, distance, markeredgecolor=color, marker=marker,
                    markerfacecolor='None', linewidth=1.5, linestyle=ls,
                    c=color, label=exp, zorder=zorder)
    
fname3 = fname+'_min-slp'
fname4 = fname+'_distance'
for fname, ax in zip([fname3, fname4], [ax1, ax2]):
    legend1, legend2 = make_legend(colors,markers,lines, ax)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    plt.savefig(fname+'.png', dpi=500)
    print(fname+'.png created!')