#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:32:22 2023

@author: daniloceano
"""

import glob

import pandas as pd

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy

import cmocean.cm as cmo

import seaborn as sns

def initialize_map(ax, row, col, datacrs):
    ax.set_extent([-52, -37, -25, -33], crs=datacrs) 
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
        

def tracks_one_image(tracks, directory):
    plt.close('all')
    fig = plt.figure(figsize=(15, 12))
    
    ax = fig.add_subplot(1, 1, 1, projection=datacrs)
    initialize_map(ax, 0, 0, datacrs)
    
    track_MPAS = pd.read_csv('track_20040324_20040324.csv', index_col=0)
    
    for trackfile in tracks:   
        
        if 'Cowan' in trackfile:
            exp = 'Cowan'
        elif 'ERA' in trackfile:
            exp = 'ERA5'
            continue
        else:
            exp = 'MPAS'
        print(exp)
        
        # Use only the dates contained in the MPAS file
        track = pd.read_csv(trackfile, index_col=0)
        track = track.loc[track_MPAS.index]
        lons, lats = track['lon'], track['lat']
            
        if exp == 'Cowan':
            marker = 'D'
            color = 'k'
        else:
            marker = 's'
            color = 'tab:blue'
            
        ax.plot(lons ,lats, markeredgecolor=color, marker=marker,
                    markerfacecolor='none', linewidth=3, linestyle='-',
                    c=color, label=exp)
        ax.scatter(lons.iloc[-1],lats.iloc[-1], s=150, marker=marker,
                    facecolor='none',zorder=100, edgecolor=color)
    
        track.index = pd.to_datetime(track.index)
        
        for index, row in track.iterrows():
            
            if index.strftime('%H:%M:%S') == '00:00:00':
            
                ax.scatter(row['lon'] ,row['lat'], edgecolor='k',
                           marker=marker, facecolor=color, linewidth=2,
                           linestyle='-', s=100)
                
                if (exp == 'Cowan') or (exp == 'MPAS' and index.day >= 27):
                    ax.text(row['lon'], row['lat']+0.5,
                            index.strftime('%m-%d'), ha='center', va='center',
                            fontsize=18)
    plt.legend(fontsize=16)
    
    fname = directory+'tracks'
    fig.savefig(fname+'.png', dpi=500)
    print(fname+'.png created!')
    
def min_slp(tracks, directory):
    
    print('plotting minimum SLP..')
    plt.close('all')
    fig1 = plt.figure(figsize=(10, 5))
    fig2 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    
    stats = {}
    distances = {}
    mins = {}
    
    track_dummy = pd.read_csv(tracks[0], index_col=0)
    track_Cowan = pd.read_csv('./track_Cowan.csv', index_col=0)
    track_Cowan_sliced = track_Cowan.loc[
        slice(track_dummy.index[0],track_dummy.index[-1])]
    
    for trackfile in tracks:   
        
        exp = trackfile.split('.csv')[0].split('track_')[1]
        print(exp)
        
        track = pd.read_csv(trackfile, index_col=0)    
        track = track.loc[track_Cowan_sliced.index]
        time, min_slp = pd.to_datetime(track.index), track['min']
        mins[exp] = min_slp
        print('data range:',min_slp.min(),'to',min_slp.max())
        
        if 'Cowan' in trackfile:
            exp = 'Cowan'
            marker = 'D'
            color = 'k'
        elif 'ERA' in trackfile:
            exp = 'ERA5'
            continue
        else:
            exp = 'MPAS'
            marker = 'o'
            color = 'tab:blue'
            
            distance = track['distance']
            mean_dist = distance.mean()
            std_dist = distance.std()
            stats[exp] = [mean_dist,std_dist]
            
            distances[exp] = distance
                    
        print(exp)
        
        ax1.plot(time,min_slp, markeredgecolor=color, marker=marker,
                    markerfacecolor='None', linewidth=3,
                    c=color, label=exp)
        
        if exp != 'Cowan' and exp !='ERA5':
            ax2.plot(time, distance, markeredgecolor=color, marker=marker,
                        markerfacecolor='None', linewidth=3,
                        c=color, label=exp)
        
        ax1.grid(color='gray', linewidth=0.5)
        ax2.grid(color='gray', linewidth=0.5)
        ax1.set_ylabel('Minimum SLP', fontsize=16)
        ax2.set_ylabel('Distance (km)', fontsize=16)
        ax1.legend(fontsize=18)
        fig1.savefig(directory+'min_slp.png')
        fig2.savefig(directory+'distance.png')
    
if __name__ == '__main__':

    tracks = glob.glob('track*')
    datacrs = ccrs.PlateCarree()
    directory = '../Figures_2403-2903/tracks/'
    # tracks_one_image(tracks, directory)
    min_slp(tracks, directory)