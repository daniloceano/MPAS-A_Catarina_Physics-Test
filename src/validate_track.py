#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:52:10 2023

@author: daniloceano
"""
import os
import glob
import argparse
import cmocean

import pandas as pd
import seaborn as sns

import cartopy
import cartopy.crs as ccrs

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

colors = {'ERA':'k', 'fritsch':'tab:orange','tiedtke':'tab:red',
          'ntiedtke':'tab:purple', 'freitas':'tab:brown','off':'tab:green',
          'Cowan': 'k'}

lines = {'ERA':'solid', 'wsm6':'dashed','thompson':'dashdot',
         'kessler':(0, (3, 1, 1, 1)),'Cowan':'solid'}

lines_bl = {'ERA':'solid', 'Cowan':'solid', 'ysu':'dashed', 'mynn':'dashdot'}

markers = {'ERA':'s', 'wsm6':'x', 'thompson':'P','kessler':'D','Cowan':'o'}

datacrs = ccrs.PlateCarree()

def get_exp_name(bench, pbl=None):
    """
    Extracts the experiment name from a given benchmark file path.

    Parameters:
        bench (str): The path to the benchmark file.

    Returns:
        str: The experiment name, which consists of the microp, cumulus, and pbl values separated by underscores.
    """
    expname = os.path.basename(bench)
    if any(x in expname for x in ['ysu', 'mynn']):
        _, _, microp, cumulus, pbl =  expname.split('.')
        pbl = pbl.split('_')[-1]
    elif "convection" in expname:
        _, microp, cumulus = expname.split('.')
    else:
        _, _, microp, cumulus =  expname.split('.')
    microp = microp.split('_')[-1]
    cumulus = cumulus.split('_')[-1]
    if pbl is not None:
        return microp+'_'+cumulus+'_'+pbl
    else:
        return microp+'_'+cumulus

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
                            framealpha=1, bbox_to_anchor=(1.105, 0.3))
    custom_lines = []
    lebels = []
    for line, marker in zip(lines,markers):
        custom_lines.append(Line2D([0], [0], color='k', lw=1,
                                linestyle=lines[line], marker=markers[marker]))
        lebels.append(line)
    legend2 = ax.legend(custom_lines, lebels, loc=4, framealpha=1,
                            bbox_to_anchor=(1.11, 0.1))
    return legend1, legend2    

def tracks_one_image(tracks, tracks_directory, figures_directory):
    plt.close('all')
    fig = plt.figure(figsize=(15, 12))
    
    ax = fig.add_subplot(1, 1, 1, projection=datacrs)
    initialize_map(ax, 0, 0, datacrs)
    
    track_dummy = pd.read_csv(tracks[0], index_col=0)
    
    track_Cowan = pd.read_csv(f'{tracks_directory}/track_Cowan.csv', index_col=0)
    track_Cowan_sliced = track_Cowan.loc[
        slice(track_dummy.index[0],track_dummy.index[-1])]
    
    for trackfile in tracks:   
        
        exp = os.path.basename(trackfile).split('.csv')[0].split('track_')[1]
        if 'sst' in exp:
            exp = exp.split('sst_')[1]

        if "off_" in exp: continue

        if "off_" in exp:
            print('skipping', exp)
        else:
            print(f'Plotting tracks for {exp}...')
        
        track = pd.read_csv(trackfile, index_col=0)
        track = track.loc[track_Cowan_sliced.index]
        lons, lats = track['lon'], track['lat']
        
        
        if exp == 'ERA5':
            microp, cumulus, bl = 'ERA', 'ERA', 'ERA'
            zorder=100
            
        elif exp == 'Cowan':
            microp, cumulus, bl = 'Cowan', 'Cowan', 'Cowan'
            zorder=101
            
        else:
            if any(substring in exp for substring in ['ysu', 'mynn']):
                    print(f"\nDEBUG: {exp.split('_')}")
                    microp, cumulus, bl = exp.split('_')
            else:
                microp, cumulus = exp.split('_')[0], exp.split('_')[1]
                bl = None
            zorder=1

        if bl:
            ls = lines_bl[bl]
        else:
            ls = lines[microp]

        marker = markers[microp]
        color = colors[cumulus]
            
        ax.plot(lons ,lats, markeredgecolor=color, marker=marker, zorder=zorder,
                    markerfacecolor='None', linewidth=1.5, linestyle=ls,
                    c=color, label=exp)
        ax.scatter(lons.iloc[0],lats.iloc[0], s=150, marker=marker,
                   facecolor='None',color=color)
        ax.scatter(lons.iloc[-1],lats.iloc[-1], s=150, marker=marker,
                    facecolor=color, zorder=100)
        
    legend1, legend2 = make_legend(colors,markers,lines, ax)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    fname = os.path.join(figures_directory, 'tracks.png')
    fig.savefig(fname, dpi=500)
    print(f'{fname} created!')
    
def tracks_subplots(tracks, tracks_dir, figures_directory):
    
    tracks = [elem for elem in tracks if 'ERA5' not in elem]
    tracks = [elem for elem in tracks if 'Cowan' not in elem]
    
    plt.close('all')
    fig = plt.figure(figsize=(10, 13))
    gs = gridspec.GridSpec(6, 3)
    
    track_era = pd.read_csv(f'{tracks_dir}/track_ERA5.csv', index_col=0)
    
    track_Cowan = pd.read_csv(f'{tracks_dir}/track_Cowan.csv', index_col=0)
    track_Cowan_sliced = track_Cowan.loc[
        slice(track_era.index[0],track_era.index[-1])]
    lons_cowan, lats_cowan = track_Cowan_sliced.lon, track_Cowan_sliced.lat
    
    track_era_sliced = track_era.loc[track_Cowan_sliced.index]
    lons_era, lats_era = track_era_sliced.lon, track_era_sliced.lat
    
    i = 0
    for row in range(6):
        for col in range(3):
            
            trackfile = tracks[i]
            
            ax = fig.add_subplot(gs[row, col], projection=datacrs,frameon=True)
            initialize_map(ax, row, col, datacrs)
            
            expname = trackfile.split('/')[-1].split('track_')[-1].split('.csv')[0]
            microp = expname.split('_')[0]
            cumulus = expname.split('_')[1]
            experiment = microp+'_'+cumulus
            print(experiment)
            if microp == 'off': continue
            
            ax.text(-50,-22,experiment,bbox=dict(facecolor='w', alpha=0.5))
            
            track = pd.read_csv(trackfile, index_col=0)
            track_sliced = track.loc[track_Cowan_sliced.index]
            lons, lats = track_sliced['lon'], track_sliced['lat']
            
            ls = lines[microp]
            marker = markers[microp]
            color = colors[cumulus]
                
            ax.plot(lons,lats,zorder=100,markeredgecolor=color,marker=marker,
                        markerfacecolor='None',linewidth=0.5, linestyle=ls,
                        c=color, label=expname)
            
            ax.plot(lons_era,lats_era,zorder=1,markeredgecolor='gray',
                    marker='s',markerfacecolor='None',
                    linewidth=0.5, linestyle='solid',
                        c='gray', label=expname)
            
            ax.plot(lons_cowan,lats_cowan,zorder=1,markeredgecolor='k',
                    marker='o',markerfacecolor='None',
                    linewidth=0.75, linestyle='solid',
                        c='k', label=expname)
                
            i+=1
            
            
    fname = figures_directory+'tracks_multipanel'
    plt.savefig(fname+'.png', dpi=500)
    print(fname+'.png created!')

def minimum_slp_and_distance(tracks, tracks_directory, tracks_figures_directory, stats_directory):
    
    print('plotting minimum SLP..')
    plt.close('all')

    fig1 = plt.figure(figsize=(15, 12))
    fig2 = plt.figure(figsize=(15, 12))

    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    
    stats = {}
    distances = {}
    mins = {}
    
    track_dummy = pd.read_csv(tracks[0], index_col=0)
    track_Cowan = pd.read_csv(f'{tracks_directory}/track_Cowan.csv', index_col=0)
    track_Cowan_sliced = track_Cowan.loc[slice(track_dummy.index[0],track_dummy.index[-1])]
    
    for trackfile in tracks:   
        
        exp = os.path.basename(trackfile).split('.csv')[0].split('track_')[1]
        print(exp)
        
        if 'off_' in exp: continue
        
        track = pd.read_csv(trackfile, index_col=0)    
        track = track.loc[track_Cowan_sliced.index]
        time, min_slp = pd.to_datetime(track.index), track['min']
        
        mins[exp] = min_slp
                    
        print('data range:',min_slp.min(),'to',min_slp.max())
            
        if exp == 'Cowan':
            microp, cumulus, bl, zorder = 'Cowan', 'Cowan', 'Cowan', 101
            x_label_color = 'black'

        else:
            distance = track['distance']
            mean_dist = distance.mean()
            std_dist = distance.std()
            stats[exp] = [mean_dist,std_dist]
            distances[exp] = distance
            zorder = 1
            x_label_color = 'black'

            if any(substring in exp for substring in ['ysu', 'mynn']):
                if 'freitas' in exp:
                    microp, _, cumulus, bl = exp.split('_')
                else:
                    microp, cumulus, bl = exp.split('_')

            elif any(substring in exp for substring in 'ERA'):
                microp, cumulus, bl, zorder = 'ERA', 'ERA', 'ERA', 100
                x_label_color = 'red'

            else:
                microp, cumulus = exp.split('_')[0], exp.split('_')[1]
                bl = None

        if bl:
            ls = lines_bl[bl]
        else:
            ls = lines[microp]

        if microp == 'off': continue

        marker = markers[microp]
        color = colors[cumulus]
        
        ax1.plot(time,min_slp, markeredgecolor=color, marker=marker,
                    markerfacecolor='None', linewidth=1.5, linestyle=ls,
                    c=color, label=exp, zorder=zorder)
        ax1.xaxis.label.set_color(x_label_color)
        
        if exp != 'Cowan':
            ax2.plot(time, distance, markeredgecolor=color, marker=marker,
                        markerfacecolor='None', linewidth=1.5, linestyle=ls,
                        c=color, label=exp, zorder=zorder)
            ax2.xaxis.label.set_color(x_label_color)
            
    df_dist = pd.DataFrame.from_dict(distances)
    df_min = pd.DataFrame.from_dict(mins)
    
    fname3 = os.path.join(tracks_figures_directory, 'min-slp.png')
    fname4 = os.path.join(tracks_figures_directory, 'distance-timeseries.png')
    for fname, ax, fig in zip([fname3, fname4], [ax1, ax2], [fig1, fig2]):
        legend1, legend2 = make_legend(colors,markers,lines, ax)
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        fig.savefig(fname, dpi=500)
        print(fname+' created!')
    
    return df_dist, df_min

def bar_plot_distances(df, fname):
    
    df_sns = pd.DataFrame(df[df.columns[0]])
    df_sns = df_sns.rename(columns={df.columns[0]:'values'})
    df_sns['exp'] = df.columns[0]
    df_sns['mean'] = df_sns['values'].mean()
    
    for col in df.columns[1:]:
        tmp = pd.DataFrame(df[col])
        tmp = tmp.rename(columns={col:'values'})
        tmp['exp'] = col
        tmp['mean'] = tmp['values'].mean()
        df_sns = pd.concat([df_sns, tmp])
    
    plt.close('all')
    plt.figure(figsize=(10,8))
    ax = sns.barplot(data=df_sns, x='exp', y='values',
                palette='cmo.matter', hue='mean',dodge=False)
    ax.get_legend().remove()
    plt.xticks(rotation=30, ha='right')
    if 'slp' in fname:
        plt.ylim(990,1015)
        ax.axhline(df_sns['values'].min(),color='gray',alpha=0.6)
    else:
        plt.ylim(round(df_sns['mean'].min()-50,-1),
                  round(df_sns['mean'].max()+50,-1))
    
    plt.savefig(fname, dpi=500)
    print(fname,'created!')
    
def normalize_df(df):
    return (df-df.min())/(df.max()-df.min())
        
def main(tracks_directory):
    experiment_directory = ('/').join(os.path.dirname(tracks_directory).split('/')[:-1])
    benchmarks = os.path.basename(os.path.dirname(tracks_directory)).split('tracks_')[1]
    
    tracks = glob.glob(f'{tracks_directory}/*csv')

    figures_directory = os.path.join(experiment_directory, f'Figures_{benchmarks}')
    tracks_figures_directory = os.path.join(figures_directory, 'tracks')
    stats_directory = os.path.join(experiment_directory, f'stats_{benchmarks}')

    for directory in [figures_directory, tracks_figures_directory, stats_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    tracks_one_image(tracks, tracks_directory, tracks_figures_directory)
    if benchmarks in ['48h_sst', '24h'] :
        tracks_subplots(tracks, tracks_directory, tracks_figures_directory)
    
    df_dist, df_min = minimum_slp_and_distance(
        tracks, tracks_directory,tracks_figures_directory, stats_directory)
        
    df_dist = df_dist.sort_index(axis=1, ascending=True)
    df_min = df_min.sort_index(axis=1, ascending=True)
    
    bar_plot_distances(df_dist, os.path.join(tracks_figures_directory, 'barplot-distances.png'))
    bar_plot_distances(df_min.drop(columns='Cowan'),
                        os.path.join(tracks_figures_directory, 'barplot-min-slp.png'))
    
    dist_mean_norm = normalize_df(df_dist.mean()).sort_index(ascending=True)
    slp_mean_norm = normalize_df(df_min.drop(columns='Cowan').mean()).sort_index(ascending=True)
    
    stats = pd.DataFrame(dist_mean_norm, columns=['distance'])
    stats['minimum presure'] = slp_mean_norm
    stats.to_csv(os.path.join(stats_directory, 'distance_min-slp_normalised.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_directory", nargs="?", default="../experiments_48h/tracks_48h/",
                        help="Path to the tracks directory")
    args = parser.parse_args()

    main(args.tracks_directory)