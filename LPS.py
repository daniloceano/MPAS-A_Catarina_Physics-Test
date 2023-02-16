#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:05:29 2022

@author: danilocoutodsouza
"""


import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cmocean
import glob
import numpy as np


def MarkerSizeKe(df):
    
    msizes = [200,400,600,800,1000]
    
    intervals = [3e5,4e5,5e5,6e5]
    data = df['Ke']
    title = 'Eddy Kinect\n    Energy\n(Ke - '+r' $J\,m^{-2})$'

    sizes = []
    for val in data:
        if val <= intervals[0]:
            sizes.append(msizes[0])
        elif val > intervals[0] and val <= intervals[1]:
            sizes.append(msizes[1])
        elif val > intervals[1] and val <= intervals[2]:
            sizes.append(msizes[2])
        elif val > intervals[2] and val <= intervals[3]:
            sizes.append(msizes[3])
        else:
            sizes.append(msizes[4])
    df['sizes'] = sizes
    
    # Plot legend
    labels = ['< '+str(intervals[0]),
              '< '+str(intervals[1]),
              '< '+str(intervals[2]),
              '< '+str(intervals[3]),
              '> '+str(intervals[3])]
    l1 = plt.scatter([],[],c='k', s=msizes[0],label=labels[0])
    l2 = plt.scatter([],[], c='k', s=msizes[1],label=labels[1])
    l3 = plt.scatter([],[],c='k', s=msizes[2],label=labels[2])
    l4 = plt.scatter([],[],c='k', s=msizes[3],label=labels[3])
    l5 = plt.scatter([],[],c='k', s=msizes[4],label=labels[4])
    leg = plt.legend([l1, l2, l3, l4, l5], labels, ncol=1, frameon=False,
                     fontsize=10, handlelength = 0.3, handleheight = 4,
                     borderpad = 1.5, scatteryoffsets = [0.1], framealpha = 1,
                handletextpad=1.5, title=title,
                scatterpoints = 1, loc = 1,
                bbox_to_anchor=(0.73, -0.57, 0.5, 1),labelcolor = '#383838')
    leg._legend_box.align = "center"
    plt.setp(leg.get_title(), color='#383838')
    plt.setp(leg.get_title(),fontsize=12)
    for i in range(len(leg.legendHandles)):
        leg.legendHandles[i].set_color('#383838')
        leg.legendHandles[i].set_edgecolor('gray')
    
    return df

def LorenzPhaseSpace(method):

    
    plt.close('all')
    fig = plt.figure(figsize=(10,10))
    plt.gcf().subplots_adjust(right=0.85)
    ax = plt.gca()
    
    for exp in exps:
        
        print(exp)
        outfile = glob.glob('./LEC_results/'+str(exp)+'*'+'/'+str(exp)+'*')[0]
        
        
        # Open data
        df = pd.read_csv(outfile)
        df['Datetime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
        
        # Get 12H means
        smoothed = df.groupby(pd.Grouper(key="Datetime", freq=method)).mean()
        # Set datetime to the date range
        starts = pd.Series(smoothed.index).dt.strftime('%Y-%m-%d %H:%M')
        ends = pd.Series(pd.DatetimeIndex(starts) + \
                         pd.Timedelta(hours=12)).dt.strftime('%Y-%m-%d %H:%M')
        smoothed['Datetime'] = pd.DataFrame(starts.astype(str)+' - '+\
                                            ends.astype(str)).values
            
        df = smoothed
        
        Ca = df['Ca']
        Ck = df['Ck']
        Ge = df['Ge']
        RAe = df['Ge']+df['BAe']
        Re = df['RKe']+df['BKe']
        df['Rae'], df['Re'] = RAe, Re
        
        # Line plot
        ax.plot(Ck,Ca,'-',c='gray',zorder=2,linewidth=3)
        
        # Scatter plot
        s = MarkerSizeKe(df)['sizes']
        # Plot limits
        ax.set_xlim(-30,30)
        ax.set_ylim(-3,12)
        
        # arrows connecting dots
        ax.quiver(Ck[:-1], Ca[:-1],
                  (Ck[1:].values-Ck[:-1].values),
                  (Ca[1:].values-Ca[:-1].values),
                  angles='xy', scale_units='xy', scale=1)
        
        # plot the moment of maximum intensity
        norm = colors.TwoSlopeNorm(vmin=-7, vcenter=0, vmax=15)
        ax.scatter(Ck.loc[s.idxmax()],Ca.loc[s.idxmax()],
                   c='None',s=s.loc[s.idxmax()]*1.1,
                   zorder=100,edgecolors='k', norm=norm, linewidth=3)
        
        # dots
        dots = ax.scatter(Ck,Ca,c=Ge,cmap=cmocean.cm.curl,s=s,zorder=100,
                        edgecolors='grey', norm=norm)
        
        
        # Marking start and end of the system
        ax.text(Ck[0], Ca[0],'A',
                zorder=101,fontsize=22,horizontalalignment='center',
                verticalalignment='center')
        ax.text(Ck.iloc[-1], Ca.iloc[-1], 'Z',
                zorder=101,fontsize=22,horizontalalignment='center',
                verticalalignment='center')
        
    # Labels
    ax.set_xlabel('Conversion from zonal to eddy Kinetic Energy (Ck - '+r' $W\,m^{-2})$',
                  fontsize=12,labelpad=40,c='#383838')
    ax.set_ylabel('Conversion from zonal to eddy Potential Energy (Ca - '+r' $W\,m^{-2})$',
                  fontsize=12,labelpad=40,c='#383838')
            
    # Gradient lines in the center of the plot
    alpha, offsetalpha = 0.3, 20
    lw, c = 2.5, '#383838'
    offsetx, offsety = 16, 4.3
    for i in range(7):
        ax.axhline(y=0+(i/offsetx),zorder=0+(i/5),linewidth=lw,
                   alpha=alpha-(i/offsetalpha),c=c)
        ax.axhline(y=0-(i/offsetx),zorder=0+(i/5),linewidth=lw,
                   alpha=alpha-(i/offsetalpha),c=c)
        ax.axvline(x=0+(i/offsety),zorder=0+(i/5),linewidth=lw,
               alpha=alpha-(i/offsetalpha),c=c)
        ax.axvline(x=0-(i/offsety),zorder=0+(i/5),linewidth=lw,
               alpha=alpha-(i/offsetalpha),c=c)
        # Vertical line showing when Ca is more important than Ck
        plt.plot(np.arange(0-(i/offsety),-40-(i/offsety),-1),
                 np.arange(0,40), c=c,zorder=1, 
                 alpha=0.2-(i/offsetalpha*.5))
        plt.plot(np.arange(0+(i/offsety),-40+(i/offsety),-1),
                 np.arange(0,40), c=c,zorder=1, linewidth=lw,
                 alpha=0.2-(i/offsetalpha*.5))
   

    # Colorbar
    cax = fig.add_axes([ax.get_position().x1+0.01,
                    ax.get_position().y0+0.32,0.02,ax.get_position().height/1.74])
    cbar = plt.colorbar(dots, extend='both',cax=cax)
    cbar.ax.set_ylabel('Generation of eddy Potential Energy (Ge - '+r' $W\,m^{-2})$',
                   rotation=270,fontsize=12,verticalalignment='bottom',
                   c='#383838',labelpad=40)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(10) 
        

    # Annotate plot
    plt.tick_params(labelsize=10)
    system = 'Catarina Physics Test'
    datasource = outfile.split('/')[-1].split('_')[1]
    start, end = str(df['Datetime'][0]),str(df['Datetime'].iloc[-1]) 
    ax.text(0,1.1,'System: '+system+' - Data from: '+datasource,
            fontsize=16,c='#242424',horizontalalignment='left',
            transform=ax.transAxes)
    ax.text(0,1.06,'Start (A):',fontsize=14,c='#242424',
            horizontalalignment='left',transform=ax.transAxes)
    ax.text(0,1.025,'End (Z):',fontsize=14,c='#242424',
            horizontalalignment='left',transform=ax.transAxes)
    ax.text(0.14,1.06,str(1979),fontsize=14,c='#242424',
            horizontalalignment='left',transform=ax.transAxes)
    ax.text(0.14,1.025,str(2020),fontsize=14,c='#242424',
            horizontalalignment='left',transform=ax.transAxes)
    annotate_fs = 10
    y_upper = 'Eddy is gaining potential energy \n from the mean flow'
    y_lower = 'Eddy is providing potential energy \n to the mean flow'
    x_left = 'Eddy is gaining kinetic energy \n from the mean flow'
    x_right = 'Eddy is providing kinetic energy \n to the mean flow'
    col_lower = 'Subisidence decreases \n eddy potential energy'
    col_upper = 'Latent heat release feeds \n eddy potential energy'
    lower_left = 'Barotropic instability'
    # upper_left = 'Eddy growth by barotropic and\n baroclinic processes'
    upper_left = 'Barotropic and baroclinic instabilities'
    lower_right = 'Eddy is feeding the local atmospheric circulation'
    # upper_right = 'Gain of eddy potential energy\n via baroclinic processes'
    upper_right = 'Baroclinic instability'
        
    ax.text(-0.08,0.12,y_lower,
            rotation=90,fontsize=annotate_fs,horizontalalignment='center',c='#19616C',
            transform=ax.transAxes)
    ax.text(-0.08,0.65,y_upper,
            rotation=90,fontsize=annotate_fs,horizontalalignment='center',c='#CF6D66',
            transform=ax.transAxes)
    ax.text(0.22,-0.08,x_left,
            fontsize=annotate_fs,horizontalalignment='center',c='#CF6D66',
            transform=ax.transAxes)
    ax.text(0.75,-0.08,x_right,
            fontsize=annotate_fs,horizontalalignment='center',c='#19616C',
            transform=ax.transAxes)
    ax.text(1.13,0.51,col_lower,
            rotation=270,fontsize=annotate_fs,horizontalalignment='center',c='#19616C'
            ,transform=ax.transAxes)
    ax.text(1.13,0.75,col_upper,
            rotation=270,fontsize=annotate_fs,horizontalalignment='center',c='#CF6D66',
            transform=ax.transAxes)
    ax.text(0.22,0.03,lower_left,
            fontsize=annotate_fs,horizontalalignment='center',c='#660066',
            verticalalignment='center', transform=ax.transAxes)
    ax.text(0.22,0.97,upper_left,
            fontsize=annotate_fs,horizontalalignment='center',c='#800000',
            verticalalignment='center', transform=ax.transAxes)
    ax.text(0.75,0.03,lower_right,
            fontsize=annotate_fs,horizontalalignment='center',c='#000066',
            verticalalignment='center', transform=ax.transAxes)
    ax.text(0.75,0.97,upper_right,
            fontsize=annotate_fs,horizontalalignment='center',c='#660066',
            verticalalignment='center', transform=ax.transAxes)
        
    fname = './Figures/LPS/LPS_'+method+'.png'
    plt.savefig(fname,dpi=500)
    print(fname+' created!')


if __name__ == "__main__":
    
    # lists = glob.glob('../raw_data/CycloneList*')
    results = glob.glob('LEC_results/*')
    exps = []
    for exp in results:
        exps.append(exp.split('/')[-1].split('_')[0])
    
    for method in ['24H']:
        LorenzPhaseSpace(method)