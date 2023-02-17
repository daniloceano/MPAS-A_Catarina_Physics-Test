#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 22:02:13 2023

@author: danilocoutodsouza
"""

import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cmocean
import glob
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import skill_metrics as sm

msizes = [20,40,60,80,100]
intervals = [3e5,4e5,5e5,6e5]
norm = colors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=8)
c='#383838'

def MarkerSizeKe(df):
    
    data = df['Ke']

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
    return df

def LegendKe():
    
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
                handletextpad=1.5, title='Ke',
                scatterpoints = 1, loc = 4,
                bbox_to_anchor=(1.22, -0.3, 0.5, 1),labelcolor = '#383838')
    leg._legend_box.align = "center"
    plt.setp(leg.get_title(), color='#383838')
    plt.setp(leg.get_title(),fontsize=12)
    for i in range(len(leg.legendHandles)):
        leg.legendHandles[i].set_color('#383838')
        leg.legendHandles[i].set_edgecolor('gray')


def LorenzPhaseSpace(ax, df, row, col):
        
    Ca = df['Ca']
    Ck = df['Ck']
    Ge = df['Ge']
    RAe = df['Ge']+df['BAe']
    Re = df['RKe']+df['BKe']
    df['Rae'], df['Re'] = RAe, Re
    
    # Line plot
    ax.plot(Ck,Ca,'-',c='gray',zorder=2,linewidth=3)
    
    s = MarkerSizeKe(df)['sizes']
    ax.set_xlim(-15,2.5)
    ax.set_ylim(-4,1)
    
    # lines in the center of the plot
    ax.axhline(y=0,linewidth=3, alpha=0.6,c='#383838')
    ax.axvline(x=0,linewidth=3,alpha=0.6,c='#383838')
       
    
    # arrows connecting dots
    ax.quiver(Ck[:-1], Ca[:-1],
              (Ck[1:].values-Ck[:-1].values),
              (Ca[1:].values-Ca[:-1].values),
              angles='xy', scale_units='xy', scale=1)
    
    # plot the moment of maximum intensity
    ax.scatter(Ck.loc[s.idxmax()],Ca.loc[s.idxmax()],
               c='None',s=s.loc[s.idxmax()]*1.1,
               zorder=100,edgecolors='k', norm=norm, linewidth=3)
    
    dots = ax.scatter(Ck,Ca,c=Ge,cmap=cmocean.cm.curl,s=s,zorder=100,
                    edgecolors='grey', norm=norm)
        
    ax.text(Ck[0], Ca[0],'A',
            zorder=101,fontsize=18,horizontalalignment='center',
            verticalalignment='center')
    ax.text(Ck.iloc[-1], Ca.iloc[-1], 'Z',
            zorder=101,fontsize=18,horizontalalignment='center',
            verticalalignment='center')
    
    if row == 5:
        ax.set_xlabel('Ck', fontsize=12,c='#383838')
    if col == 0:
        ax.set_ylabel('Ca', fontsize=12,c='#383838')
        
    return dots

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
            
era_file = glob.glob('LEC_results_48h/*MPAS*/*MPAS*.csv')[0]
era_data = pd.read_csv(era_file)
era_data['Datetime'] = pd.to_datetime(era_data.Date
                                    ) + pd.to_timedelta(era_data.Hour,unit='h')


plt.close('all')
fig = plt.figure(figsize=(10, 13))
gs = gridspec.GridSpec(6, 3)
results = glob.glob('LEC_results_48h/*track')
i = 0
for row in range(6):
    for col in range(3):
    
        exp = results[i]
        dirname = exp.split('/')[-1]
        outfile = glob.glob(
            './LEC_results_48h/'+str(dirname)+'*'+'/'+str(dirname)+'*csv')[0]
                
        # Open data
        df = pd.read_csv(outfile)
        df['Datetime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour,
                                                                   unit='h')
    
        ax = fig.add_subplot(gs[row, col], frameon=True)
        dots = LorenzPhaseSpace(ax, df, row, col)
        
        expname = exp.split('/')[1].split('_MPAS_track')[0]
        ax.text(-15,1.5,expname,c=c)
        
        i+=1
        
LegendKe()

# Colorbar
cax = fig.add_axes([ax.get_position().x1+0.02,
                ax.get_position().y0+0.3,0.02,ax.get_position().height*3])
cbar = plt.colorbar(dots, extend='both',cax=cax)
cbar.ax.set_ylabel('Ge',fontsize=12,verticalalignment='bottom',c='#383838',
                   labelpad=10)
for t in cbar.ax.get_yticklabels():
      t.set_fontsize(10) 
      
plt.subplots_adjust(right=0.85,hspace=0.4, bottom=0.05, top=0.95, left=0.06)     
      
plt.savefig('Figures_48h/validate_LPS.png',dpi=500)
