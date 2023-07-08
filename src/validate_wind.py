#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 20:04:44 2023

@author: daniloceano
"""

import glob
import os
import f90nml
import datetime
import argparse

import numpy as np
import pandas as pd
import xarray as xr

import skill_metrics as sm

import matplotlib.pyplot as plt
from matplotlib import rcParams
from metpy.calc import wind_speed

class BenchData:
    def __init__(self, path, times):
        self.bench = path
        self.data = xr.open_dataset(path+'/latlon.nc').chunk({"Time": -1})
        self.data = self.data.assign_coords({"Time":times})
        

    def get_exp_name(self):
        expname = os.path.basename(self.bench)

        microp_options = ["thompson", "kessler", "wsm6", "off"]
        microp = next((option for option in microp_options if option in expname), None)
        if microp is None:
            raise ValueError("Microp option not found in experiment name.")

        cumulus_options = ["ntiedtke", "tiedtke", "freitas", "fritsch", "off"]
        cumulus = next((option for option in cumulus_options if option in expname), None)
        if cumulus is None:
            raise ValueError("Cumulus option not found in experiment name.")

        pbl_options = ["ysu", "mynn"]
        pbl = next((option for option in pbl_options if option in expname), None)

        return f"{microp}_{cumulus}_{pbl}" if pbl else f"{microp}_{cumulus}"
        
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
    rcParams.update({'font.size': 14}) # font size of axes text
    sm.taylor_diagram(sdevs,crmsds,ccoefs,
                      markerLabelColor = 'b', 
                      markerLabel = experiments,
                      markerColor = 'r', markerLegend = 'on', markerSize = 15, 
                      titleRMS = 'off', widthRMS = 2.0,
                      colRMS = '#728B92', styleRMS = '--',  
                      widthSTD = 2, styleSTD = '--', colSTD = '#8A8A8A',
                      titleSTD = 'on',
                      colCOR = 'k', styleCOR = '-',
                      widthCOR = 1.0, titleCOR = 'off',
                      colObs = 'k', markerObs = '^',
                      titleOBS = 'QUICKSCAT', styleObs =':',
                      alpha = 1)

def main(benchmarks_directory, benchmarks_name, experiment_directory, quickscat_file):
    stats_directory = os.path.join(experiment_directory, f'stats_{benchmarks_name}')
    figures_directory = os.path.join(experiment_directory, f'Figures_{benchmarks_name}')

    ## Start the code ##
    benchs = glob.glob(f"{benchmarks_directory}/run*")

    # Dummy for getting model times
    model_output = benchs[0]+'/latlon.nc'
    namelist_path = benchs[0]+"/namelist.atmosphere"

    # Open data and namelist
    model_data = xr.open_dataset(model_output)
    namelist = f90nml.read(glob.glob(namelist_path)[0])
    times = get_times_nml(namelist,model_data)

    first_day = datetime.datetime.strftime(times[0], '%Y-%m-%d')
    last_day = datetime.datetime.strftime(times[-2], '%Y-%m-%d')

    # Reference: quickscat
    da_reference = convert_lon(xr.open_dataset(quickscat_file),'lon').sel(
                        lat=slice(model_data.latitude[-1],model_data.latitude[0]),
                        lon=slice(model_data.longitude[0],model_data.longitude[-1])
                        ).sel(time=slice(first_day,last_day))
    u = da_reference.uwnd
    v = da_reference.vwnd
    reference = 'quickscat'

    # Reference: ERA5
    # da_reference = convert_lon(xr.open_dataset(args.ERA5),'longitude').sel(
    #                     latitude=slice(
    #                         model_data.latitude[0],model_data.latitude[-1]),
    #                     longitude=slice(
    #                         model_data.longitude[0],model_data.longitude[-1])
    #                     ).sel(time=slice(first_day,last_day))
    # u = da_reference.u10
    # v = da_reference.v10
    # reference = 'ERA5'                          

    print('\nOpening all data and putting it into a dictionary...')
    benchmarks = [BenchData(bench, times) for bench in benchs]
    data = {}

    for benchmark in benchmarks:
        exp_name = benchmark.get_exp_name()
        print('\n',exp_name)

        if 'off_' in exp_name: continue

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
        
    # =============================================================================
    # Plot Taylor Diagrams and do Statistics ##
    # =============================================================================

    for var in ['u10','v10','windspeed']:
        
        print('\n-------------------------------')
        print(var)
        

        fname = f'{benchmarks_name}_{var}'
        
        ccoef = [data[exp][var]['ccoef'] for exp in data.keys() if exp != 'IMERG']
        crmsd  = [data[exp][var]['crmsd'] for exp in data.keys() if exp != 'IMERG']
        sdev = [data[exp][var]['sdev'] for exp in data.keys() if exp != 'IMERG']
        ccoef, crmsd, sdev = np.array(ccoef),np.array(crmsd),np.array(sdev)

        print('plotting taylor diagrams..')
        fig = plt.figure(figsize=(10,10))
        plot_taylor(sdev,crmsd,ccoef,list(data.keys()))
        plt.tight_layout(w_pad=0.1)

        figures_stats_directory = os.path.join(figures_directory, 'stats_wind')
        os.makedirs(figures_stats_directory, exist_ok=True)
        outname = f'{figures_stats_directory}/{fname}-taylor.png'
        fig.savefig(outname, dpi=500)    
        print(f'Saved {outname}')
        
        df_stats = pd.DataFrame(crmsd,
                            index=[exp for exp in data.keys() if exp != 'IMERG'],
                            columns=['rmse'])
        df_stats['ccoef'] = ccoef
        
        # Normalize values for comparison
        df_stats_norm = (df_stats-df_stats.min())/(df_stats.max()-df_stats.min())
        fname_stats = f'{stats_directory}/{var}_RMSE_normalised.csv'
        df_stats_norm.sort_index(ascending=True).to_csv(fname_stats)
        print(f'Saved {fname_stats}') 
        
        for df, title in zip([df_stats, df_stats_norm],
                            ['stats', 'stats_normalised']):
            for col in df.columns:
                plt.close('all')
                f, ax = plt.subplots(figsize=(10, 10))
                ax.bar(df.index,df[col].values)
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                outname = f'{figures_stats_directory}/{var}_{title}_{col}.png'
                f.savefig(outname, dpi=500)
                print(f'Saved {outname}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("benchmarks_directory", nargs="?",
                        default='/p1-nemo/danilocs/mpas/MPAS-BR/benchmarks/Catarina_physics-test/Catarina_250-8km.microp_scheme.convection_scheme',
                        help="Path to MPAS benchmarks")
    parser.add_argument("benchmarks_name", nargs="?",default="48h",
                        help="Name to use for this set of experiments")
    parser.add_argument("experiment_directory", nargs="?",default='../experiments_48h',
                        help="Path to expeirment directory for saving results")
    parser.add_argument("quickscat_file", nargs="?",
                        default='/p1-nemo/danilocs/mpas/MPAS-BR/met_data/QUICKSCAT/Catarina_20040321-20040323_v11l30flk.nc',
                        help="Path to QUICKSCAT file")
    args = parser.parse_args()

    main(args.benchmarks_directory, args.benchmarks_name, args.experiment_directory, args.quickscat_file)