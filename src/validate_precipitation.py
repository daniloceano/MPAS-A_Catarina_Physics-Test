# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    validate_precipitation.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/08 09:52:10 by Danilo            #+#    #+#              #
#    Updated: 2023/07/10 16:47:44 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import glob
import f90nml
import datetime
import argparse

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

prec_levels = [0.1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
cmap_precipitation = colors.ListedColormap(['#D3E6F1','#2980B9', '#A9DFBF','#196F3D',
    '#F9E79F', '#F39C12', '#f37012', '#E74C3C', '#943126', '#E6B0AA', '#7a548e'], N=len(prec_levels)-1)

def get_times_nml(namelist, model_data):
    """
    Calculates the times of the model data.

    Parameters:
        namelist (dict): The namelist containing the configuration start time and run duration.
        model_data (pd.DataFrame): The model data.

    Returns:
        pd.DatetimeIndex: The times of the model data.
    """
    start_date_str = namelist['nhyd_model']['config_start_time']
    run_duration_str = namelist['nhyd_model']['config_run_duration']
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d_%H:%M:%S')
    run_duration = datetime.datetime.strptime(run_duration_str, '%d_%H:%M:%S')
    finish_date = start_date + datetime.timedelta(days=run_duration.day, hours=run_duration.hour)
    times = pd.date_range(start_date, finish_date, periods=len(model_data.Time) + 1)[1:]
    return times

def get_experiment_parameters(experiments):
    """
    A function that takes a list of experiments and returns the times, first day, and last day.

    Parameters:
    - experiments: a list of experiments

    Returns:
    - times: a list of times
    - first_day: a string representing the first day
    - last_day: a string representing the last day
    """

    # Dummy for getting model times
    dummy = experiments[0]
    model_output = f'{dummy}/latlon.nc'
    namelist_path = f'{dummy}/namelist.atmosphere'

    # open data and namelist
    model_data = xr.open_dataset(model_output).chunk({"Time": -1})
    namelist = f90nml.read(glob.glob(namelist_path)[0])
    times = get_times_nml(namelist,model_data)
    first_day = datetime.datetime.strftime(times[0], '%Y-%m-%d')
    last_day = datetime.datetime.strftime(times[-2], '%Y-%m-%d')

    parameters = {
        'times': times,
        'first_day': first_day,
        'last_day': last_day,
        'max_latitude': float(model_data.latitude[0]),
        'max_longitude': float(model_data.longitude[-1]),
        'min_latitude': float(model_data.latitude[-1]),
        'min_longitude': float(model_data.longitude[0]),
    }

    return parameters 

def get_exp_name(experiment):
    """
    Returns the name of the experiment based on the given experiment path.

    Parameters:
        experiment (str): The path of the experiment.

    Returns:
        str: The name of the experiment.
    """
    expname = os.path.basename(experiment)

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

def get_model_accprec(model_data):
    """
    Returns the accumulated precipitation from the given model data.
    
    Parameters:
        model_data (dict): A dictionary containing the model data.
        
    Returns:
        float: The accumulated precipitation.
    """
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

def process_experiment_data(data, experiment, experiment_name, imerg_accprec, times): 
    """
    Processes experiment data and adds it to the given data dictionary.

    Parameters:
    - data: The dictionary to which the processed data will be added.
    - experiment: The path to the experiment.
    - experiment_name: The name of the experiment.
    - imerg_accprec: The IMERG accumulated precipitation data.
    - times: The times associated with the model data.

    Returns:
    - The updated data dictionary.
    """ 
    model_data = xr.open_dataset(f'{experiment}/latlon.nc').chunk({"Time": -1})
    model_data = model_data.assign_coords({"Time":times})

    acc_prec = get_model_accprec(model_data)
    acc_prec = acc_prec.where(acc_prec >= 0, 0)
    acc_prec_interp = acc_prec.interp(
        latitude=imerg_accprec.lat,
        longitude=imerg_accprec.lon,
        method='cubic',
        assume_sorted=False
    )
    interp =  acc_prec_interp.where(acc_prec_interp >=0, 0).transpose('lat', 'lon')
    
    print('limits for prec data:',float(acc_prec.min()),float(acc_prec.max()))
    print('limits for interp prec data:',float(acc_prec_interp.min()),
          float(acc_prec_interp.max()))
    
    stats = sm.taylor_statistics(imerg_accprec.values.ravel(),
                                 interp.values.ravel())
    
    ccoef = stats['ccoef'][1]
    if np.isnan(ccoef):
        ccoef = 0
    
    data[experiment_name] = {
        'data': acc_prec,
        'interp': interp,
        'ccoef': ccoef,
        'crmsd': stats['crmsd'][1],
        'sdev': stats['sdev'][1],
        'willmot_d_index': willmot_d_index(imerg_accprec.values.ravel(), interp.values.ravel())
    }

    return data

def willmot_d_index(observed, modeled):
    """
    Calculate the Willmot's D index between observed and modeled data.

    Parameters:
        observed (numpy.ndarray): An array of observed data.
        modeled (numpy.ndarray): An array of modeled data.

    Returns:
        float: The Willmot's D index between the observed and modeled data.
    """
    obs_mean = observed.mean()
    mod_mean = modeled.mean()
    
    numerator = np.sum(np.abs(observed - modeled))
    denominator = np.sum(np.abs(observed - obs_mean)) + np.sum(np.abs(modeled - mod_mean))
        
    return  1 - (numerator / denominator)

def call_taylor_diagram(sdevs,crmsds,ccoefs,experiments):
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
    
def define_figure_parameters(benchmarks_name):
    """
    Define figure parameters based on the given benchmark name.
    
    Parameters:
        benchmarks_name (str): The name of the benchmark.
        
    Returns:
        tuple: A tuple containing the values of ncol, nrow, imax, and figsize.
    """
    if benchmarks_name == '48h':
        ncol, nrow, imax = 3, 5, 14
        figsize = (10, 10)
    elif (benchmarks_name == '48h_sst') or (benchmarks_name == '96h_sst'):
        ncol, nrow, imax = 2, 2, 3
        figsize = (10, 8)
    elif benchmarks_name == '48h_pbl':
        ncol, nrow, imax = 3, 4, 11
        figsize = (8, 8)
    elif benchmarks_name == '2403-2903':
        ncol, nrow, imax = 1, 1, 1
        figsize = (5, 5)
    else:
        ncol, nrow, imax = 3, 6, 18
        figsize = (10, 12)
    return ncol, nrow, imax, figsize
    
def configure_gridlines(ax, col, row):
    """
    Configure gridlines for the map.

    Parameters:
        ax (AxesSubplot): The axes on which to configure the gridlines.
        col (int): The column index of the map.
        row (int): The row index of the map.

    Returns:
        None
    """
    # Configure gridlines for the map
    gl = ax.gridlines(
        draw_labels=True,
        zorder=2,
        linestyle='dashed',
        alpha=0.8,
        color='#383838'
    )
    gl.xlabel_style = {'size': 12, 'color': '#383838'}
    gl.ylabel_style = {'size': 12, 'color': '#383838'}
    gl.right_labels = None
    gl.top_labels = None
    gl.bottom_labels = None if row != 5 else gl.bottom_labels
    gl.left_labels = None if col != 0 else gl.left_labels

def plot_precipitation_panels(
        data, imerg_accprec, experiments, benchmarks_name,
          figures_directory, bias_levels, bias_norm, bias_flag=False):
    """
    Plot precipitation panels for the given benchmarks.

    Parameters:
    - benchmarks_name (str): The name of the benchmarks.
    - bias (bool): Whether to plot bias or not. Default is False.

    Returns:
    None
    """
    print('\nPlotting maps...')
    plt.close('all')

    ncol, nrow, imax, figsize = define_figure_parameters(benchmarks_name)
    print('Figure will have ncols:', ncol, 'rows:', nrow, 'n:', imax)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol)
    datacrs = ccrs.PlateCarree()

    i = 0
    for col in range(ncol):
        for row in range(nrow):
            
            if i == imax:
                break

            experiment = experiments[i]
            experiment = get_exp_name(experiment)
            print('\n',experiment)

            if 'off_' in experiment: continue
            
            prec = data[experiment]['data']
            prec_interp = data[experiment]['interp']
                            
            ax = fig.add_subplot(gs[row, col], projection=datacrs,frameon=True)
            ax.set_extent([-55, -30, -20, -35], crs=datacrs) 
            ax.text(-50,-19,experiment)
            configure_gridlines(ax, col, row)
            
            if bias_flag == False:
                print('Plotting accumulate prec..')
                cf = ax.contourf(prec.longitude, prec.latitude, prec,
                                    cmap=cmap_precipitation, levels=prec_levels)
                print('prec limits:',float(prec.min()), float(prec.max()))
            else:
                print('Plotting bias..')
                bias = prec_interp-imerg_accprec
                cf = ax.contourf(imerg_accprec.lon, imerg_accprec.lat,bias,
                                    cmap=cmo.balance_r,
                                    levels=bias_levels, norm=bias_norm)
                print('bias limits:',float(bias.min()), float(bias.max()))
            ax.coastlines(zorder = 1)
            i+=1

    cb_axes = fig.add_axes([0.85, 0.18, 0.04, 0.6])
    fig.colorbar(cf, cax=cb_axes, orientation="vertical") 
    fig.subplots_adjust(wspace=0.1,hspace=0, right=0.8)

    if bias_flag == False:
        fname = f"{figures_directory}/{benchmarks_name}_acc_prec.png"
    else:
        fname = f"{figures_directory}/{benchmarks_name}_acc_prec_bias.png"
    fig.savefig(fname, dpi=500)
    print(fname,'saved')

def plot_imerg_precipitation(imerg_accprec, imerg_file, figures_directory):
    """
    Plots IMERG accumulated precipitation.

    Parameters:
    - imerg_accprec: The IMERG accumulated precipitation data.

    Returns:
    None
    """
    print('\nPlotting IMERG data..')
    plt.close('all')

    fig = plt.figure(figsize=(10, 10))
    datacrs = ccrs.PlateCarree()

    ax = fig.add_subplot(111, projection=datacrs,frameon=True)
    ax.set_extent([-55, -30, -20, -35], crs=datacrs) 
    configure_gridlines(ax, 5, 0)

    cf = ax.contourf(imerg_accprec.lon, imerg_accprec.lat,
                    imerg_accprec, cmap=cmap_precipitation,
                    levels=prec_levels)
    
    fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.1)
    ax.coastlines(zorder = 1)

    imergname = os.path.basename(imerg_file)
    fname_imerg = f"{figures_directory}/{imergname}.png"
    fig.savefig(fname_imerg, dpi=500)
    print(fname_imerg,'saved')

def plot_pdfs(data, imerg_accprec, benchmarks_name, experiments, figures_directory):
    """
    Plot PDFs for the accumulated precipitation data.

    Parameters:
    - imerg_accprec: A Pandas DataFrame containing the IMERG accumulated precipitation data.
    - benchmarks_name: A string representing the name of the benchmarks.
    - experiments: A list of strings representing the names of the experiments.

    Returns:
    None.
    """
    plt.close('all')

    print('\nPlotting PDFs..')

    nbins = 100
    params_imerg = st.weibull_min.fit(imerg_accprec.values.ravel())
    x_imerg = np.linspace(st.weibull_min.ppf(0.01, *params_imerg),
                    st.weibull_min.ppf(0.99, *params_imerg), nbins)

    ncol, nrow, imax, figsize = define_figure_parameters(benchmarks_name)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol)

    i = 0
    for col in range(ncol):
        for row in range(nrow):
            if i == imax:
                break
        
            ax = fig.add_subplot(gs[row, col], frameon=True)
        
            experiment = experiments[i]
            experiment = get_exp_name(experiment)
            print('\n',experiment)

            if 'off_' in experiment: continue
            
            reference = imerg_accprec.values.ravel()
            predicted =  data[experiment]['interp'].values.ravel()
            
            if experiment != 'off_off':
                ax.hist(reference, bins=nbins, color='k', lw=1, alpha=0.3,
                        density=True, histtype='step',label='IMERG', zorder=1) 
                
                ax.hist(predicted, bins=nbins, color='tab:red',  lw=1, alpha=0.3,
                        density=True, histtype='step', label=experiment, zorder=100)
                ax.set_yscale('log')  
                ax.text
                ax.text(200, 0.1, experiment)
                i+=1
                
    fig.subplots_adjust(hspace=0.25)
    fname_pdf = f"{figures_directory}/{benchmarks_name}_PDF.png"
    fig.savefig(fname_pdf, dpi=500)    
    print(fname_pdf,'saved')

def plot_taylor_diagrams(benchmarks_name, data, figures_directory):
    """
    Plots Taylor diagrams for the total accumulated precipitation.

    Parameters:
        benchmarks_name (str): The name of the benchmarks.
        data (dict): A dictionary containing the data for different experiments.
        figures_directory (str): The directory to save the generated figures.

    Returns:
        crmsd (list): A list of the RMSE values.
        ccoef (list): A list of the R values.
        d_index (list): A list of the D-index values.
    """
    ccoef = [data[exp]['ccoef'] for exp in data.keys() if exp != 'IMERG']
    crmsd  = [data[exp]['crmsd'] for exp in data.keys() if exp != 'IMERG']
    sdev = [data[exp]['sdev'] for exp in data.keys() if exp != 'IMERG']
    d_index = [data[exp]['willmot_d_index'] for exp in data.keys() if exp != 'IMERG']
    keys = [ exp for exp in data.keys() if exp != 'IMERG']

    ccoef, crmsd, sdev = np.array(ccoef),np.array(crmsd),np.array(sdev)

    print('plotting taylor diagrams..')
    fig = plt.figure(figsize=(10,10))

    call_taylor_diagram(sdev,crmsd,ccoef,keys)

    plt.tight_layout(w_pad=0.1)

    fname = f"{figures_directory}/{benchmarks_name}_taylor.png"
    fig.savefig(fname, dpi=500)    
    print(fname, 'created!')

    return crmsd, ccoef, d_index

def precipitation_statistics_to_csv(data, crmsd, ccoef, d_index, stats_directory):
    """
    Generates a CSV file containing precipitation statistics.

    Args:
        crmsd (numpy.ndarray): An array of root mean square errors (RMSE) for each experiment.
        ccoef (numpy.ndarray): An array of correlation coefficients (CCoef) for each experiment.
        d_index (numpy.ndarray): An array of D-indices for each experiment.
        stats_directory (str): The directory where the statistics file will be saved.

    Returns:
        tuple: A tuple containing two pandas DataFrames. The first DataFrame contains the original statistics, 
        including RMSE, CCoef, and D-index. The second DataFrame contains the normalized statistics, where 
        each value is scaled to a range of 0 to 1.

    """
    df_stats = pd.DataFrame(
        crmsd,
        index=[exp for exp in data.keys() if exp != 'IMERG'],
        columns=['rmse']
    )

    df_stats['ccoef'] = ccoef
    df_stats['d_index'] = d_index

    df_stats_normalised = (df_stats-df_stats.min())/(df_stats.max()-df_stats.min()) 
    df_stats_normalised.sort_index(ascending=True
                                   ).to_csv(f'{stats_directory}/precip_RMSE_normalised.csv')
    return df_stats, df_stats_normalised  

def plot_precipitation_statistics(df_stats, df_stats_normalised, benchmarks_name, figures_directory):
    for data, title in zip([df_stats, df_stats_normalised],['stats', 'stats normalised']):
        for col in data.columns:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.bar(data.index, data[col].values)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

            stats_prec_directory = f"{figures_directory}/stats_prec"
            os.makedirs(stats_prec_directory, exist_ok=True)
            fname_stats = f"{stats_prec_directory}/{benchmarks_name}_{title}_{col}.png"
            fig.savefig(fname_stats, dpi=500)
            print(fname_stats, 'saved')  

def main(benchmarks_directory, benchmarks_name, experiment_directory, imerg_file):

    ## Inputs ##
    experiments = glob.glob(benchmarks_directory+'/run*')
    experiments = sorted(experiments)
    
    stats_directory = os.path.join(experiment_directory, f'stats_{benchmarks_name}')
    figures_directory = os.path.join(experiment_directory, f'Figures_{benchmarks_name}')

    ## Start the code ##
    parameters = get_experiment_parameters(experiments)

    # Open IMERG data
    imerg = xr.open_dataset(imerg_file).sel(
        lat=slice(parameters["min_latitude"], parameters["max_latitude"]),
        lon=slice(parameters["min_longitude"], parameters["max_longitude"]),
        time=slice(parameters["first_day"],parameters["last_day"]))
    imerg_accprec = imerg.precipitationCal.cumsum(dim='time')[-1].transpose('lat', 'lon')
    print(imerg)                                                                                      
    print('Using IMERG data from',parameters["first_day"],'to',parameters["last_day"])                                   
    print('Maximum acc prec:',float(imerg_accprec.max()))


    print('\nOpening all data and putting it into a dictionary...')

    data = {'IMERG': imerg_accprec}

    max_precipitation = float('-inf')
    max_bias = float('-inf')
    min_bias = float('inf')

    for experiment in experiments:
        experiment_name = get_exp_name(experiment)
        print('\n', experiment_name)

        if 'off_' in experiment_name: continue
        
        data = process_experiment_data(data, experiment, experiment_name, imerg_accprec, parameters["times"])

        acc_prec = data[experiment_name]['data']
        interp = data[experiment_name]['interp']

        experiment_max = np.max(acc_prec).compute().item()
        experiment_bias = interp - imerg_accprec
        experiment_maximum_bias = np.max(experiment_bias).compute().item()
        experiment_minimum_bias = np.min(experiment_bias).compute().item()

        max_precipitation = max(max_precipitation, experiment_max)
        max_bias = max(max_bias, experiment_maximum_bias)
        min_bias = min(min_bias, experiment_minimum_bias)

    bias_levels = np.arange(min_bias*0.6,max_bias*0.6,20)
    bias_norm = colors.TwoSlopeNorm(vmin=min_bias*0.6, vcenter=0, vmax=max_bias*0.6)

    ## Make plots
    plot_precipitation_panels(data, imerg_accprec, experiments, benchmarks_name,
                                figures_directory, bias_levels, bias_norm)
    plot_precipitation_panels(data, imerg_accprec, experiments, benchmarks_name,
                                figures_directory, bias_levels, bias_norm, bias_flag=True)
    plot_imerg_precipitation(imerg_accprec, imerg_file, figures_directory)
    plot_pdfs(data, imerg_accprec, benchmarks_name, experiments, figures_directory)
    crmsd, ccoef, d_index = plot_taylor_diagrams(benchmarks_name, data, figures_directory)
    df_stats, df_stats_normalised = precipitation_statistics_to_csv(
                                                            data, crmsd, ccoef, d_index, stats_directory)
    plot_precipitation_statistics(df_stats, df_stats_normalised, benchmarks_name, figures_directory)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("benchmarks_directory", nargs="?",
                        default='/p1-nemo/danilocs/mpas/MPAS-BR/benchmarks/Catarina_physics-test/Catarina_250-8km.microp_scheme.convection_scheme',
                        help="Path to MPAS benchmarks")
    parser.add_argument("benchmarks_name", nargs="?",default="48h",
                        help="Name to use for this set of experiments")
    parser.add_argument("experiment_directory", nargs="?",default='../experiments_48h',
                        help="Path to expeirment directory for saving results")
    parser.add_argument("imerg_file", nargs="?",
                        default='/p1-nemo/danilocs/mpas/MPAS-BR/met_data/IMERG/IMERG_20040321-20040323.nc',
                        help="Path to IMERG file")
    args = parser.parse_args()

    main(args.benchmarks_directory, args.benchmarks_name, args.experiment_directory, args.imerg_file)
