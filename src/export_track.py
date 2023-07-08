#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:43:19 2023

@author: daniloceano
"""
import os
import dask
import glob
import f90nml
import argparse
import datetime

import pandas as pd
import numpy as np
import xarray as xr
import dask.array as da

from metpy.units import units
from wrf import interplevel
from metpy.constants import g, Rd
from metpy.calc import virtual_temperature
from geopy.distance import geodesic

lat_domain = slice(-20,-35)
lon_domain = slice(-55,-30)

# Helper functions

def get_experiment_name(bench, pbl=None):
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

def get_simulation_times(namelist,model_data):
    """
    Get the simulation times based on the given namelist and model data.

    Parameters:
    namelist (dict): A dictionary containing the namelist data.
    model_data (pd.DataFrame): A DataFrame containing the model data.

    Returns:
    pd.DatetimeIndex: A Pandas DatetimeIndex object representing the simulation times.
    """
    start_date_str = namelist['nhyd_model']['config_start_time']
    run_duration_str = namelist['nhyd_model']['config_run_duration']
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d_%H:%M:%S')
    run_duration = datetime.datetime.strptime(run_duration_str, '%d_%H:%M:%S')
    finish_date = start_date + datetime.timedelta(days=run_duration.day, hours=run_duration.hour)
    times = pd.date_range(start_date, finish_date, periods=len(model_data.Time) + 1)[1:]
    return times

def open_model_dataset(benchmark_path, times=None):
    """
    Open and retrieve a model dataset.

    Args:
        bench (str): The directory of the dataset.
        times (Optional[List[str]]): A list of time values to select.

    Returns:
        xr.Dataset: The model dataset.
    """
    data_file = os.path.join(benchmark_path, 'latlon.nc')

    with dask.config.set(array={'slicing': {'split_large_chunks': True}}):
        data = xr.open_dataset(data_file)
        
    data = data.sortby('latitude', ascending=False).sel(latitude=lat_domain, longitude=lon_domain)

    if times:
        data = data.assign_coords({"Time":times})
    
    return data

def convert_pressure_to_slp(pressure, z, zlevs):
    """
    Convert the given pressure to sea-level pressure.

    Parameters:
        pressure (DataArray): The pressure values.
        z (DataArray): The heights.
        zlevs (array-like): The levels.

    Returns:
        DataArray: The sea-level pressure.
    """    
    pres_height = interplevel(pressure, z[:,:-1], zlevs)
    slp = pres_height.isel(level=1)
    return slp

def surface_pressure_to_mslp(surface_pressure,
                             mean_virtual_temperature, surface_height):
    """
    Calculates the mean sea level pressure (MSLP) from surface pressure, mean 
    virtual temperature, and surface height.

    Parameters:
        surface_pressure (float): The surface pressure in Pa.
        mean_virtual_temperature (float): The mean virtual temperature in K.
        surface_height (float): The surface height in m.

    Returns:
        float: The mean sea level pressure (MSLP) in Pa.
    """    
    a = Rd / g
    _ = surface_height / (a * mean_virtual_temperature)
    _ = np.exp(_.metpy.dequantify())
    mslp = (surface_pressure) * _
    return mslp

def get_track(slp, TimeIndexer):
    """
    Generate a track dataframe based on the given sea level pressure data and time indexer.

    Args:
        slp (xarray.Dataset): The sea level pressure data.
        TimeIndexer (str): The name of the time indexer in the sea level pressure data.

    Returns:
        pandas.DataFrame: A dataframe containing longitude, latitude, and minimum value for each time step.
    """
    min_var, times = [], []
    lats, lons = [], []
    slp = slp.sortby('latitude', ascending=True).sortby('longitude', ascending=True)
    for t in slp[TimeIndexer]:
        datestr = pd.to_datetime(t.values)
        times.append(str(datestr))
        
        if t == slp[TimeIndexer][0]:
            ivar = slp.sel({TimeIndexer:t})
        else:
            ivar = slp.sel({TimeIndexer: t}).sel(
                latitude=slice(lats[-1] - 7.55, lats[-1] + 7.5),
                longitude=slice(lons[-1] - 7.5, lons[-1] + 7.5))
        
        varmin = ivar.min()
        min_var.append(float(varmin))
        
        loc = ivar.argmin(dim=['latitude','longitude'])
        lats.append(float(ivar['latitude'][loc['latitude'].compute()]))
        lons.append(float(ivar['longitude'][loc['longitude'].compute()]))
    
    track = pd.DataFrame([lons, lats, min_var]).transpose()
    track.columns = ['lon','lat','min']
    track.index = times
    return track

def process_track(track, track_Cowan_sliced, experiment_name):
    
    df_dist = pd.DataFrame({
        'lat_ref': track_Cowan_sliced.lat,
        'lon_ref': track_Cowan_sliced.lon,
        'lat_model': track.lat,
        'lon_model': track.lon
    })

    df_dist = df_dist.loc[track_Cowan_sliced.index]
    track = track.dropna()
    track['distance'] =  df_dist.apply(lambda row: calculate_distance(row), axis=1)

    track_name =  f'track_{experiment_name}.csv'
    track_path = os.path.join(args.output_directory, track_name)
    track.to_csv(track_path)
    print(f"{track_path} saved")  
    return track

def calculate_distance(row):
    """
    Calculate the distance between two coordinates in kilometers.

    Parameters:
        row (pandas.Series): A row of data containing latitude and longitude values.

    Returns:
        float: The distance between the two coordinates in kilometers.
    """
    start = (row['lat_ref'], row['lon_ref'])
    end = (row['lat_model'], row['lon_model'])
    return geodesic(start, end).km  

def main(args):
    # Validate inputs
    if not os.path.isdir(args.bench_directory):
        print("Invalid benchmark directory path.")
        return

    if args.ERA5 and not os.path.isfile(args.ERA5):
        print("Invalid ERA5 file path.")
        return
    
    ## Start the code ##
    benchs = glob.glob(args.bench_directory+'/run*')
    print(benchs)

    # Dummy for getting model times
    model_output = benchs[0]+'/latlon.nc'
    namelist_path = benchs[0]+"/namelist.atmosphere"

    # open data and namelist
    print('openinning first bench just to get dates...')
    model_data = open_model_dataset(benchs[0])
    namelist = f90nml.read(glob.glob(namelist_path)[0])
    times = get_simulation_times(namelist,model_data.compute()).tolist()
    first_day = datetime.datetime.strftime(times[0], '%Y-%m-%d %HZ')
    last_day = datetime.datetime.strftime(times[-1], '%Y-%m-%d %HZ')                      
    print('Analysis is from',first_day,'to',last_day)  

    print('\nOpening all data and putting it into a dictionary...')
    if 'grib' in os.path.splitext(args.ERA5)[1]:
        mslp = xr.open_dataset(args.ERA5, engine='cfgrib',
                            filter_by_keys={'typeOfLevel': 'surface'}
                            ).sel(time=slice(times[0],times[-1]),
                            latitude=slice(-20,-35),longitude=slice(-55,-30)).msl
    else:
        mslp = xr.open_dataset(args.ERA5).sel(time=slice(times[0],times[-1]),
                        latitude=slice(-20,-35),longitude=slice(-55,-30)).msl
        
    mslp = (mslp * units(mslp.units)).metpy.convert_units('hPa')    

    track_Cowan = pd.read_csv(os.path.join(args.output_directory, 'track_Cowan.csv'), index_col=0)
    track_Cowan_sliced = track_Cowan.loc[slice(first_day, last_day)]  

    era_track = get_track(mslp, 'time') 
    era_track_processed = process_track(era_track, track_Cowan_sliced, 'ERA5')  
                        
    for bench in benchs:
        experiment_name = get_experiment_name(bench)
        print('\n', 'processing track for', experiment_name)
        
        if "off_" in experiment_name: continue
        if "off_" in experiment_name: print('skipping', experiment_name)

        model_data  = open_model_dataset(bench, times=times)

        surface_pressure = model_data['surface_pressure'] * units.Pa
        surface_height = model_data['zgrid'].isel(nVertLevelsP1=0) * units.m
        surface_t = model_data.t2m * units.K
        surface_miximg_ratio = model_data.q2 * units('kg/kg')
        mean_virtual_temperature = virtual_temperature(
            surface_t,surface_miximg_ratio
            ).mean(dim='Time')
        
        slp = surface_pressure_to_mslp(surface_pressure, mean_virtual_temperature, surface_height)
        slp = slp.metpy.convert_units('hPa')
        slp = slp.sel(Time=track_Cowan_sliced.index)
        
        track = get_track(slp, 'Time')
        track_processed = process_track(track, track_Cowan_sliced, experiment_name)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bdir', '--bench_directory', type=str,
                        default='/p1-nemo/danilocs/mpas/MPAS-BR/benchmarks/Catarina_physics-test/Catarina_250-8km.microp_scheme.convection_scheme/',
                        help='path to benchmark directory')
    parser.add_argument('-odir', '--output_directory', type=str,
                        default='../experiments_48h/tracks_48h',
                        help='path to directory to save data')
    parser.add_argument('-e', '--ERA5', type=str,
                        default='/p1-nemo/danilocs/mpas/MPAS-BR/met_data/ERA5/DATA/2004/Catarina-PhysicsTest_ERA5.grib',
                        help='path to ERA5 file')
    args = parser.parse_args()
    main(args)
