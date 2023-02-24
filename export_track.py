#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:43:19 2023

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

from metpy.constants import g
from metpy.constants import Rd
from metpy.calc import density
from metpy.calc import height_to_geopotential

from geopy.distance import geodesic

lat_domain = slice(-20,-35)
lon_domain = slice(-55,-30)

def get_exp_name(bench):
    expname = bench.split('/')[-1].split('run.')[-1]
    microp = expname.split('.')[0].split('_')[-1]
    cumulus = expname.split('.')[-1].split('_')[-1] 
    return microp+'_'+cumulus

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

def open_dataset(bench, times=None):
    data =  xr.open_dataset(bench+'/latlon.nc').sortby(
        'latitude', ascending=False).sel(
            latitude=lat_domain,longitude=lon_domain)
    if times:
        data = data.assign_coords({"Time":times})
    return data

def pressure_to_slp(pressure, z, zlevs):
    pres_height = interplevel(pressure, z[:,:-1], zlevs)
    slp = pres_height.isel(level=1)
    return slp

def pressure_to_mslp(surface_pressure, surface_hgt, surface_t,
                     surface_miximg_ratio):
    zsurf = ((Rd*surface_t/g
              )*np.log(surface_pressure.metpy.dequantify()/100000.0)
             ).squeeze()
    rho = density(surface_pressure, surface_t, surface_miximg_ratio)
    return surface_pressure - (rho*g*zsurf)/100.0

def get_track(slp, TimeIndexer):
    min_var, times = [], []
    lats, lons = [], []
    for t in slp[TimeIndexer]:
        datestr = pd.to_datetime(t.values)
        times.append(str(datestr))
        ivar = slp.sel({TimeIndexer:t})
        
        varmin = ivar.min()
        min_var.append(float(varmin))
        
        loc = ivar.argmin(dim=['latitude','longitude'])
        lats.append(float(ivar['latitude'][loc['latitude']]))
        lons.append(float(ivar['longitude'][loc['longitude']]))
    
    track = pd.DataFrame([lons, lats, min_var]).transpose()
    track.columns = ['lon','lat','min']
    track.index = times
    return track

def calculate_distance(row):
    start = (row['lat_ref'], row['lon_ref'])
    end = (row['lat_model'], row['lon_model'])
    return geodesic(start, end).km  

## Parser options ##
parser = argparse.ArgumentParser()
parser.add_argument('-bdir','--bench_directory', type=str, required=True,
                        help='''path to benchmark directory''')
parser.add_argument('-o','--output', type=str, default=None,
                        help='''output name to append file''')
parser.add_argument('-e','--ERA5', type=str, default=None,
                        help='''wether to validade with ERA5 data''')
args = parser.parse_args()

## Start the code ##
benchs = glob.glob(args.bench_directory+'/run*')

# Dummy for getting model times
model_output = benchs[0]+'/latlon.nc'
namelist_path = benchs[0]+"/namelist.atmosphere"

# open data and namelist
model_data = open_dataset(benchs[0])
namelist = f90nml.read(glob.glob(namelist_path)[0])
times = get_times_nml(namelist,model_data.compute()).tolist()
first_day = datetime.datetime.strftime(times[0], '%Y-%m-%d %HZ')
last_day = datetime.datetime.strftime(times[-1], '%Y-%m-%d %HZ')                      
print('Analysis is from',first_day,'to',last_day)  

# For interpolating pressure from height to isobaric
z = model_data.zgrid.expand_dims({'Time':times})
zmax = float(z.compute().max())
dz = 100
zlevs = np.arange(0, zmax, dz) * units.m

print('\nOpening all data and putting it into a dictionary...')
mslp = xr.open_dataset(args.ERA5, engine='cfgrib',
                filter_by_keys={'typeOfLevel': 'surface'}
                ).sel(time=slice(times[0],times[-1]),
                latitude=slice(-20,-35),longitude=slice(-55,-30)).msl
mslp = (mslp * units(mslp.units)).metpy.convert_units('hPa')                    
era_track = get_track(mslp, 'time')
era_track.to_csv('tracks_48h/track_ERA5.csv')             
print('tracks_48h/track_ERA5.csv saved')                      
                      
for bench in benchs:
    experiment = get_exp_name(bench)
    print('\n',experiment)
    
    print('computing slp...')
    model_data  = open_dataset(bench, times=times)
    
    surface_pressure = model_data['surface_pressure'] * units.Pa
    z0 = model_data['zgrid'].isel(nVertLevelsP1=0) * units.m
    surface_hgt = height_to_geopotential(z0)/g
    surface_t = model_data.t2m * units.K
    surface_miximg_ratio = model_data.q2 * units('kg/kg')
    slp = pressure_to_mslp(surface_pressure, surface_hgt, surface_t,
                         surface_miximg_ratio)/100
    slp = slp.metpy.dequantify().where(model_data['xland'] == 2, 9999)[0].plot()
    
    # slp = pressure_to_slp(pressure_dask.compute(), z.compute(), zlevs)
    
    print('getting track..')
    track = get_track(slp, 'Time')
    
    df_dist = pd.DataFrame({
    'lat_ref': era_track.lat, 'lon_ref': era_track.lon,
    'lat_model': track.lat, 'lon_model': track.lon})
        
    track['distance'] =  df_dist.apply(
        lambda row: calculate_distance(row), axis=1)
    
    track.to_csv('tracks_48h/track_'+experiment+'.csv')   
    print('tracks_48h/track_'+experiment+'.csv saved')