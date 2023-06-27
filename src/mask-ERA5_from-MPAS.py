#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 22:27:31 2022

@author: danilocoutodsouza
"""

import xarray as xr
import argparse

def convert_lon(df,LonIndexer):
    df.coords[LonIndexer] = (df.coords[LonIndexer] + 180) % 360 - 180
    df = df.sortby(df[LonIndexer])
    return df

## Parser options ##
parser = argparse.ArgumentParser()

parser.add_argument('-e','--ERA5', type=str, required=True,
                        help='''ERA5 file to be masked''')
parser.add_argument('-m','--MPAS', type=str, default=None, required=True,
                        help='''MPAS-A file to read mask from''')

args = parser.parse_args()

era_file = args.ERA5
masked_file = args.MPAS

era_data = convert_lon(xr.open_dataset(era_file), 'longitude')
masked_data = convert_lon(xr.open_dataset(masked_file),'longitude')

print('resampling masked data to match same timesteps as ERA5...')
masked_data = masked_data.resample(Time='1H').mean()

min_lat = masked_data.latitude.min()
max_lat = masked_data.latitude.max()
min_lon = masked_data.longitude.min()
max_lon = masked_data.longitude.max()

print('slicing ERA5 to masked data domain...')
era_data_sliced = era_data.sel(latitude=slice(max_lat,min_lat)
                               ).sel(longitude=slice(min_lon,max_lon))
    
print('interpolating ERA5 data to masked data grid...')
interp_data = masked_data.interp(latitude=era_data_sliced.latitude,
                                 longitude=era_data_sliced.longitude)
print('ok')

print('adjusting datasets time and levels to have the same length, if necessary..')
if len(interp_data.Time) > len(era_data_sliced.time):
    interp_data = interp_data.sel(Time=slice(era_data_sliced.time.min().values,
                                             era_data_sliced.time.max().values))
elif len(era_data_sliced.time) > len(interp_data.Time):
    era_data_sliced = era_data_sliced.sel(time=slice(interp_data.Time.min().values,
                                                     interp_data.Time.max().values))
else:
    print('it was not necessary for time dimension')
if len(interp_data.level) > len(era_data_sliced.isobaricInhPa):
    interp_data = interp_data.sel(level=slice(era_data_sliced.level.min().values,
                                             era_data_sliced.level.max().values)) 
elif len(era_data_sliced.isobaricInhPa) > len(interp_data.level):
    era_data_sliced = era_data_sliced.sel(isobaricInhPa=slice(interp_data.level.min().values,
                                                     interp_data.level.max().values))
else:
    print('it was not necessary for level dimension')

interp_data2 = interp_data.copy()
interp_data2 = interp_data2.rename({'Time': 'time'})
interp_data2.coords['time'] = era_data_sliced.coords['time']
interp_data2 = interp_data2.reset_coords('standard_height',drop=True)

print('applying mask to ERA5 data...')
var_masked = era_data_sliced.where(~interp_data2.uwnd.isnull())

print('returning temperature to its original shape..')
var_masked['t'] = era_data_sliced['t']

print('saving masked ERA5 data..') 
fname = era_file.split('.grb')[0]+'-masked.nc'
var_masked.to_netcdf(fname)
