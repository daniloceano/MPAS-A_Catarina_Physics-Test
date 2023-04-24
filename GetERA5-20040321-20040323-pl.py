# List of variables https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-DataorganisationandhowtodownloadERA5

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type':'reanalysis',
        'format':'grib',
        'pressure_level':[
            '1','2','3',
            '5','7','10',
            '20','30','50',
            '70','100','125',
            '150','175','200',
            '225','250','300',
            '350','400','450',
            '500','550','600',
            '650','700','750',
            '775','800','825',
            '850','875','900',
            '925','950','975',
            '1000'
        ],
        'date':'20040324/20040329',
        'area':'-5/-50/-50/-10',
        'time':'00/to/23/by/1',
        'variable':[
            'geopotential','vertical_velocity',
            'temperature','u_component_of_wind','v_component_of_wind'
        ]
    },
    'Catarina-2403-2903.grib')
