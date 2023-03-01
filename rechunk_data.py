"""
rechunking the dataset into desired chunk

"""
import xarray as xr
from dask.diagnostics import ProgressBar

if __name__ == '__main__':

    ds = xr.open_dataset('/home/tropical2extratropic/data/q_ml_1980.nc')

    # one time one chunk file 
    target_chunks = {
        'time': len(ds['time']),
        'latitude': 10,
        'longitude': 10,
        'level': len(ds['level']),
        }

    encoding_list = {}
    varname = 'q'
    # variable encoding
    encoding_list[varname] = {}
    encoding_list[varname]['chunksizes'] = [
        target_chunks['time'],
        target_chunks['level'],
        target_chunks['latitude'],
        target_chunks['longitude']
        ]
    encoding_list[varname]['contiguous'] = False
    encoding_list[varname]['zlib'] = False
    encoding_list[varname]['complevel'] = 2
    encoding_list[varname]['shuffle'] = False
    # encoding_list[varname]['missing_value'] = missing_value
    # encoding_list[varname]['_FillValue'] = _FillValue
    # encoding_list[varname]['dtype'] = 'float32'

    # output netcdf file rechunked
    with ProgressBar():
        ds.to_netcdf('/home/tropical2extratropic/data/q_ml_1980_rechunked1.nc',encoding=encoding_list)