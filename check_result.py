import xarray as xr
import numpy as np


data_dir = '/home/tropical2extratropic/data/'
gpu_result = xr.open_dataset(data_dir+'q_1980_opt_anom_gpu.nc')
cpu_result = xr.open_dataset(data_dir+'q_1980_opt_anom.nc')

print('==================')
print('q_anom')
print((np.abs(cpu_result.q_anom)).mean().values)
diff = (np.abs(cpu_result.q_anom-gpu_result.q_anom)).mean().values
print(diff)
print(diff/(np.abs(cpu_result.q_anom)).mean().values)

gpu_result = xr.open_dataset(data_dir+'q_1980_opt_tfilter_gpu.nc')
cpu_result = xr.open_dataset(data_dir+'q_1980_opt_tfilter.nc')

print('==================')
print('q_filtered')
print((np.abs(cpu_result.q_filter)).mean().values)
diff = (np.abs(cpu_result.q_filter-gpu_result.q_filter)).mean().values
print(diff)
print(diff/(np.abs(cpu_result.q_filter)).mean().values)

gpu_result = xr.open_dataset(data_dir+'dp_gpu.nc')
cpu_result = xr.open_dataset(data_dir+'dp.nc')

print('==================')
diff = (np.abs(cpu_result.dp-gpu_result.dp)).mean().values
print(diff)

gpu_result = xr.open_dataset(data_dir+'q_vint_1980_opt_tfilter_gpu.nc')
cpu_result = xr.open_dataset(data_dir+'q_vint_1980_opt_tfilter.nc')

print('==================')
print('q_int')
print(np.abs(cpu_result.q_vint).mean().values)
diff = (np.abs(cpu_result.q_vint-gpu_result.q_vint)).mean().values
print(diff)
print(diff/np.abs(cpu_result.q_vint).mean().values)