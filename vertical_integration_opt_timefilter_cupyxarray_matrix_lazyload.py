#!/home/6embdqs6/.conda/envs/vint/bin/python
import time
import cupy_xarray
import cupy as cp
import numpy as np
import xarray as xr
from cupyx.scipy.ndimage import convolve1d
import nvtx
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

@nvtx.annotate()
def get_A_B_erai(levelSize=60):
    """
    This function return A and B array used to calculate the half level
    (interface of model levels) pressure.

    source: ECMWF ERA-Interim GRIB file (the pv array in the function)
    
    return A, B
    """
    pv =  [
      0.0000000000e+000, 2.0000000000e+001, 3.8425338745e+001, 6.3647796631e+001, 9.5636962891e+001,
      1.3448330688e+002, 1.8058435059e+002, 2.3477905273e+002, 2.9849584961e+002, 3.7397192383e+002,
      4.6461816406e+002, 5.7565112305e+002, 7.1321801758e+002, 8.8366040039e+002, 1.0948347168e+003,
      1.3564746094e+003, 1.6806403809e+003, 2.0822739258e+003, 2.5798886719e+003, 3.1964216309e+003,
      3.9602915039e+003, 4.9067070313e+003, 6.0180195313e+003, 7.3066328125e+003, 8.7650546875e+003,
      1.0376125000e+004, 1.2077445313e+004, 1.3775324219e+004, 1.5379804688e+004, 1.6819472656e+004,
      1.8045183594e+004, 1.9027695313e+004, 1.9755109375e+004, 2.0222203125e+004, 2.0429863281e+004,
      2.0384480469e+004, 2.0097402344e+004, 1.9584328125e+004, 1.8864750000e+004, 1.7961359375e+004,
      1.6899468750e+004, 1.5706449219e+004, 1.4411125000e+004, 1.3043218750e+004, 1.1632757813e+004,
      1.0209500000e+004, 8.8023554688e+003, 7.4388046875e+003, 6.1443164063e+003, 4.9417773438e+003,
      3.8509133301e+003, 2.8876965332e+003, 2.0637797852e+003, 1.3859125977e+003, 8.5536181641e+002,
      4.6733349609e+002, 2.1039389038e+002, 6.5889236450e+001, 7.3677425385e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000, 0.0000000000e+000,
      7.5823496445e-005, 4.6139489859e-004, 1.8151560798e-003, 5.0811171532e-003, 1.1142909527e-002,
      2.0677875727e-002, 3.4121163189e-002, 5.1690407097e-002, 7.3533833027e-002, 9.9674701691e-002,
      1.3002252579e-001, 1.6438430548e-001, 2.0247590542e-001, 2.4393314123e-001, 2.8832298517e-001,
      3.3515489101e-001, 3.8389211893e-001, 4.3396294117e-001, 4.8477154970e-001, 5.3570991755e-001,
      5.8616840839e-001, 6.3554745913e-001, 6.8326860666e-001, 7.2878581285e-001, 7.7159661055e-001,
      8.1125342846e-001, 8.4737491608e-001, 8.7965691090e-001, 9.0788388252e-001, 9.3194031715e-001,
      9.5182150602e-001, 9.6764522791e-001, 9.7966271639e-001, 9.8827010393e-001, 9.9401944876e-001,
      9.9763011932e-001, 1.0000000000e+000 ]
    ## extract A and B 
    A = cp.array(pv[:levelSize+1], dtype='float32')
    B = cp.array(pv[levelSize+1:], dtype='float32')


    return A, B

@nvtx.annotate()
def cal_dp(ps,model='erai'):
    """
    Calculate the n+1 dim dp matrix for vertical dp intergal.
    input:
    ps - numpy array in n dim
    return:
    dp - nunpy array in n+1 dim with the first n dim in the shape of 
        ps and n+1 dim with the same length of model level 
    """
    if model in ['erai']:
        A,B = get_A_B_erai()
    nlevel = np.max(A.shape)-1
    broadcast_shape_ps = [nlevel]
    broadcast_shape = []
    for n in ps.shape:
        broadcast_shape_ps.append(n)
        broadcast_shape.append(n)
    broadcast_shape.append(nlevel)
    broadcast_shape_ps = tuple(broadcast_shape_ps)
    broadcast_shape = tuple(broadcast_shape)
    dA = A[1:]-A[:-1]
    dB = B[1:]-B[:-1]
    dA = cp.broadcast_to(dA,(broadcast_shape))
    dB = cp.broadcast_to(dB,(broadcast_shape))
    ps = cp.broadcast_to(ps,(broadcast_shape_ps))
    ps = cp.moveaxis(ps, 0, -1)
    dp = dA+dB*ps
    return dp

@nvtx.annotate()
def mlevel_vint(da_var,da_log_ps,model='erai'):
    var_gpu = da_var.data
    ps_gpu = cp.exp(da_log_ps.data, dtype='float32')

    # gravitional constant
    g = cp.int32(9.81)
    # calculate dp matrix for vertical integration
    dp = cal_dp(ps_gpu,model=model)
    dp = cp.moveaxis(dp, -1, 1)

    # dp_cpu = cp.asnumpy(dp)
    # da_dp = da_var.copy(data=dp_cpu)
    # # da_q_vint = q_vi
    # ds_dp = xr.Dataset()
    # ds_dp.attrs['comments'] = 'variable vertical integrated along model level'
    # ds_dp['dp'] = da_dp
    # ds_dp['dp'].attrs['long_name'] = 'vertical integrated q along model level'
    # ds_dp.to_netcdf('/home/tropical2extratropic/data/dp_gpu.nc')

    # vertical integration from 0 to ps int(var/g*dp)
    var_vint = cp.sum(var_gpu*dp,axis=1, dtype='float32')/g


    return var_vint

@nvtx.annotate()
def lanczos_low_pass_weights(window, cutoff):
    """
    Calculate weights for a low pass Lanczos filter.
    Inputs:
    ================
    window: int
        The length of the filter window (odd number).
    cutoff: float
        The cutoff frequency(1/cut off time steps)
    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
#     sigma = 1.   # edit for testing to match with Charlotte ncl code
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

@nvtx.annotate()
def lanczos_filter_matrix(da_var_anom, window, cutoff):

    wt = lanczos_low_pass_weights(window, cutoff)

    with nvtx.annotate("wt_to_cupy", color="green"):
        wt = cp.asarray(wt)

    with nvtx.annotate("scipy convolve", color="red"):
        var_anom_filtered = convolve1d(
            da_var_anom.astype('float64').data,
            wt,
            axis=0,
            output='float64')
    
    del wt

    da_var_anom_filtered = da_var_anom.copy(data=var_anom_filtered)

    #output high pass
    return da_var_anom-da_var_anom_filtered

if __name__ == '__main__':

    # read data
    t0 = time.time()
    ds = xr.open_mfdataset('./data/q_ml_1980_rechunked.nc')
    da_lp = xr.open_mfdataset('./data/zlnsp_ml_1980.nc').lnsp
    da_lp = da_lp.astype('float32')
    ds['q'] = ds.q.astype('float32')
    t1 = time.time()
    total = t1-t0
    print("read data (lazy)",total,"secs")

    # host to device
    t0 = time.time()
    step = 50
    q_vi = cp.array(da_lp.copy().data)
    da_lp_gpu = da_lp.load().cupy.as_cupy()
    seg_time = []
    for stride in range(0,len(da_lp.longitude.data)+1, step):

        t00 = time.time()
        da_q_load = ds.q.isel(longitude=slice(stride,stride+step+1)).load()
        t11 = time.time()
        seg_time.append(t11-t00)
        print("read data segment",t11-t00,"secs")
        da_q_gpu_load = da_q_load.cupy.as_cupy()
        # da_q_gpu = ds.q.map_blocks(cp.asarray)
        # da_lp_gpu = da_lp.map_blocks(cp.asarray)

        ##### calculate high frequency
        # calculate ano and low pass
        window = 96+96+1
        cutoff = 1/(8*4)   # 6hourly data (4 times daily) for 8 days
        da_q_gpu_load = da_q_gpu_load - da_q_gpu_load.mean(dim='time')

        # da_anom_cpu = da_anom_gpu.as_numpy()
        # da_q_anom = ds.q.copy(data=da_anom_cpu.data)
        # ds_q_anom = xr.Dataset()
        # ds_q_anom.attrs['comments'] = 'variable time filtered along time dim'
        # ds_q_anom['q_anom'] = da_q_anom
        # ds_q_anom['q_anom'].attrs['long_name'] = 'variable time filtered along time dim'
        # ds_q_anom.to_netcdf('/home/tropical2extratropic/data/q_1980_opt_anom_gpu.nc')

        da_q_gpu_load = lanczos_filter_matrix(da_q_gpu_load,window,cutoff)

        # da_anom_cpu = da_anom_gpu.as_numpy()
        # da_q_filter = ds.q.copy(data=da_anom_cpu.data)
        # ds_q_filter = xr.Dataset()
        # ds_q_filter.attrs['comments'] = 'variable time filtered along time dim'
        # ds_q_filter['q_filter'] = da_q_filter
        # ds_q_filter['q_filter'].attrs['long_name'] = 'variable time filtered along time dim'
        # ds_q_filter.to_netcdf('/home/tropical2extratropic/data/q_1980_opt_tfilter_gpuMatrix.nc')

        # calculate vertical integration
        q_vi[:,:,stride:(stride+step+1)] = mlevel_vint(da_q_gpu_load,da_lp_gpu.isel(longitude=slice(stride,stride+step+1)),model='erai')

    t1 = time.time()
    total_seg_load_time = np.sum(np.array(seg_time))
    total = t1-t0-total_seg_load_time
    print("total data seg load",total_seg_load_time,'secs')
    print("time filter + vertical integration",total,"secs")

    q_vi_cpu = cp.asnumpy(q_vi)
    # da_q_vint = ds.q.isel(level=0,drop=True).copy(data=q_vi_cpu)
    # # da_q_vint = q_vi
    # ds_q_vint = xr.Dataset()
    # ds_q_vint.attrs['comments'] = 'variable vertical integrated along model level'
    # ds_q_vint['q_vint'] = da_q_vint
    # ds_q_vint['q_vint'].attrs['long_name'] = 'vertical integrated q along model level'
    # ds_q_vint.to_netcdf('/home/tropical2extratropic/data/q_vint_1980_opt_tfilter_gpuMatrix.nc')
