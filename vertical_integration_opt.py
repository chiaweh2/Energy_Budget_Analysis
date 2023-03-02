#!/home/6embdqs6/.conda/envs/vint/bin/python
import numpy as np
import xarray as xr
import time
from scipy.ndimage import convolve1d

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
    A = np.array(pv[:levelSize+1],dtype='float32')
    B = np.array(pv[levelSize+1:],dtype='float32')


    return A, B

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
    dA = np.broadcast_to(dA,(broadcast_shape))
    dB = np.broadcast_to(dB,(broadcast_shape))
    ps = np.broadcast_to(ps,(broadcast_shape_ps))
    ps = np.moveaxis(ps, 0, -1)
    dp=dA+dB*ps
    return dp


def mlevel_vint(da_var,da_log_ps,model='erai'):
    var = da_var.data
    ps = np.exp(da_log_ps.data)

    # gravitional constant
    g = np.int32(9.81)
    # calculate dp matrix for vertical integration
    dp = cal_dp(ps,model=model)
    dp = np.moveaxis(dp, -1, 1)
    da_dp = da_var.copy(data=dp)
    # da_q_vint = q_vi
    ds_dp = xr.Dataset()
    ds_dp.attrs['comments'] = 'variable vertical integrated along model level'
    ds_dp['dp'] = da_dp
    ds_dp['dp'].attrs['long_name'] = 'vertical integrated q along model level'
    ds_dp.to_netcdf('/home/tropical2extratropic/data/dp.nc')

    
    # vertical integration from 0 to ps int(var/g*dp)
    var_vint = np.sum(var*dp,axis=1, dtype='float32')/g
    return var_vint

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


def lanczos_filter_4d(da_var_anom, window, cutoff):

    wt = lanczos_low_pass_weights(window, cutoff)

    var_anom_filtered = convolve1d(
            da_var_anom.astype('float32').data,
            wt,
            axis=0,
            output='float32')

    da_var_anom_filtered = da_var_anom.copy(data=var_anom_filtered)
    # da_var_anom_filtered = da_var_anom.copy()

    # nlat=da_var_anom_filtered.latitude.size
    # nlon=da_var_anom_filtered.longitude.size
    # nlev=da_var_anom_filtered.level.size

    # for llev in range(nlev):
    #     for llon in range(nlon):
    #         for llat in range(nlat):
    #             # da_var_anom_filtered[:,llev,llat,llon] = np.convolve(
    #             #     wt,
    #             #     da_var_anom[:,llev,llat,llon].data,
    #             #     mode='same'
    #             #     )
    #             da_var_anom_filtered[:,llev,llat,llon] = convolve1d(
    #                 da_var_anom[:,llev,llat,llon].astype('float64').data,
    #                 wt,
    #                 output = 'float64',
    #                 mode='reflect'
    #                 )                
    #             # note: no need to add "values"
    #             ## note: change the [:,llev] depending on the dimension of the array

    return da_var_anom_filtered

if __name__ == '__main__':

    # read data
    t0 = time.time()
    # print('read data')
    ds = xr.open_dataset('./data/q_ml_1980_rechunked.nc').load()
    da_lp = xr.open_dataset('./data/zlnsp_ml_1980.nc').lnsp.load()
    da_lp = da_lp.astype('float32')
    ds['q'] = ds.q.astype('float32')
    # ds = xr.open_dataset('./data/q_ml_1980.nc').isel(time=slice(0,500)).load()
    # da_lp = xr.open_dataset('./data/zlnsp_ml_1980.nc').lnsp.isel(time=slice(0,500)).load()
    t1 = time.time()
    total = t1-t0
    print("read data",total,"secs")

    ##### calculate high frequency
    t0 = time.time()
    # calculate ano and low pass
    window = 96+96+1
    cutoff = 1/(8*4)   # 6hourly daily data (4 times daily) for 8 days
    da_anom = ds.q - ds.q.mean(dim='time')
    # da_q_anom = ds.q.copy(data=da_anom.data)
    # ds_q_anom = xr.Dataset()
    # ds_q_anom.attrs['comments'] = 'variable time filtered along time dim'
    # ds_q_anom['q_anom'] = da_q_anom
    # ds_q_anom['q_anom'].attrs['long_name'] = 'variable time filtered along time dim'
    # ds_q_anom.to_netcdf('/home/tropical2extratropic/data/q_1980_opt_anom.nc')

    da_anom_lowpass = lanczos_filter_4d(da_anom,window,cutoff)
    #calculate high pass
    da_anom = da_anom-da_anom_lowpass
    # da_q_filter = ds.q.copy(data=da_anom.data)
    # ds_q_filter = xr.Dataset()
    # ds_q_filter.attrs['comments'] = 'variable time filtered along time dim'
    # ds_q_filter['q_filter'] = da_q_filter
    # ds_q_filter['q_filter'].attrs['long_name'] = 'variable time filtered along time dim'

    # ds_q_filter.to_netcdf('/home/tropical2extratropic/data/q_1980_opt_tfilter.nc')   


    # calculate vertical integration
    q_vi = mlevel_vint(da_anom,da_lp,model='erai')
    t1 = time.time()
    total = t1-t0
    print("vertical integration",total,"secs")

    # da_q_vint = ds.q.isel(level=0,drop=True).copy(data=q_vi)
    # # da_q_vint = q_vi
    # ds_q_vint = xr.Dataset()
    # ds_q_vint.attrs['comments'] = 'variable vertical integrated along model level'
    # ds_q_vint['q_vint'] = da_q_vint
    # ds_q_vint['q_vint'].attrs['long_name'] = 'vertical integrated q along model level'

    # ds_q_vint.to_netcdf('/home/tropical2extratropic/data/q_vint_1980_opt_tfilter.nc')

