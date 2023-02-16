import numpy as np
import xarray as xr
import time

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
    A = np.array(pv[:levelSize+1])
    B = np.array(pv[levelSize+1:])
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
    dA = A[:-1]-A[1:]
    dB = B[:-1]-B[1:]
    dA = np.broadcast_to(dA,(broadcast_shape))
    dB = np.broadcast_to(dB,(broadcast_shape))
    ps = np.broadcast_to(ps,(broadcast_shape_ps))
    ps = np.moveaxis(ps, 0, -1)
    dp=dA+dB*ps
    return dp


def mlevel_vint(var,ps,model='erai'):
    # gravitional constant
    g=9.81
    # calculate dp matrix for vertical integration
    dp = cal_dp(ps,model=model)
    dp = np.moveaxis(dp, -1, 0)
    # vertical integration from 0 to ps int(var/g*dp)
    var_vint = np.sum(var*dp,axis=0)/g
    return var_vint


if __name__ == '__main__':

    # read data
    t0 = time.time()
    ds = xr.open_dataset('/Projects/erai_modellevel/q_ml_1980_rechunked.nc').isel(time=0).load()
    da_lp = xr.open_dataset('/Projects/erai_flux/modellevel/zlnsp_ml_1980.nc').lnsp.isel(time=0).load()
    t1 = time.time()
    total = (t1-t0)
    print("read data",total,"secs")

    # calculate vertical integration
    t0 = time.time()
    q_vi = mlevel_vint(ds.q.values,np.exp(da_lp).values,model='erai')
    t1 = time.time()
    total = (t1-t0) 
    print("vertical integration",total,"secs")

    da_q_vint = ds.q.isel(level=0,drop=True).copy(data=q_vi)
    # da_q_vint = q_vi
    ds_q_vint = xr.Dataset()
    ds_q_vint.attrs['comments'] = 'variable vertical integrated along model level'
    ds_q_vint['q_vint'] = da_q_vint
    ds_q_vint['q_vint'].attrs['long_name'] = 'vertical integrated q along model level'

    # ds_q_vint.to_netcdf('/Projects/erai_modellevel/q_vint_1980.nc',encoding=encoding_list)
 


