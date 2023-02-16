import time
import numpy as np
import xarray as xr



def cal_ml_rhodzVI(da_var,da_sp):
    # A and B parameters to calculate pressures for model levels,    ########## HH: these are the A, B on "half levels" (interface of model levels)
    #  extracted from an ECMWF ERA-Interim GRIB file and then hardcoded here
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
    da_level = da_var['level']  
    ## calculate A, b, 
    levelSize=60
    A = pv[0:levelSize+1]
    B = pv[levelSize+1:]
    #Get a list of level numbers in reverse order
    da_reversedlevel=da_level.isel(level=slice(None, None, -1)).astype(int)
    #Integrate up into the atmosphere from lowest level
    # T_ini=time.time()
    g=9.81
    # calculate first da_Ph_levplusone (surface, lev = 60)
    da_Ph_levplusone = A[levelSize] + (B[levelSize]*da_sp)  ## the level at sfc (because it is the first "level plus one" corresponding to first model level <level = 60>)
    da_var_VI=np.zeros(np.shape(da_sp))
    for lev in da_reversedlevel.values:  ###### HH: 60 (sfc), 59... -> 1 (top)  ## loop through "model levels"!
        # print(lev)
        da_var_level=da_var.sel(level=lev)
        #compute the pressures (on half-levels) Ph:Phalf
        da_Ph_lev = A[lev-1] + (B[lev-1] * da_sp)
        dP = da_Ph_levplusone-da_Ph_lev
        da_var_VI = da_var_VI + (da_var_level*dP/g)
        da_Ph_levplusone = da_Ph_lev
    # dT=(time.time()-T_ini)
    # print(str(dT/60)+'min')
    return da_var_VI


##### main script ####
t0 = time.time()
ds = xr.open_dataset('./data/q_ml_1980.nc').isel(time=slice(0,100)).load()
da_lp = xr.open_dataset('./data/zlnsp_ml_1980.nc').lnsp.isel(time=slice(0,100)).load()
t1 = time.time()
total = (t1-t0) 
print("read data",total,"secs")

t0 = time.time()
da_ps = np.exp(da_lp)
da_q_vi=cal_ml_rhodzVI(ds.q,da_ps)
t1 = time.time()
total = (t1-t0)
print("vertical integration",total,"secs")


# da_q_vi.to_netcdf('/Projects/erai_modellevel/q_vint_1980_hh.nc')
