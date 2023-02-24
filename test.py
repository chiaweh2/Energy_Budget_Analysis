import numpy as np
import cupy as cp

print('a')
#x_cpu=np.array([1,2,3])
#print(x_cpu)
x_gpu = cp.array([1, 2, 3])

print(x_gpu)
