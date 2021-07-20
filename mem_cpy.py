import numpy as np
from numba import cuda, jit

@cuda.jit
def add_constant(n, arr):
    pos = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    if n<arr.size:
        arr[pos] += n

dNum = 1<<10
# ndarray in the host 
h_Array = np.arange(dNum, dtype=int)

# allocate an empty device ndarray
d_Array = cuda.device_array_like(h_Array)
# copy from host to device
d_Array = cuda.to_device(h_Array)

# kernel
nThreads = 128
nBlocks = (dNum+nThreads-1)//nThreads
add_constant[nBlocks, nThreads](1, d_Array)

print(h_Array[0])

# copy from device to host
d_Array.copy_to_host(h_Array)

print(h_Array[0])