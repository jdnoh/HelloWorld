from numba import jit, cuda

@cuda.jit
def hello_fromGPU(n):
    tid = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    print("Hello World from thread ", n,tid)

def hello_fromCPU():
    print("Hello World from CPU\n")

hello_fromGPU[2,3](0)
#cuda.synchronize()
hello_fromCPU()
