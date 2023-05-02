# RTX-CUDA-template
A simple RTX-CUDA interoperability test program based on the SAXPY example from NVIDIA's RTX Compute Samples https://github.com/NVIDIA/rtx_compute_samples

The sample code simulates `n` randomly generated numbers, by shifting their value (as if was a 1D coordinate) by a random translation, and computes the minimum value at each time step. The user can choose one of the following approaches for computing such minimum value in GPU:
- CUDA Kernel
- CUB
- Thrust
- RTX OptiX 

## Dependencies
- CUDA 12 or later
- OptiX 7.7 or later
- Nvidia driver 530.41 or later
- CUB (for comparison)
- THrust (for comparison)

## Compile and run
Change OptiX path in `Makefile`
```
make
./rtxcuda <dev> <n> <s> <alg>
dev = device ID
n   = problem size
s   = number of simulation steps
alg = algorithm
   1 -> WARP_SHUFFLE
   2 -> CUB
   3 -> THRUST
   4 -> RTX_CLOSEST_HIT
```

## Example execution
`./rtxcuda 0 $((2**20)) 1 4`
