#pragma once
#include <curand_kernel.h>

__global__ void kernel_setup_prng(int n, int seed, curandState *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    if(id < n){
        curand_init(seed, id, 0, &state[id]);
    }
}

template <typename T>
__global__ void kernel_random_array(int n, T max, curandState *state, T *array){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= n){ return; }
    float x = curand_uniform(&state[id]);
    array[id] = x*max;
}

curandState* setup_curand(int n, int seed) {
    curandState *devStates;
    cudaMalloc((void **)&devStates, n * sizeof(curandState));

    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_setup_prng<<<grid, block>>>(n, seed, devStates);
    CUDA_CHECK (cudaDeviceSynchronize() );

    return devStates;
}

template <typename T>
T* create_random_array_dev(int n, T max, curandState* devStates){
    T* darray;
    cudaMalloc(&darray, sizeof(T)*n);

    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_random_array<<<grid,block>>>(n, max, devStates, darray);
    CUDA_CHECK( cudaDeviceSynchronize() );

    return darray;
}

__global__ void kernel_random_uniform_particles(int n, Particle* parray, float3 dim, float smin, float smax, curandState* state){
    	int id = threadIdx.x + blockIdx.x * blockDim.x;
    	if (id >= n) return;
	parray[id].id 	= id;
	parray[id].pos 	= make_float3(	curand_uniform(&state[id]) * dim.x, 
					curand_uniform(&state[id]) * dim.y,
					curand_uniform(&state[id]) * dim.z );
	parray[id].size = curand_uniform(&state[id]) * (smax-smin) + smin; 
        //printf("tid: %d\n", id);
}

Particle* create_random_uniform_particles(int n, float3 dim_world, float smin, float smax, curandState* devStates) {
    Particle* d_particle;
    cudaMalloc(&d_particle, sizeof(Particle)*n);

    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_random_uniform_particles<<<grid,block>>>(n, d_particle, dim_world, smin, smax, devStates);
    CUDA_CHECK( cudaDeviceSynchronize() );

    return d_particle;
}

