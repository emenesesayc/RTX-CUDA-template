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

__global__ void kernel_brownian_border_movement_particles(Particle* d_particles, curandState* curand_states, int n, float factor, float3 dim)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n) {
    //if(tid < n){
        //d_particles[tid].size += 0.2f;
        float3 npos = d_particles[tid].pos;
        float3 mov = make_float3(   curand_uniform(&curand_states[tid]) * 2.f - 1.f,
                                    curand_uniform(&curand_states[tid]) * 2.f - 1.f,
                                    curand_uniform(&curand_states[tid]) * 2.f - 1.f);// * make_float3(factor);
        npos.x += mov.x * factor;
        npos.y += mov.y * factor;
        npos.z += mov.z * factor;
        if(npos.x < 0 || npos.x > dim.x) npos.x -= 2*mov.x;
        if(npos.y < 0 || npos.y > dim.y) npos.y -= 2*mov.y;
        if(npos.z < 0 || npos.z > dim.z) npos.z -= 2*mov.z;
        d_particles[tid].pos = npos;
        tid += gridDim.x*blockDim.x;
    }
}

__global__ void kernel_brownian_modulo_movement_particles(Particle* d_particles, curandState* curand_states, int n, float factor, float3 dim)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n) {
    //if(tid < n){
        //d_particles[tid].size += 0.2f;
        float3 npos = d_particles[tid].pos;
        float3 mov = make_float3(   curand_uniform(&curand_states[tid]) * 2.f - 1.f,
                                    curand_uniform(&curand_states[tid]) * 2.f - 1.f,
                                    curand_uniform(&curand_states[tid]) * 2.f - 1.f);// * make_float3(factor);
        npos.x = fmod( (npos.x + (mov.x * factor)), dim.x );
        npos.y = fmod( (npos.y + (mov.y * factor)), dim.y );
        npos.z = fmod( (npos.z + (mov.z * factor)), dim.z );
        d_particles[tid].pos = npos;
        tid += gridDim.x*blockDim.x;
    }
}


void move_particles_brownian(bool modulo, Particle* d_particles, curandState* curand_states, int n, float factor, float3 dim)
{
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1);
    if(modulo) {
        kernel_brownian_modulo_movement_particles<<<grid, block>>>(d_particles, curand_states, n, factor, dim);
    } else {
        kernel_brownian_border_movement_particles<<<grid, block>>>(d_particles, curand_states, n, factor, dim);
        //kernel_brownian_movement_particles<<<1,1>>>(d_particles, curand_states, n, factor, dim);
    }

    cudaDeviceSynchronize();
}


