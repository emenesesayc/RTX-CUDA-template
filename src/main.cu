#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#define BSIZE 1024
#define WARPSIZE 32
#define ALG_CLASSIC         0
#define ALG_GRID            1
#define ALG_RTX             2
const char *algStr[3] = {"CLASSIC", "GRID", "RTX"};
size_t NMAX;
char LOG[2048];
size_t LOG_SIZE = 1;

#include "src/rtx_params.h"
#include "common/common.h"
#include "common/Timer.h"
#include "src/rand.cuh"
#include "src/tools.h"
#include "src/device_tools.cuh"
#include "src/device_simulation.cuh"
#include "src/cuda_methods.cuh"
#include "rtx_params.h"
#include "src/rtx_functions.h"
#include "src/rtx.h"


int main(int argc, char *argv[]) {
    printf("----------------------------------\n");
    printf("  RTX-CUDA Template by Temporal   \n");
    printf("----------------------------------\n");

    CmdArgs args = get_args(argc, argv);
    int dev = args.dev;
    int n = args.n;
    int steps = args.steps;
    int alg = args.alg;
    float radius = args.r;
    NMAX = args.nmax;
    int GPU = args.alg;

    cudaSetDevice(dev);
    print_gpu_specs(dev);


    // 1) data on GPU, result has the resulting array and the states array
    float3 dim_world = make_float3(100.f, 100.f, 100.f);
    float min_psize = radius;
    float max_psize = radius;
    printf("Generating random particles\n");
    curandState* devStates = setup_curand(n, args.seed);
    Particle* d_particles = create_random_uniform_particles(n, dim_world, min_psize, max_psize, devStates);
    Particle* h_particles = (Particle*)malloc(sizeof(Particle) * n);
    CUDA_CHECK( cudaMemcpy(h_particles, d_particles, sizeof(Particle) * n, cudaMemcpyDeviceToHost) );

    int* h_neighbors = new int[n * args.nmax];
    int* h_nneigh = new int[n];
    int *d_neighbors, *d_nneigh;
    CUDA_CHECK( cudaMalloc(&d_nneigh, sizeof(int) * n) );
    CUDA_CHECK( cudaMalloc(&d_neighbors, sizeof(int) * n * NMAX) );

    RTNN rtnn;

    // 2) computation
    //TODO simlation loop
    // method should compute only one iterarion
    // setup
    switch(alg){
        case ALG_CLASSIC:
            break;
        case ALG_GRID:
            //TODO
            break;
        case ALG_RTX:
            printf("--------------------- RTX OptiX Nearest-Neighbors Searching %-5s ---------------------\n", algStr[args.alg]);
            rtnn.n = n;
            rtnn.d_nneigh = d_nneigh;
            rtnn.d_neighbors = d_neighbors;
            rtnn.d_particles = d_particles;
            rtnn.radius = args.r;
            rtnn.args = args;
            rtnn.config();
            rtnn.build_geom();
            rtnn.set_params();
            break;
    }

    // simulation
    //for (int it = 0; it < args.steps; ++it) {
        switch(alg){
            case ALG_CLASSIC:
                nn_cpu(h_particles, n, h_neighbors, h_nneigh, radius, args);
                break;
            case ALG_GRID:
                //TODO
                break;
            case ALG_RTX:
                rtnn.nn();
                break;
        }
        // TODO particle movement
        // add check simulation inside main simulation?
    //}

    // cleanup
    switch(alg){
        case ALG_CLASSIC:
            break;
        case ALG_GRID:
            //TODO
            break;
        case ALG_RTX:
            rtnn.cleanup();
            break;
    }

    if (GPU) {
        CUDA_CHECK( cudaMemcpy(h_nneigh, d_nneigh, sizeof(int) * n, cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaMemcpy(h_neighbors, d_neighbors, sizeof(int) * n * NMAX, cudaMemcpyDeviceToHost) );
    }

    //print_particles_array(h_particles, n);
    print_int_array("number of neighbors", h_nneigh, n);
    print_all_neighbors(h_neighbors, h_nneigh, n, h_particles);

    if (args.check) {

    }
    printf("Benchmark Finished\n");
    return 0;
}
