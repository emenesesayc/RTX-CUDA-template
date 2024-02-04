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

    int* neighbors = new int[n * args.nmax];
    int* nneigh = new int[n];

    // 2) computation
    //TODO simlation loop
    // method should compute only one iterarion
    switch(alg){
        case ALG_CLASSIC:
            nn_cpu(h_particles, n, neighbors, nneigh, radius, args);
            break;
        case ALG_GRID:
            //TODO
            break;
        case ALG_RTX:
            nn_rtx(n, steps, alg, h_particles, d_particles, devStates, neighbors, nneigh, radius, args);
            break;
    }
    // TODO particle movement
    // add check simulation inside main simulation?

    //print_particles_array(h_particles, n);
    print_int_array("number of neighbors", nneigh, n);
    print_all_neighbors(neighbors, nneigh, n, h_particles);

    printf("dist2(p3,p4) = %f\n", distance2(h_particles[3].pos, h_particles[4].pos));

    if (args.check) {

    }
    printf("Benchmark Finished\n");
    return 0;
}
