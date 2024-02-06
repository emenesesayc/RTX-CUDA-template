#pragma once
#include <assert.h>
#include <math.h>
//#include "src/utils.cuh"


void cpu_nns(Particle* particles, int N, int* neighbors, int* nneigh, float radius, CmdArgs args)
{
    //printf("cpu_nns: N:%d, NMAX:%d, radius:%f\n", N, NMAX, radius);
    //print_int_array("neighbors", neighbors, N*NMAX);
    //print_int_array("nneigh", nneigh, N);


    float radius2 = radius * radius;
    for(int i=0; i<N; i++){
        Particle p = particles[i];
        for(int j=i+1; j<N; j++){
            Particle pn = particles[j];
            if(distance2(&p, &pn) < radius2) {
                neighbors[i*NMAX + nneigh[i]++] = j;
                neighbors[j*NMAX + nneigh[j]++] = i;
                assert(nneigh[i] <= NMAX && nneigh[j] <= NMAX);
            }
        }
    }
    //printf("cpu_nns2\n");
}


void nn_cpu(Particle* particles, int N, int* neighbors, int* nneigh, float radius, CmdArgs args)
{
    printf("--------------------- CPU Naive Nearest-Neighbors Searching ---------------------\n");
    Timer timer;
    timer.restart();
    //init number of neighbors to zero to every particles
    memset(nneigh, 0, N*sizeof(int));
    cpu_nns(particles, N, neighbors, nneigh, radius, args);
    timer.stop(); printf("done: %f ms\n",timer.get_elapsed_ms());
    printf("-------------------------------------------------------------------------\n\n");
}
