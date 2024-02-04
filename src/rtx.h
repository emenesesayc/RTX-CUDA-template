#pragma once

void nn_rtx(int n, int steps, int alg, Particle* particles, Particle *d_particles, curandState* devStates, int* neighbors, int* nneigh, float radius_searching, CmdArgs args) {
    Timer timer;
    printf("--------------------- RTX OptiX Nearest-Neighbors Searching %-5s ---------------------\n", algStr[alg]);

    // 1) Create AABBs of particles (TODO GPU)
    printf("Generating AABB..................\n"); fflush(stdout); timer.restart();
    
    OptixAabb *aabb_array = (OptixAabb*)malloc(sizeof(OptixAabb) * n), *d_aabb_array;
    for(int i=0; i<n; i++){
        float minx = particles[i].pos.x - radius_searching;
        float miny = particles[i].pos.y - radius_searching;
        float minz = particles[i].pos.z - radius_searching;
        float maxx = particles[i].pos.x + radius_searching;
        float maxy = particles[i].pos.y + radius_searching;
        float maxz = particles[i].pos.z + radius_searching;
        aabb_array[i] = {minx, miny, minz, maxx, maxy, maxz};
        //printf("aabb cpu x: min: %f, max: %f, radius: %f\n", aabb_array[i].minX, aabb_array[i].maxX, radius_searching);
    } 
    CUDA_CHECK(cudaMalloc(&d_aabb_array, sizeof(OptixAabb) * n));
    CUDA_CHECK(cudaMemcpy(d_aabb_array, aabb_array, sizeof(OptixAabb) * n, cudaMemcpyHostToDevice));

    timer.stop(); printf("done: %f ms\n",timer.get_elapsed_ms());


    // 2) RTX OptiX Config (ONCE)
    printf("RTX Config...........................\n"); fflush(stdout);
    timer.restart();
    GASstate state;
    printf("creating context...\n");
    createOptixContext(state);
    printf("creating modules...\n");
    loadAppModule(state);
    printf("creating group...\n");
    createGroupsClosestHit(state);
    printf("creating pipeline...\n");
    createPipeline(state);
    printf("populate sbt...\n");
    populateSBT(state);

    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());


    // 3) Populate and move parameters to device (ONCE)
    printf("Malloc and init arrays of params to the GPU ........."); fflush(stdout); 
    timer.restart();

    int *d_neighbors, *d_nneigh;
    CUDA_CHECK( cudaMalloc(&d_neighbors, sizeof(int) * n * NMAX) );
    CUDA_CHECK( cudaMalloc(&d_nneigh, sizeof(int) * n) );
    memset(nneigh, 0, sizeof(int) * n); //TODO GPU
    //CUDA_CHECK( cudaMemcpy(d_nneigh, nneigh, sizeof(int) * n, cudaMemcpyHostToDevice) );  //TODO GPU

    Params params;
    params.handle = state.gas_handle; //after the BVH construction
    params.r = radius_searching;
    params.d_neighbors = d_neighbors;
    params.d_nneigh = d_nneigh;
    params.d_particles = d_particles;
    params.d_aabb_array = d_aabb_array;
    params.NMAX = NMAX;
    Params *device_params;
    printf("(%7.3f MB)....", (double)sizeof(Params)/1e3); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&device_params, sizeof(Params)));
    //CUDA_CHECK(cudaMemcpy(device_params, &params, sizeof(Params), cudaMemcpyHostToDevice)); //after the BVH construction
    timer.stop();
    printf("done: %f ms\n", timer.get_elapsed_ms());
 

    // 4) Simulation
    int STEP_REBUILD = 5;
    bool GAS_ALLOW_COMPACTION = true;
    bool BUFFER_SAVING_MODE = true;


    printf("Simulating for %i steps\n", steps);

    printf(AC_MAGENTA "First copy AABB on GPU..................."); fflush(stdout); timer.restart();
    update_aabb_from_particles(d_particles, d_aabb_array, n, radius_searching);
    timer.stop(); printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);

    printf(AC_MAGENTA "First Build AS on GPU..................."); fflush(stdout); timer.restart();
    //buildASFromDeviceData(state, n, d_aabb_array, GAS_ALLOW_COMPACTION, BUFFER_SAVING_MODE);
    if(!BUFFER_SAVING_MODE){
        buildAS_classic(state, n, d_aabb_array, GAS_ALLOW_COMPACTION);
    }
    else {
        init_buildAS_optimized(state, n, d_aabb_array, GAS_ALLOW_COMPACTION);
        buildAS_optimized(state, GAS_ALLOW_COMPACTION);
        //buildASFromDeviceData(state, n, d_aabb_array, GAS_ALLOW_COMPACTION);
    }

    params.handle = state.gas_handle; //handle as been init by the construction of the BVH, update it in params
    CUDA_CHECK(cudaMemcpy(device_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
    timer.stop(); printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);


    for(int ki = 0; ki<steps; ++ki)
    {

        //CUDA_CHECK( cudaMemcpy(d_nneigh, nneigh, sizeof(int) * n, cudaMemcpyHostToDevice) );
        printf(AC_BOLDCYAN "\tInit nneigh........................" AC_RESET); fflush(stdout); timer.restart();
        init_nneigh(d_nneigh, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop(); printf(AC_BOLDCYAN "done: %f ms\n" AC_RESET, timer.get_elapsed_ms());


        // 4.3) launch optix
        printf(AC_BOLDCYAN "\tOptiX Launch [ %-3s ]..............." AC_RESET, algStr[alg]); fflush(stdout); timer.restart();
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(device_params), sizeof(Params), &state.sbt, n, 1, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop(); printf(AC_BOLDCYAN "done: %f ms\n" AC_RESET, timer.get_elapsed_ms());


        // 4.4) copying the result
        /*printf(AC_BOLDCYAN "\tCopying result DEVICE -> HOST........" AC_RESET); fflush(stdout);
        timer.restart();
        CUDA_CHECK( cudaMemcpy(neighbors, d_neighbors, sizeof(int) * n * NMAX, cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaMemcpy(nneigh, d_nneigh, sizeof(int) * n, cudaMemcpyDeviceToHost) );
        timer.stop(); printf(AC_BOLDCYAN "done: %f ms\n" AC_RESET, timer.get_elapsed_ms());*/


        // 4.5) update positions of particles following a pseudo brownian movement with CUDA kernel
        printf(AC_BOLDCYAN "\tParticles Random Movement.........." AC_RESET); fflush(stdout); timer.restart();
        move_particles_brownian(false, d_particles, devStates, n, 0.001f, make_float3(1,1,1));       
        timer.stop(); printf(AC_BOLDCYAN "done: %f ms\n" AC_RESET, timer.get_elapsed_ms());
        //CUDA_CHECK( cudaMemcpy(particles, d_particles, sizeof(Particle) * n, cudaMemcpyDeviceToHost) );
        //print_particles_array(particles, n);

        
        printf(AC_BOLDCYAN "\tUpdate AABB on GPU................."); fflush(stdout); timer.restart();
        update_aabb_from_particles(d_particles, d_aabb_array, n, radius_searching);
        timer.stop(); printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);
        

        if(ki % STEP_REBUILD == STEP_REBUILD-1){   
            printf(AC_MAGENTA "Rebuild AS on GPU......................"); fflush(stdout); timer.restart();

            //re_buildASFromDeviceData(state, n, d_aabb_array);
            //buildASFromDeviceData(state, n, d_aabb_array, GAS_ALLOW_COMPACTION, BUFFER_SAVING_MODE);
            if(!BUFFER_SAVING_MODE){
                buildAS_classic(state, n, d_aabb_array, GAS_ALLOW_COMPACTION);
            }
            else {
                buildAS_optimized(state, GAS_ALLOW_COMPACTION);
                //re_buildASFromDeviceData(state);
            }


            //params.handle = state.gas_handle; //handle as been init by the construction of the BVH, update it in params
            //CUDA_CHECK(cudaMemcpy(device_params, &params, sizeof(Params), cudaMemcpyHostToDevice));

            CUDA_CHECK( cudaDeviceSynchronize() );
            timer.stop(); printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);
        }
        else {             
            printf(AC_YELLOW "\tUpdating AS on GPU......................................"); fflush(stdout); timer.restart();
            updateASFromDevice(state, BUFFER_SAVING_MODE);
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop(); printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);
        }

        
    }

    CUDA_CHECK( cudaMemcpy(nneigh, d_nneigh, sizeof(int) * n, cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(neighbors, d_neighbors, sizeof(int) * NMAX * n, cudaMemcpyDeviceToHost) );

    // 6) clean up
    printf("cleaning up RTX environment..........\n"); fflush(stdout);
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    for (int i = 0; i < 3; ++i) {
        OPTIX_CHECK(optixProgramGroupDestroy(state.program_groups[i]));
    }
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(device_params));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    printf("done: %f ms\n", timer.get_elapsed_ms());
    printf("-------------------------------------------------------------------------\n\n");
    //TODO free malloc
}

