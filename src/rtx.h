#pragma once

struct RTNN {
    int n;
    int *d_nneigh;
    int *d_neighbors;
    float radius;
    Particle *d_particles;
    GASstate state;
    Params *d_params;
    CmdArgs args;

    
    //int STEP_REBUILD = 5;
    bool GAS_ALLOW_COMPACTION = true;
    bool BUFFER_SAVING_MODE = true;
    OptixAabb *d_aabb_array;

    void config() {
        Timer timer;
        printf("RTX Config...........................\n"); fflush(stdout);
        timer.restart();
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
    }

    void build_geom() {
        Timer timer;
        printf(AC_MAGENTA "Build AABBs on GPU..................."); fflush(stdout); timer.restart();
        timer.restart();
        CUDA_CHECK(cudaMalloc(&d_aabb_array, sizeof(OptixAabb) * n));
        update_aabb_from_particles(d_particles, d_aabb_array, n, radius);
        timer.stop(); printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);

        printf(AC_MAGENTA "First Build AS on GPU..................."); fflush(stdout); timer.restart();
        if(!BUFFER_SAVING_MODE){
            buildAS_classic(state, n, d_aabb_array, GAS_ALLOW_COMPACTION);
        }
        else {
            init_buildAS_optimized(state, n, d_aabb_array, GAS_ALLOW_COMPACTION);
            buildAS_optimized(state, GAS_ALLOW_COMPACTION);
        }
    }

    void set_params() {
        Timer timer;
        printf("Set OptiX Params ........."); fflush(stdout);
        timer.restart();
        Params params;
        params.handle = state.gas_handle; //after the BVH construction
        params.r = radius;
        params.d_neighbors = d_neighbors;
        params.d_nneigh = d_nneigh;
        params.d_particles = d_particles;
        params.d_aabb_array = d_aabb_array;
        params.NMAX = NMAX;
        printf("(%7.3f MB)....", (double)sizeof(Params)/1e3); fflush(stdout);
        CUDA_CHECK(cudaMalloc(&d_params, sizeof(Params)));
        CUDA_CHECK(cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
        timer.stop();
        printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);
    }

    void nn() {
        Timer timer;
        timer.restart();

        printf(AC_BOLDCYAN "\tInit nneigh........................" AC_RESET); fflush(stdout); timer.restart();
        init_nneigh(d_nneigh, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop(); printf(AC_BOLDCYAN "done: %f ms\n" AC_RESET, timer.get_elapsed_ms());

        // 4.3) launch optix
        printf(AC_BOLDCYAN "\tOptiX Launch [ %-3s ]..............." AC_RESET, algStr[args.alg]); fflush(stdout);
        timer.restart();
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(d_params), sizeof(Params), &state.sbt, n, 1, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop(); printf(AC_BOLDCYAN "done: %f ms\n" AC_RESET, timer.get_elapsed_ms());
    }

    void cleanup() {
        // 6) clean up
        printf("cleaning up RTX environment..........\n"); fflush(stdout);
        OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
        for (int i = 0; i < 3; ++i) {
            OPTIX_CHECK(optixProgramGroupDestroy(state.program_groups[i]));
        }
        OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
        OPTIX_CHECK(optixDeviceContextDestroy(state.context));

        CUDA_CHECK(cudaFree(d_params));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    }

};


