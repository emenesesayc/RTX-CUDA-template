#pragma once

template <typename IntegerType>
__device__ __host__ IntegerType roundUp(IntegerType x, IntegerType y) {
    return ((x + y - 1) / y) * y;
}

void launch_cuda(int n, float a, float *x, float *y);

std::string loadPtx(std::string filename) {
    std::ifstream ptx_in(filename);
    return std::string((std::istreambuf_iterator<char>(ptx_in)), std::istreambuf_iterator<char>());
}

struct GASstate {
    OptixDeviceContext context = 0;

    size_t temp_buffer_size = 0;
    CUdeviceptr d_temp_buffer = 0;
    CUdeviceptr d_temp_aabb = 0;

    unsigned int aabb_flags = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput aabb_input = {};
    OptixTraversableHandle gas_handle;
    //CUdeviceptr d_gas_output_buffer;
    CUdeviceptr d_gas_original_output_buffer; //temporary global value
    CUdeviceptr d_gas_compact_output_buffer;  //temporary global value
    CUdeviceptr d_gas_final_output_buffer;
    size_t gas_output_buffer_size = 0;

    OptixAccelBuildOptions accel_build_options = {};
    OptixAccelBuildOptions accel_update_options = {};
    OptixAccelEmitDesc emitProperty;
    bool need_compaction;
    bool buffer_saving_mode;

    OptixModule ptx_module = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline = 0;

    OptixProgramGroup program_groups[3];
    OptixShaderBindingTable sbt = {};
};

void createOptixContext(GASstate &state) {
  CUDA_CHECK( cudaFree(0) ); // creates a CUDA context if there isn't already one
  OPTIX_CHECK(optixInit() ); // loads the optix library and populates the function table

  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &optixLogCallback;
  options.logCallbackLevel = 4;
  //options.logCallbackLevel = 1;

  OptixDeviceContext optix_context = nullptr;
  optixDeviceContextCreate(0, // use current CUDA context
                           &options, &optix_context);

  state.context = optix_context;
}

// load ptx and create module
void loadAppModule(GASstate &state) {
  std::string ptx = loadPtx(BUILD_DIR "/ptx/rtx_kernels.ptx");

  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  //module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  //module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  state.pipeline_compile_options.usesMotionBlur = false;
  state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  state.pipeline_compile_options.numPayloadValues = 2;
  state.pipeline_compile_options.numAttributeValues = 2; // 2 is the minimum
  state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    //LOG_SIZE=0;
  OPTIX_CHECK(optixModuleCreate(state.context, &module_compile_options,
                                       &state.pipeline_compile_options, ptx.c_str(),
                                       ptx.size(), LOG, &LOG_SIZE, &state.ptx_module));
  print_log("create module", LOG, LOG_SIZE);
}

void createGroupsClosestHit(GASstate &state) {
  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc prog_group_desc[3] = {};

  // raygen
  prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  prog_group_desc[0].raygen.module = state.ptx_module;
  prog_group_desc[0].raygen.entryFunctionName = "__raygen__rtx__nn";

  // we need to create these but the entryFunctionNames are null
  prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  prog_group_desc[1].miss.module = state.ptx_module;
  prog_group_desc[1].miss.entryFunctionName = "__miss__rtx";
  //prog_group_desc[1].miss.entryFunctionName = nullptr;



  // closest hit
  prog_group_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

  //prog_group_desc[2].hitgroup.moduleCH = state.ptx_module;
  //prog_group_desc[2].hitgroup.entryFunctionNameCH = "__closesthit__rtx";
  //prog_group_desc[2].hitgroup.moduleAH = state.ptx_module;
  //prog_group_desc[2].hitgroup.entryFunctionNameAH = "__anyhit__rtx";

  prog_group_desc[2].hitgroup.moduleCH = nullptr;
  prog_group_desc[2].hitgroup.entryFunctionNameCH = nullptr;
  prog_group_desc[2].hitgroup.moduleAH = nullptr;
  prog_group_desc[2].hitgroup.entryFunctionNameAH = nullptr;


  prog_group_desc[2].hitgroup.moduleIS = state.ptx_module;
  prog_group_desc[2].hitgroup.entryFunctionNameIS = "__intersection__radius";


  OPTIX_CHECK(optixProgramGroupCreate(state.context, prog_group_desc, 3, &program_group_options, LOG, &LOG_SIZE, state.program_groups));
  print_log("program group create", LOG, LOG_SIZE);
}



void createPipeline(GASstate &state) {
  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = 1;
  //pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  OPTIX_CHECK(optixPipelineCreate(  state.context,
                                    &state.pipeline_compile_options,
                                    &pipeline_link_options,
                                    state.program_groups,
                                    3, LOG, &LOG_SIZE,
                                    &state.pipeline));
  //print_log("create pipeline", LOG, LOG_SIZE);
}


void populateSBT(GASstate &state) {
  char *device_records;
  CUDA_CHECK(cudaMalloc(&device_records, 3 * OPTIX_SBT_RECORD_HEADER_SIZE));

  char *raygen_record = device_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *miss_record = device_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *hitgroup_record = device_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE;

  char sbt_records[3 * OPTIX_SBT_RECORD_HEADER_SIZE];
  OPTIX_CHECK(optixSbtRecordPackHeader( state.program_groups[0], sbt_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader( state.program_groups[1], sbt_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader( state.program_groups[2], sbt_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE));

  CUDA_CHECK(cudaMemcpy(device_records, sbt_records, 3 * OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice));

  state.sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_record);

  state.sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record);
  state.sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  state.sbt.missRecordCount = 1;

  state.sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_record);
  state.sbt.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  state.sbt.hitgroupRecordCount = 1;
}

void print_compaction_rate(size_t compact_bytes, size_t original_bytes)
{
    printf("\n");
    if(compact_bytes < original_bytes){
        float compaction_rate = ((float)compact_bytes / original_bytes) * 100.f;
        printf(AC_BOLDRED "COMPACTION RATE: %f%% (%uK instead of %uK, win: %uK)" AC_RESET, 
                            compaction_rate,
                            compact_bytes / 1024,
                            original_bytes / 1024,
                            (original_bytes - compact_bytes) / 1024);
    } else {
        printf(AC_BOLDRED "NO COMPACTION: (compaction: %uK, original: %uK)" AC_RESET, 
                            compact_bytes / 1024,
                            original_bytes / 1024);
    }
    printf("\n");
}



void buildAS_classic(GASstate &state, int n, OptixAabb* d_aabb_array, bool compact_mode) 
{
    //Timer timer;
    //timer.restart();

    state.d_temp_aabb = reinterpret_cast<CUdeviceptr>(d_aabb_array);
    state.aabb_input.type                                 = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    state.aabb_input.customPrimitiveArray.aabbBuffers     = &state.d_temp_aabb;
    state.aabb_input.customPrimitiveArray.numPrimitives   = n;
    state.aabb_input.customPrimitiveArray.flags           = &state.aabb_flags;
    state.aabb_input.customPrimitiveArray.numSbtRecords   = 1;

    //create build & update options
    state.accel_build_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    if(compact_mode) {
        state.accel_build_options.buildFlags =  OPTIX_BUILD_FLAG_ALLOW_COMPACTION | 
                                            OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    } else { state.accel_build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; }
    //state.accel_build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; 
  

    state.accel_update_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    if(compact_mode){
        state.accel_update_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | 
                                            OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    } else { state.accel_update_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; }
    //state.accel_update_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;  

  
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(state.context, &state.accel_build_options, &state.aabb_input, 1, &gas_buffer_sizes) );

    state.temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    size_t outputSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_gas_original_output_buffer), outputSizeOffset + 8) );
    state.gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;
  
    if(compact_mode)  state.emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    else              state.emitProperty.type = OPTIX_PROPERTY_TYPE_AABBS;
    state.emitProperty.result = (CUdeviceptr)((char *)state.d_gas_original_output_buffer + outputSizeOffset);

    //timer.stop();
    //printf("init done: %f ms\n",timer.get_elapsed_ms());

    //timer.restart();
    OPTIX_CHECK( optixAccelBuild(
        state.context,
        0, 
        &state.accel_build_options, 
        &state.aabb_input,
        1,
        state.d_temp_buffer,
        state.temp_buffer_size,
	state.d_gas_original_output_buffer, //state.d_gas_output_buffer,
	state.gas_output_buffer_size,
        &state.gas_handle,
        &state.emitProperty, 
        1) 
    );

    //timer.stop();
    //printf("building done: %f ms\n",timer.get_elapsed_ms());

    //timer.restart();
    //if(!buffer_saving_mode){ 
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(state.d_temp_buffer)) ); 
    //}

    state.need_compaction = false;
    if(compact_mode)
    {
        size_t compacted_gas_size;
        CUDA_CHECK( cudaMemcpy(&compacted_gas_size, (void *)state.emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost) );
        //printf("\ncompacted size: %uK\n", compacted_gas_size / 1024);
    
        print_compaction_rate(compacted_gas_size, gas_buffer_sizes.outputSizeInBytes);
        if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) 
        { 
            state.need_compaction = true;
            CUDA_CHECK( cudaFree(reinterpret_cast<void*>(state.d_gas_compact_output_buffer)) );
            CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&state.d_gas_compact_output_buffer), compacted_gas_size + 8) );

            //timer.stop();
            //printf("init compact done: %f ms\n",timer.get_elapsed_ms());


            //timer.restart();
            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( state.context,
                                    0,
                                    state.gas_handle,
                                    state.d_gas_compact_output_buffer, //state.d_gas_output_buffer,
                                    compacted_gas_size, //state.gas_output_buffer_size,
                                    &state.gas_handle));

            //timer.stop();
            //printf("compaction done: %f ms\n",timer.get_elapsed_ms());


            state.gas_output_buffer_size = compacted_gas_size;

            //if(!buffer_saving_mode){ 
            CUDA_CHECK( cudaFree(reinterpret_cast<void*>(state.d_gas_original_output_buffer)) ); 
            //}
        } 
    }
    //timer.stop();
    //state.buffer_saving_mode = buffer_saving_mode;
}


void init_buildAS_optimized(GASstate &state, int n, OptixAabb* d_aabb_array, bool compact_mode)
{
    state.d_temp_aabb = reinterpret_cast<CUdeviceptr>(d_aabb_array);
    state.aabb_input.type                                 = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    state.aabb_input.customPrimitiveArray.aabbBuffers     = &state.d_temp_aabb;
    state.aabb_input.customPrimitiveArray.numPrimitives   = n;
    state.aabb_input.customPrimitiveArray.flags           = &state.aabb_flags;
    state.aabb_input.customPrimitiveArray.numSbtRecords   = 1;

    //create build & update options
    state.accel_build_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    if(compact_mode) {
        state.accel_build_options.buildFlags =  OPTIX_BUILD_FLAG_ALLOW_COMPACTION | 
                                            OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    } else { state.accel_build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; }
    //state.accel_build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; 
  

    state.accel_update_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    if(compact_mode){
        state.accel_update_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | 
                                            OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    } else { state.accel_update_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; }
    //state.accel_update_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;  

  
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(state.context, &state.accel_build_options, &state.aabb_input, 1, &gas_buffer_sizes) );

    state.temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    size_t outputSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_gas_original_output_buffer), outputSizeOffset + 8) );
    state.gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;

    //already create the compact buffer of the size of a normal buffer
    if(compact_mode){
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_gas_compact_output_buffer), outputSizeOffset) );
    }
  
    if(compact_mode)  state.emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    else              state.emitProperty.type = OPTIX_PROPERTY_TYPE_AABBS;
    state.emitProperty.result = (CUdeviceptr)((char *)state.d_gas_original_output_buffer + outputSizeOffset);


}


void buildAS_optimized(GASstate &state, bool compact_mode)
{
    OPTIX_CHECK( optixAccelBuild(
        state.context,
        0, 
        &state.accel_build_options, 
        &state.aabb_input,
        1,
        state.d_temp_buffer,
        state.temp_buffer_size,
	state.d_gas_original_output_buffer,
	state.gas_output_buffer_size,
        &state.gas_handle,
        &state.emitProperty, 
        1) 
    );


    state.need_compaction = false;
    if(compact_mode)
    {
        size_t compacted_gas_size;
        CUDA_CHECK( cudaMemcpy(&compacted_gas_size, (void *)state.emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost) );
        //printf("\ncompacted size: %uK\n", compacted_gas_size / 1024);
    
        print_compaction_rate(compacted_gas_size, state.gas_output_buffer_size);
        if (compacted_gas_size < state.gas_output_buffer_size) 
        { 
            state.need_compaction = true;
            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( state.context,
                                    0,
                                    state.gas_handle,
                                    state.d_gas_compact_output_buffer,
                                    compacted_gas_size,
                                    &state.gas_handle));

            state.gas_output_buffer_size = compacted_gas_size;
        } 
    }

}




void updateASFromDevice(GASstate &state, bool buffer_saving_mode)
{   
    if(buffer_saving_mode) {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_temp_buffer), state.temp_buffer_size));
    }

    if(state.need_compaction){
        OPTIX_CHECK( optixAccelBuild(
            state.context,
            0, 
            &state.accel_update_options,//&gas_accel_options, 
            &state.aabb_input,
            1,
            state.d_temp_buffer,
            state.temp_buffer_size,
	    state.d_gas_compact_output_buffer, //state.d_gas_output_buffer,
	    state.gas_output_buffer_size,
            &state.gas_handle,
            nullptr, 
            0) 
        );
    } else {
        OPTIX_CHECK( optixAccelBuild(
            state.context,
            0, 
            &state.accel_update_options,//&gas_accel_options, 
            &state.aabb_input,
            1,
            state.d_temp_buffer,
            state.temp_buffer_size,
	    state.d_gas_original_output_buffer, //state.d_gas_output_buffer,
	    state.gas_output_buffer_size,
            &state.gas_handle,
            nullptr, 
            0) 
        );
    }

    if(!buffer_saving_mode){ 
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(state.d_temp_buffer)) );
    }

}
