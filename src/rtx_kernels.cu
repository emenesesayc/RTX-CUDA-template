#include <optix.h>
#include <math.h>
#include <cuda_runtime.h>
#include "rtx_params.h"


extern "C" static __constant__ Params params;



extern "C" __global__ void __raygen__rtx__nn() {
    const uint3 idx = optixGetLaunchIndex();

    OptixVisibilityMask visibilityMask = 1;
    float3 ray_origin = params.d_particles[idx.x].pos;
    float3 ray_direction = make_float3(0,1,0);  //arbitrary
    unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    unsigned int SBToffset = 0;
    unsigned int SBTstride = 0;
    unsigned int missSBTindex = 0;
    optixTrace(params.handle, ray_origin, ray_direction, 0, 1e-16f, 0,
        visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex);
}


extern "C" __global__ void __intersection__radius() {
    uint ray_id = optixGetLaunchIndex().x;
    int obj_id = optixGetPrimitiveIndex();
    int NMAX = params.NMAX;

    if (obj_id != ray_id) {
        Particle* particles = params.d_particles;
        float3 pos_ray = particles[ray_id].pos;
        float3 pos_obj = particles[obj_id].pos;
        float r = params.r;
        float d2 = distance2(pos_ray, pos_obj);

        if (d2 < r*r) {
            int* neighbors = params.d_neighbors;
            int nneigh = params.d_nneigh[ray_id];
            params.d_neighbors[ ray_id * NMAX + nneigh] = obj_id;
            params.d_nneigh[ray_id] = nneigh + 1;
        }
    }
}



extern "C" __global__ void  __closesthit__rtx() {
    /* nothing for the moment */
}

extern "C" __global__ void  __anyhit__rtx() {
    /* nothing for the moment */
}

extern "C" __global__ void  __miss__rtx() {
    /* nothing for the moment */
}


