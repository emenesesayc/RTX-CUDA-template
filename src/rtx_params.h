#pragma once

struct Particle {
	int id;
	float size;
	float3 pos;
};

struct Params {
  	OptixTraversableHandle handle;
  	int* d_neighbors;
        int* d_nneigh;
        Particle* d_particles;
        OptixAabb* d_aabb_array;
	float r;
        int NMAX;
};

static __forceinline__ __device__ __host__ float distance2(float3 a, float3 b){
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return dx*dx + dy*dy + dz*dz;
}

static __forceinline__ __device__ __host__ float distance(float3 a, float3 b){
    return sqrt(distance2(a, b));
}
