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
	float radius_searching;
        int NMAX;
};
