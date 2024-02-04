#pragma once


__host__ __device__ float distance2(Particle* a, Particle* b) {
    return distance2(a->pos, b->pos);
}
__host__ __device__ float distance(Particle* a, Particle* b) {
    return sqrt(distance2(a->pos, b->pos));
}


// print
#define PRINT_LIMIT 32
__global__ void kernel_print_array_dev(int n, float *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %f\n", tid, i, darray[i]);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_vertices_dev(int ntris, float3 *v){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<ntris && i<PRINT_LIMIT; ++i){
        printf("tid %i --> vertex[%i] = (%f, %f, %f)\n", tid, 3*i+0, v[3*i+0].x, v[3*i+0].y, v[3*i+0].z);
        printf("tid %i --> vertex[%i] = (%f, %f, %f)\n", tid, 3*i+1, v[3*i+1].x, v[3*i+1].y, v[3*i+1].z);
        printf("tid %i --> vertex[%i] = (%f, %f, %f)\n", tid, 3*i+2, v[3*i+2].x, v[3*i+2].y, v[3*i+2].z);
        printf("\n");
    }
    if(i < ntris){
        printf("...\n");
    }
}

void print_array_dev(int n, float *darray){
    printf("Printing random array:\n");
    kernel_print_array_dev<<<1,1>>>(n, darray);
    cudaDeviceSynchronize();
}

void print_vertices_dev(int ntris, float3 *devVertices){
    printf("Printing vertices:\n");
    kernel_print_vertices_dev<<<1,1>>>(ntris, devVertices);
    cudaDeviceSynchronize();
}

void print_int_array(const char* name, int* array, int n){
    if(n==0) { printf("%s[%d] = ~\n", name, n); return; }
    printf("%s[%d] = {", name, n);
    for(int i=0; i<min(n-1, PRINT_LIMIT); i++){
        printf("%d, ", array[i]);
    }
    printf("%d", array[n-1]);
    if(n>PRINT_LIMIT) printf(", ... }\n");
    else printf("}\n");
}

void print_particle(Particle* p){
    printf("id: %d, pos(%.4f,%.4f,%.4f), size: %.2f\n", p->id, p->pos.x, p->pos.y, p->pos.z, p->size);
}

void print_particles_array(Particle* array, int n){
    printf("particle array[%d] = {\n", n);
    for(int i=0; i<min(n, PRINT_LIMIT); i++){
        printf("\t");
        print_particle(&array[i]);
    }
    if(n>PRINT_LIMIT) printf("\t...\n");
    printf("}\n");
}

void print_log(const char* message, char* log, size_t n){
    log[n] = '\0';
    if(n < 2 || strlen(log) < 1) return;
    printf(AC_CYAN "LOG[%d]: " AC_BLUE "%s" AC_BLUE ": %s" AC_RESET, n, message, log);
    //printf("%s\n",log);
    /*for(int i=0;i<n;i++){
        printf("%c(%d)", log[i], (int)log[i]);
    }
    printf("\n");*/
}

void print_all_neighbors(int* neighbors, int* nneigh, int n, Particle* p){
    char strid[16];
    printf("List of neighbors of %d particles (NMAX=%d) :\n", n, NMAX);
    for(int i=0; i<min(n, PRINT_LIMIT); i++){
        sprintf(strid, "id:%d(%.1f,%.1f,%.1f)", i, p[i].pos.x, p[i].pos.y, p[i].pos.z);
        printf("\t");
        print_int_array(strid, &neighbors[i*NMAX], nneigh[i]);
    }
    printf("\t...\n");
}

// Geometry generation
__global__ void kernel_gen_vertices(int ntris, float *array, float3 *vertices){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < ntris){
        int k = 3*tid;
        float xval = array[tid];
        vertices[k+0] = make_float3(xval,  0,  1);
        vertices[k+1] = make_float3(xval,  1, -1);
        vertices[k+2] = make_float3(xval, -1, -1);
    }
}

__global__ void kernel_gen_triangles(int ntris, float *array, uint3 *triangles){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < ntris){
        int k = 3*tid;
        triangles[tid] = make_uint3(k, k+1, k+2);
    }
}

float3* gen_vertices_dev(int ntris, float *darray){
    // vertices data
    float3 *devVertices;
    cudaMalloc(&devVertices, sizeof(float3)*3*ntris);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((ntris+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_vertices<<<grid, block>>>(ntris, darray, devVertices);
    cudaDeviceSynchronize();
    return devVertices;
}


uint3* gen_triangles_dev(int ntris, float *darray){
    // data array
    uint3 *devTriangles;
    cudaMalloc(&devTriangles, sizeof(uint3)*ntris);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((ntris+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_triangles<<<grid, block>>>(ntris, darray, devTriangles);
    cudaDeviceSynchronize();
    return devTriangles;
}


void cpuprint_array(int np, float *dp){
    float *hp = new float[np];
    cudaMemcpy(hp, dp, sizeof(float)*np, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0; i<np; ++i){
        printf("array [%i] = %f\n", i, hp[i]);
    }
}


bool compare_neighbors_result(int* neighbors_a, int* nneigh_a, int* neighbors_b, int* nneigh_b, int n)
{
    //comparing neighbors's numbers
    for(int i=0; i<n; i++){
        if(nneigh_a[i] != nneigh_b[i]) {
            printf(AC_BOLDRED "DIFFERENT NEIGHBORS NUMBER FOR PARTICLE %d: %d != %d.\n" AC_RESET, i, nneigh_a[i], nneigh_b[i]);
            return false;
        }
    }

    //comparing neighbors itself
    for(int pi=0; pi<n; pi++){
       for(int i=0; i<NMAX; i++){
            bool is_same = false;
            int id_i = neighbors_a[pi*NMAX + i];
            for(int j=0; j<NMAX; j++){
                if(id_i == neighbors_a[pi*NMAX + j]){
                    is_same = true;
                }
            }
            if(!is_same){
                printf(AC_BOLDRED "DIFFERENT NEIGHBORS NUMBER FOR PARTICLE %d: NEIGHBOR %d IS NOT FOUND.\n" AC_RESET, pi, id_i);
                return false;
            }
        }
    }
    printf(AC_BOLDGREEN "COMPARISON SUCCEEDED !\n" AC_RESET);
    return true;
}

__global__ void kernel_update_aabbs(Particle* d_particles, OptixAabb* d_aabb_array, int n, float radius){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n) {
        OptixAabb aabb = d_aabb_array[tid];
        float3 pos = d_particles[tid].pos;
        aabb.minX = pos.x - radius;
        aabb.minY = pos.y - radius;
        aabb.minZ = pos.z - radius;
        aabb.maxX = pos.x + radius;
        aabb.maxY = pos.y + radius;
        aabb.maxZ = pos.z + radius;
        d_aabb_array[tid] = aabb;
        tid += gridDim.x*blockDim.x;
    }
}

void update_aabb_from_particles(Particle* d_particles, OptixAabb* d_aabb_array, int n, float radius)
{
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1);
    kernel_update_aabbs<<<grid, block>>>(d_particles, d_aabb_array, n, radius);
    CUDA_CHECK( cudaDeviceSynchronize() );
}


__global__ void kernel_init_nneigh(int* d_nneigh, int n){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while(tid < n) {
        d_nneigh[tid] = 0;
        tid += gridDim.x*blockDim.x;
    }
}

void init_nneigh(int* d_nneigh, int n)
{
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1);
    kernel_init_nneigh<<<grid, block>>>(d_nneigh, n);
    CUDA_CHECK( cudaDeviceSynchronize() );
}
