#define HEIGHT 900
#define WIDTH 1600
#define NUM 16
#define MAX_DEPTH 10
#define SAMPLE 4
#include "demo.h"
#include <stdio.h>
#include "vector_functions.hpp"
#include <device_functions.h>


__shared__  Camera cas;
__shared__ Light l1;
__shared__ Sphere s0;
__shared__ Sphere s1;

__device__ inline float shadowRay(float3 pos, curandState *state, int idx)
{
    float3 st = l1.sample(idx, SAMPLE, state);
    Ray shadow_ray(
        make_float3(pos.x - st.x, pos.y - st.y, pos.z - st.z),
        st
    );


    float t0 = s0.hit(shadow_ray);
    float t1 = s1.hit(shadow_ray);

    if (t0 > TMIN && t0 < 1.0f - TMIN || t1 > TMIN && t1 < 1.0f - TMIN)
        return 0.0f;
    else return 5000.0f / (5000.0f + dot(shadow_ray.dir, shadow_ray.dir));
}

__device__ float3 getColor(Ray &r, curandState *state)
{
    float t0 = s0.hit(r);
    float t1 = s1.hit(r);
    float t2 = l1.hit(r);
    float3 color = make_float3(1.0f, 1.0f, 1.0f);
    
    if (t2 > TMIN && (t2 <t1 || t1 <TMIN) && (t2 < t0 || t0 < TMIN))
    {
        return  color;
    }
    color = make_float3(0.0f, 0.0f, 0.0f);
    float t;

    if (t1 > TMIN && t0 > TMIN)
        t = min(t0, t1);
    else
        t = max(t0, t1);

    if (t < TMIN)
        return color;
    //else return make_float3(0.0f);

    float3 pos = make_float3(r.pos.x + r.dir.x * t,
        r.pos.y + r.dir.y * t,
        r.pos.z + r.dir.z * t
        );

    float per_c = 1.0f / SAMPLE / SAMPLE;
        for (int i = 0; i < SAMPLE * SAMPLE; ++i)
        {
            float pers = shadowRay(pos, state, i);
            color.x += per_c * pers;
            color.y += per_c * pers;
            color.z += per_c * pers;
        }

    return color;
}

__global__ void launch(curandState *state, float3 *in)
{
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        cas = Camera(WIDTH, HEIGHT, make_float3(0.25f, 3.0f, 20.5f));
        l1 = Light(make_float3(-10.0f, 12.0f, -3.0f), make_float3(0.0f, 5.0f, 0.0f), make_float3(0.0f, 0.0f, -5.0f));
        s0 = Sphere(make_float3(0.0f, -1006.0f, -5.0f), 1000.0f);
        s1 = Sphere(make_float3(0.0f, 2.0f, -5.0f), 8.0f);
        //printf("\nHello, world from the device!\n %d %d", blockIdx.x, blockIdx.y);
    }

    __syncthreads();

    int i =blockIdx.x *  blockDim.x + threadIdx.x, 
        j = blockIdx.y *  blockDim.y + threadIdx.y;

    int s_id = j * 200 + i;
    curand_init(312312,  s_id , 0, &state[s_id]);
    i *= 8;
    j *= 30;

    int id = j * WIDTH + i;
    Ray r;
    float3 color;

    for (int y = 0; y < 30; y++, j++, id += WIDTH)
        for (int x = 0; x < 8; x++)
        {
            color = make_float3(0.0f);
            for (int k = 0; k < SAMPLE; k++)
            {
                r = cas.generateRay(i + x + (float)curand(state) / MAXN, j + (float)curand(state) / MAXN);
                float3 tmp = getColor(r, &state[s_id]);
                color = make_float3(
                    color.x + tmp.x / SAMPLE,
                    color.y + tmp.y / SAMPLE,
                    color.z + tmp.z / SAMPLE
                );
            }
            in[id + x] = color;
        }

    
    //color = make_float3(1.0f, 0.0f, 1.0f);
}   


float3 color_buffer[WIDTH * HEIGHT + 10];
void drawToFile()
{
    FILE *out = fopen("test.ppm", "w");

    int count = 0;

    fprintf(out, "P3\n%d %d\n255\n",WIDTH, HEIGHT);
    for (int j = 0; j < HEIGHT; j++)
        for (int i = 0; i < WIDTH; i++, count++)
            fprintf(out, "%d %d %d\n", (int)(color_buffer[count].x * 255.99f), (int)(color_buffer[count].y * 255.99f), (int)(color_buffer[count].z * 255.99f));

    fclose(out);
    return;
}
int main()
{
    constexpr unsigned int block_x(20), block_y(3);
    constexpr unsigned int threads_x(10), threads_y(10);
    
    dim3 blocksNum(block_x, block_y);
    dim3 threadsPerBlock(threads_x, threads_y);

    curandState *state;
    cudaError_t cudaStatus;
    float3 *buffer;
    cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaMalloc((void **)&state, sizeof(curandState) * 200 * 30);
    cudaStatus = cudaMalloc((void**)&buffer, sizeof(float3) * WIDTH * HEIGHT);
    launch <<< blocksNum, threadsPerBlock >>> (state, buffer);

    cudaStatus = cudaDeviceSynchronize();
    
    cudaStatus = cudaMemcpy(color_buffer, buffer, sizeof(float3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    cudaFree(state);
    cudaFree(buffer);
    drawToFile();
    return 0;
}