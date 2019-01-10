#pragma once
#include <helper_math.h>
#include <cuda_runtime.h>
#include "surface_functions.h"
#include "device_atomic_functions.h"

#ifndef CUDA_FUNC
#define CUDA_FUNC  __host__ __device__
#endif // !CUDA_FUNC

CUDA_FUNC float3 saturate(float3 color)
{
    return make_float3(
        fminf(1.0f, fmaxf(0.0f, color.x)),
        fminf(1.0f, fmaxf(0.0f, color.y)),
        fminf(1.0f, fmaxf(0.0f, color.z))
    );
}

__device__ __forceinline__ float atomicMaxFloat(float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}
//Use ACES tone mapping
//Inspired by https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
__device__ float3 HDR(float3 color)
{
    auto RRT_ODF_fit = [](float3 v) {
        return (v * (v + make_float3(0.0245786f, 0.0245786f, 0.0245786f) - make_float3(0.000090537f, 0.000090537f, 0.000090537f)) / 
            (v * (v * 0.983729f + 0.4329510f) + 0.238081f));
    };

    color = make_float3(
        0.59719f * color.x + 0.35458f * color.y + 0.04823f * color.z,
        0.07600f * color.x + 0.90834f * color.y + 0.01566f * color.z,
        0.02840f * color.x + 0.13383f * color.y + 0.83777f * color.z
    );

    color = RRT_ODF_fit(color);

    color = make_float3(
        1.60475 * color.x + -0.53108 * color.y + -0.07367 * color.z,
        -0.10208 * color.x + 1.10813 * color.y + -0.00605 * color.z,
        -0.00327 * color.x + -0.07276 * color.y + 1.07602 * color.z
        );

    return saturate(color);
}

__global__ void HDRKernel(cudaSurfaceObject_t surface, int width_per_thread, int height_per_thread, int width_per_block, int height_per_block)
{
    int stx = blockIdx.x * width_per_block + threadIdx.x * width_per_thread;
    int sty = blockIdx.y * height_per_block + threadIdx.y * height_per_thread;
    //printf("%f ", *Ymax);
    float4 color;
    float3 tmp;
    for(int x = stx;x < stx + width_per_thread; x++)
        for (int y = sty; y < sty + height_per_thread; y++)
        {
            surf2Dread(&color, surface, x * sizeof(float4), y);
            tmp = HDR(make_float3(color.x, color.y, color.z));
            surf2Dwrite(make_float4(tmp.x, tmp.y, tmp.z, 1.0f), surface, x * sizeof(float4), y);
        }

}
//Incompleted
//
__global__ void filterKernel(cudaSurfaceObject_t surface, cudaSurfaceObject_t surface_tmp, int width_per_thread, int height_per_thread, int width_per_block, int height_per_block, int width, int height)
{
    int stx = blockIdx.x * width_per_block + threadIdx.x * width_per_thread;
    int sty = blockIdx.y * height_per_block + threadIdx.y * height_per_thread;

    float4 color;
    float3 tmp;
    float3 res;
	float4 temp[10];
	float ty[10];
	float swaptf;
	float4 swaptf4;
	int count;
    for (int x = stx; x < stx + width_per_thread; x++)
        for (int y = sty; y < sty + height_per_thread; y++)
        {
            res = make_float3(0.0f, 0.0f, 0.0f);
            //3px x 3px window
			count = 0;
            for(int i = x - 1; i <= x + 1; i++)
                for (int j = y - 1; j <= y + 1; j++)
                {	
					if (i >= 0 && i < width && j >= 0 && j < height) {
						surf2Dread(&color, surface_tmp, i * sizeof(float4), j);
						temp[count] = color;
						ty[count++]= 0.299f * color.x + 0.587f * color.y +
							0.114f * color.z;
					}

                }
			for(int i=0;i<count;i++)
				for (int j = 0; j < i; j++) {
					if (ty[i] > ty[j]) {
						swaptf = ty[i];
						swaptf4 = temp[i];
						ty[i] = ty[j];
						temp[i] = temp[j];
						ty[j] = swaptf;
						temp[j] = swaptf4;
					}
				}

            surf2Dwrite(temp[count/2] , surface, x * sizeof(float4), y);
        }
}