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
//Use Filmic ACES tone mapping
//From https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
__device__ __forceinline__ float3 HDR(float3 x)
{
    static constexpr float a = 2.51f;
    static constexpr float b = 0.03f;
    static constexpr float c = 2.43f;
    static constexpr float d = 0.59f;
    static constexpr float e = 0.14f;
    return saturate((x*(a*x + b)) / (x*(c*x + d) + e));
}

__global__ void HDRKernel(cudaSurfaceObject_t surface_w, cudaSurfaceObject_t surface_r, int width_per_thread, int height_per_thread, int width_per_block, int height_per_block, float4 *out)
{
    int stx = blockIdx.x * width_per_block + threadIdx.x * width_per_thread;
    int sty = blockIdx.y * height_per_block + threadIdx.y * height_per_thread;
    //printf("%f ", *Ymax);
    float4 color;
    float3 tmp;

    for(int x = stx;x < stx + width_per_thread; x++)
        for (int y = sty; y < sty + height_per_thread; y++)
        {
            surf2Dread(&color, surface_r, x * sizeof(float4), y);
            tmp = HDR(make_float3(color.x, color.y, color.z));
            surf2Dwrite(make_float4(tmp.x, tmp.y, tmp.z, 1.0f), surface_w, x * sizeof(float4), y);
            //out[y * WIDTH + x] = make_float4(tmp.x, tmp.y, tmp.z, 1.0f);
        }

}
//Incompleted
//
__global__ void filterKernel(cudaSurfaceObject_t surface_tmp, int width_per_thread, int height_per_thread, int width_per_block, int height_per_block, int width, int height, float4 *out)
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

            //surf2Dwrite(temp[count / 2], surface, x * sizeof(float4), y);
            out[y * width + x] = temp[count / 2];
        }
}