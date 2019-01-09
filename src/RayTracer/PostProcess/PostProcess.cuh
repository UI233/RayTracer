#pragma once
#include <helper_math.h>
#include <cuda_runtime.h>
#include "surface_functions.h"
#include "device_atomic_functions.h"
#ifndef CUDA_FUNC
#define CUDA_FUNC  __host__ __device__
#endif // !CUDA_FUNC


__device__ __forceinline__ float atomicMaxFloat(float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

inline __device__ float RGB2Y(const float3 &color)
{
	float Y;
	Y = 0.299f * color.x + 0.587f * color.y +
		0.114f * color.z;
	return Y;
}

//Incompleted
CUDA_FUNC float3 HDR(float3 color, float Ymax)
{	
	float3 Y;
	Y.x = 0.299f * color.x + 0.587f * color.y +
		0.114f * color.z;
	Y.y = -0.147f * color.x - 0.289f * color.y +
		0.435f * color.z;
	Y.z = 0.615f * color.x - 0.515f * color.y -
		0.1f * color.z;
	float a;
    if (Ymax == 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);

	Y.x = logf(Y.x) / logf(Ymax);

	float3 R;
	R.x = fminf(1.0f, fmaxf(0.0f, Y.x + 1.1398f * Y.z));
	R.y = fminf(1.0f, fmaxf(0.0f, 0.9996 * Y.x - 0.3954 * Y.y - 0.5805 * Y.z));
	R.z = fminf(1.0f, fmaxf(0.0f, 1.002 * Y.x + 2.0361 * Y.y - 0.0005 * Y.z));
	return R;
}

__global__ void HDRKernel(cudaSurfaceObject_t surface_tmp, int width_per_thread, int height_per_thread, int width_per_block, int height_per_block,  float *Ymax)
{
    int stx = blockIdx.x * width_per_block + threadIdx.x * width_per_thread;
    int sty = blockIdx.y * height_per_block + threadIdx.y * height_per_thread;

    float4 color;
    float3 tmp;
    for(int x = stx;x < stx + width_per_thread; x++)
        for (int y = sty; y < sty + height_per_thread; y++)
        {
            surf2Dread(&color, surface_tmp, x * sizeof(float4), y);
            tmp = HDR(make_float3(color.x, color.y, color.z), *Ymax);
            surf2Dwrite(make_float4(tmp.x, tmp.y, tmp.z, 1.0f), surface_tmp, x * sizeof(float4), y);
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