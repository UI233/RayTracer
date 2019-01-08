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
CUDA_FUNC float3 filter(float3 color, float Ymax)
{	
	float3 Y;
	Y.x = 0.299f * color.x + 0.587f * color.y +
		0.114f * color.z;
	Y.y = -0.147f * color.x - 0.289f * color.y +
		0.435f * color.z;
	Y.z = 0.615f * color.x - 0.515f * color.y -
		0.1f * color.z;
	float a;

	Y.x = logf(Y.x) / logf(Ymax);

	float3 R;
	R.x = fminf(1.0f, fmaxf(0.0f, Y.x + 1.1398f * Y.z));
	R.y = fminf(1.0f, fmaxf(0.0f, 0.9996 * Y.x - 0.3954 * Y.y - 0.5805 * Y.z));
	R.y = fminf(1.0f, fmaxf(0.0f, 1.002 * Y.x + 2.0361 * Y.y - 0.0005 * Y.z));
	return R;
}