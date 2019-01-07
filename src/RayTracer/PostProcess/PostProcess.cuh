#pragma once
#include <helper_math.h>
#include <cuda_runtime.h>
#include "surface_functions.h"
#ifndef CUDA_FUNC
#define CUDA_FUNC  __host__ __device__
#endif // !CUDA_FUNC

inline __device__ float3 RGB2Y(const float3 &color)
{
	float3 Y;
	Y.x = 0.299f * color.x + 0.587f * color.y +
		0.114f * color.z;
	Y.y = -0.147f * color.x - 0.289f * color.y +
		0.435f * color.z;
	Y.z = 0.615f * color.x - 0.515f * color.y -
		0.1f * color.z;
	if (Y.x > 255.0f) Y.x = 255.0f;
	if (Y.x < 0.0f)   Y.x = 0.0f;
	return Y;
}

inline CUDA_FUNC float3 HDR(const float3 & color)
{
    return make_float3(0.0f, 0.0f, 0.0f);
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
	Y.x = float(int(255 * logf(Y.x) / logf(Ymax)));

	float3 R;
	R.x = fmin(255.0f, fmax(0.0f, Y.x[i] + 1.1398f * Y.z[i]));
	R.y = fmin(255.0f, fmax(0.0f, 0.9996 * Y.x[i] - 0.3954 * Y.y[i] - 0.5805 * Y.z[i]));
	R.y = fmin(255.0f, fmax(0.0f, 1.002 * Y.x[i] + 2.0361 * Y.y[i] - 0.0005 * Y.z[i]));
	Return R;
}