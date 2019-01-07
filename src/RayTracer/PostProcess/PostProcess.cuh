#pragma once
#include <helper_math.h>
#include <cuda_runtime.h>
#include "surface_functions.h"
#ifndef CUDA_FUNC
#define CUDA_FUNC  __host__ __device__
#endif // !CUDA_FUNC

inline __device__ float3 RGB2Y(const float3 &color)
{

}

inline CUDA_FUNC float3 HDR(const float3 & color)
{
    return make_float3(0.0f, 0.0f, 0.0f);
}

//Incompleted
CUDA_FUNC float3 filter(float3 color, float Ymax)
{

}