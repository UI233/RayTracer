#pragma once
#include "helper_math.h"

#ifndef CUDA_FUNC
#define CUDA_FUNC __host__ __device__
#endif

class Ray
{
public:
    CUDA_FUNC Ray() = default;
    CUDA_FUNC ~Ray() = default;
    CUDA_FUNC Ray(const Ray &r) = default;
    CUDA_FUNC Ray(const float3 &o, const float3 &r);
    CUDA_FUNC inline float3 getPos(float t) const;
    CUDA_FUNC inline float3 getDir() const;
    CUDA_FUNC inline float3 getOrigin() const;
private:
    float3 origin;
    float3 direction;
    //Make the class aligned to 32 byte
    float2 dummy;
};