#pragma once
#ifndef RAY_H
#define RAY_H
#ifndef CUDA_FUNC
#define CUDA_FUNC __host__ __device__
#endif
#include "helper_math.h"
class Ray
{
public:
    CUDA_FUNC Ray() = default;
    CUDA_FUNC ~Ray() = default;
    CUDA_FUNC Ray(const Ray &r) = default;
    CUDA_FUNC Ray(const float3 &o, const float3 &r);
    CUDA_FUNC float3 getPos(float t) const;
    CUDA_FUNC float3 getDir() const;
    CUDA_FUNC float3 getOrigin() const;
private:
    float3 origin;
    float3 direction;
    //Make the class aligned to 32 byte
    float2 dummy;
};

class IntersectRecord
{
public:
    CUDA_FUNC IntersectRecord() = default;
    CUDA_FUNC ~IntersectRecord() = default;
    CUDA_FUNC IntersectRecord(const float3 &p, const float3 &n, const Ray &r, const float &dis);

    float3 pos, normal;
    float t;
    Ray wo;
};

#endif