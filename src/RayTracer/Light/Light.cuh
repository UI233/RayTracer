#pragma once
#include "../Model/Model.cuh"
#include "thrust/device_vector.h"

class Light 
{
public:
    CUDA_FUNC virtual ~Light() = default;
    CUDA_FUNC Light() = default;
    CUDA_FUNC virtual float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const = 0;
};

class PointLight : public Light
{
public:
    CUDA_FUNC  PointLight() = default;
    CUDA_FUNC ~PointLight() = default;
    CUDA_FUNC PointLight(const float3 &position, const float3 &color);
    CUDA_FUNC float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;
private:
    float3 pos, illum;
};

class DirectionalLight : public Light
{
public:
    CUDA_FUNC DirectionalLight() = default;
    CUDA_FUNC ~DirectionalLight() = default;
    CUDA_FUNC DirectionalLight(const float3 &direction, const float3 &color);
    CUDA_FUNC float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;

private:
    float3 dir, illum;
};
