#pragma once
#include "../Model/Model.cuh"
#include "thrust/device_vector.h"
namespace light
{
    enum LIGHT_TYPE
    {
        POINT_LIGHT,
        DIR_LIGHT,
        TRIANGLE_LIGHT,
        TYPE_NUM
    };
}
class Light 
{
public:
    CUDA_FUNC virtual ~Light() = default;
    CUDA_FUNC Light() = default;
    //CUDA_FUNC virtual float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const = 0;
    CUDA_FUNC virtual float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const = 0;
};

class PointLight : public Light
{
public:
    CUDA_FUNC  PointLight() = default;
    CUDA_FUNC ~PointLight() = default;
    //CUDA_FUNC virtual float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const override;
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
    //CUDA_FUNC float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const override;
    CUDA_FUNC float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;

private:
    float3 dir, illum;
};

class TriangleLight : public Light
{
public:
    CUDA_FUNC TriangleLight() = default;
    CUDA_FUNC ~TriangleLight() = default;
    CUDA_FUNC TriangleLight(const Triangle& triangle, const float3& light_color) : tri(triangle), color(light_color) {}
    CUDA_FUNC float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;
    CUDA_FUNC bool hit(Ray &r, IntersectRecord &rec);
    //CUDA_FUNC float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const override;

private:
    Triangle tri;
    float3 color;
};
