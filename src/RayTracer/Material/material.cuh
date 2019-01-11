#pragma once
#include "../Ray/Ray.cuh"
#include "curand.h"
#include "curand_kernel.h"
//All computations happens in the local coordinate system(TBN space) where the normal vector would be (0, 1, 0)
//The computation for T vector and B vector will be posted in issue.
class BRDF
{
public:
    CUDA_FUNC BRDF() = default;
    CUDA_FUNC virtual ~BRDF() = default;
    CUDA_FUNC virtual float3 f(const float3 &wo, const float3 &wi, float3 color = make_float3(-1.0f, -1.0f, -1.0f)) const = 0;
    //emited ray, incident ray, possibility for wi, the 2-D sample points between [0, 1]
    CUDA_FUNC virtual float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample, float3 color = make_float3(-1.0f, -1.0f, -1.0f)) const = 0;
    CUDA_FUNC virtual float PDF(const float3 &wo, const float3 &wi) const = 0;
    CUDA_FUNC virtual bool isSpecular() const = 0;
};

class Lambertian : public BRDF
{
public:
    CUDA_FUNC Lambertian() = default;
    CUDA_FUNC Lambertian(const float3 &f) : color(f) {}
    CUDA_FUNC ~Lambertian() = default;
    CUDA_FUNC float3 f(const float3 &wo, const float3 &wi, float3 color = make_float3(-1.0f, -1.0f, -1.0f))const  override;
    CUDA_FUNC float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &, float3 color = make_float3(-1.0f, -1.0f, -1.0f)) const override;
    CUDA_FUNC bool isSpecular() const override;
    CUDA_FUNC float PDF(const float3 &wo, const float3 &wi) const;
private:
    float3 color;
};
//
//class GGX : public BRDF
//{
//public:
//    CUDA_FUNC GGX() = default;
//    CUDA_FUNC GGX(); // with parameter
//    CUDA_FUNC ~GGX();
//private:
//};