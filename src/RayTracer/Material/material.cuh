#pragma once
#include "../Ray/Ray.cuh"
#include "curand.h"
#include "curand_kernel.h"
//All computations happens in the local coordinate system(TBN space) where the normal vector would be (0, 1, 0)
//The computation for T vector and B vector will be posted in issue.
class Material
{
public:
    CUDA_FUNC Material() = default;
    CUDA_FUNC virtual ~Material() = default;
    CUDA_FUNC virtual float3 f(Ray &wo, Ray &wi) const = 0;
    //emited ray, incident ray, possibility for wi, the 2-D sample points between [0, 1]
    CUDA_FUNC virtual float3 sample_f(Ray &wo, Ray *wi, float *pdf, const float2 &sample) const = 0;
    CUDA_FUNC virtual float PDF(Ray &wo, Ray &wi) const = 0;
    CUDA_FUNC virtual bool isSpecular() const = 0;
};

class Lambertian : public Material
{
public:
    CUDA_FUNC Lambertian() = default;
    CUDA_FUNC Lambertian(const float3 &f) : color(f) {}
    CUDA_FUNC ~Lambertian() = default;
    CUDA_FUNC float3 f(Ray &wo, Ray &wi)const  override;
    CUDA_FUNC float3 sample_f(Ray &wo, Ray *wi, float *pdf, const float2 &) const override;
    CUDA_FUNC bool isSpecular() const override;
    CUDA_FUNC float PDF(Ray &wo, Ray &wi) const;
private:
    float3 color;
};