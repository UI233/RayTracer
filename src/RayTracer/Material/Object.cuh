#pragma once
#include "material.cuh"

namespace material
{
    enum MATERIAL_TYPE
    {
        LAMBERTIAN,
        FRESNEL,
        GGX,
        MATERIAL_NUM
    };
}

class Material
{
public:
    ~Material() {
        if (brdfs != nullptr)
            free(brdfs);
    };
    Material() :brdfs(nullptr) {};
    Material(float3 N, float3 T) : normal(T), tangent(T) {};
    CUDA_FUNC virtual float3 f(const float3 &wo, const float3 &wi) const;
    //emited ray, incident ray, possibility for wi, the 2-D sample points between [0, 1]
    CUDA_FUNC  float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample) const;
    CUDA_FUNC float PDF(const float3 &wo, const float3 &wi) const;
    CUDA_FUNC float3 world2Local(const const float3 &) const;
    CUDA_FUNC float3 local2World(const const float3 &) const;
    CUDA_FUNC bool isSpecular() const
    {
        return m_type == material::FRESNEL;
    }
    CUDA_FUNC bool setUpNormal(float3 N, float3 T)
    {
        normal = N;
        tangent = T;
    }
protected:
    float3 normal;
    float3 tangent;
    BRDF *brdfs;
    material::MATERIAL_TYPE m_type;
};
