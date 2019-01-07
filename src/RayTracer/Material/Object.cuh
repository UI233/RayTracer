#pragma once
#include "material.cuh"

namespace material
{
    enum BXDF_PROP
    {
        REFLECATION = 1,
        TRANSMISSION = 2,
        SPECULAR = 4
    };

    enum MATERIAL_TYPE
    {
        LAMBERTIAN = 1,
        FRESNEL = 7, // 111
        GGX = 9,//1001
        MATERIAL_NUM = 3
    };
}

class Material
{
public:
    ~Material() {
        /*if (brdfs != nullptr)
            free(brdfs);*/
    };
    Material() :brdfs(nullptr) {};
    Material(BRDF *b, material::MATERIAL_TYPE type) : brdfs(b), m_type(type){};
    CUDA_FUNC float3 f(float3 normal, float3 tangent, const float3 &wo, const float3 &wi) const;
    //emited ray, incident ray, possibility for wi, the 2-D sample points between [0, 1]
    CUDA_FUNC  float3 sample_f(float3 normal, float3 tangent, const float3 &wo, float3 *wi, float *pdf, const float2 &sample) const;
    CUDA_FUNC float PDF(float3 normal, float3 tangent, const float3 &wo, const float3 &wi) const;
    CUDA_FUNC float3 world2Local(float3 normal, float3 tangent, const  float3 &) const;
    CUDA_FUNC float3 local2World(float3 normal, float3 tangent, const  float3 &) const;
    CUDA_FUNC bool isSpecular() const
    {
        return m_type  & material::SPECULAR;
    }

    CUDA_FUNC bool isTrans() const
    {
        return m_type & material::TRANSMISSION;
    }


    float eta;
    BRDF *brdfs;
protected:
    material::MATERIAL_TYPE m_type;
};
