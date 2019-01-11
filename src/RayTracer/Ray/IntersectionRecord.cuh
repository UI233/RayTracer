#pragma once
#ifndef INTERSECTIONRECORD_H
#define INTERSECTIONRECORD_H
#include "../Material/Object.cuh"
#include "../Matrix/Matrix.cuh"
#ifndef INF
#define INF 1000000.0f
#endif // !INFINITY

CUDA_FUNC float findFloatUp(float v);

CUDA_FUNC float findFloatDown(float v);

CUDA_FUNC float3 offsetFromPoint(float3 origin, float3 normal, float3 error_bound, float3 d);


class IntersectRecord
{
public:
    CUDA_FUNC IntersectRecord() : t(INF), isLight(false), lightidx(-1), color(make_float3(-1.0f, -1.0f, -1.0f)) {}
    CUDA_FUNC ~IntersectRecord() = default;

    float3 pos, normal, tangent;
    float3 color;
    float t;
    Ray wo;
    int lightidx;
    int light_type;
    float pdf_light, pdf_surface;
    Material *material;
    bool isLight;
    int material_type;

    //Make sure that the light wounldn't reintersect the surface 
    CUDA_FUNC Ray spawnRay(const float3 &w)
    {
        float3 ro = offsetFromPoint(pos, normal, make_float3(0.01f, 0.01f, 0.01f), w);
        return Ray(ro, w);
    }

    CUDA_FUNC float3 f(const float3 &wo, const float3 &wi) const
    {
        return material->f(normal, tangent, wo, wi, color);
    }

    //emited ray, incident ray, possibility for wi, the 2-D sample points between [0, 1]
    CUDA_FUNC  float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample) const
    {
        return material->sample_f(normal, tangent, wo, wi, pdf,sample, color);
    }

    CUDA_FUNC float PDF( const float3 &wo, const float3 &wi) const
    {
        return material->PDF(normal, tangent, wo, wi);
    }

};

#endif