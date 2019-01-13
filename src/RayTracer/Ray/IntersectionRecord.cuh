#pragma once
#ifndef INTERSECTIONRECORD_H
#define INTERSECTIONRECORD_H
#include "../Material/Object.cuh"
#include "../Matrix/Matrix.cuh"
#include "../Camera/Camera.cuh"
#ifndef INF
#define INF 1000000.0f
#endif // !INFINITY

CUDA_FUNC float findFloatUp(float v);

CUDA_FUNC float findFloatDown(float v);

CUDA_FUNC float3 offsetFromPoint(float3 origin, float3 normal, float3 error_bound, float3 d);


class IntersectRecord
{
public:
    CUDA_FUNC IntersectRecord() : t(INF), isLight(false), lightidx(-1), color(make_float3(-1.0f, -1.0f, -1.0f)), global_cam(nullptr) {}
    CUDA_FUNC ~IntersectRecord() = default;

    float3 pos, normal, tangent;
    float3 color;
    float t;
    Ray wo;
    float2 uv;
    int lightidx;
    int light_type;
    float pdf_light, pdf_surface;
    Material *material;
    bool isLight;
    int material_type;
    float2 du, dv;
    Camera *global_cam;
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

    //variable:emited ray, incident ray, possibility for wi, the 2-D sample points between [0, 1]
    CUDA_FUNC  float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample) const
    {
        return material->sample_f(normal, tangent, wo, wi, pdf,sample, color);
    }

    CUDA_FUNC float PDF( const float3 &wo, const float3 &wi) const
    {
        return material->PDF(normal, tangent, wo, wi);
    }
    
    CUDA_FUNC float4 getdxdy(float3 pos1, float2 uv1, float3 pos2, float2 uv2)
    {
        if (!global_cam)
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float2 duv = uv2 - uv1;
        float4 res;
        float2 xy1 = global_cam->getxy(pos1), xy2 = global_cam->getxy(pos2);
        float2 dxy = xy2 - xy1;

        res.x = duv.x / dxy.x;
        res.y = duv.y / dxy.x;
        res.z = duv.x / dxy.y;
        res.w = duv.y / dxy.y;

        if (duv.x == 0.0f)
            res.z = res.x = 0.0f;
        if (duv.y = 0.0f)
            res.w = res.y = 0.0f;

        return make_float4(fabsf(res.x), fabsf(res.y), fabsf(res.z), fabsf(res.w));
    }
};

#endif