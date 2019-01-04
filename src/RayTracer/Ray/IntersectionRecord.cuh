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
    CUDA_FUNC IntersectRecord() = default;
    CUDA_FUNC ~IntersectRecord() = default;
    CUDA_FUNC IntersectRecord(const float3 &p, const float3 &n, const Ray &r, const float &dis = INF) : pos(p), normal(n), t(dis), wo(r) {};

    float3 pos, normal;
    //World2Local
    mat4 transformation;
    float t;
    Ray wo;
    void* light;
    int light_type;
    float pdf_light, pdf_surface;
    Material *material;
    bool isLight;
    int material_type;

    //Make sure that the light wounldn't reintersect the surface 
    CUDA_FUNC Ray spawnRay(const float3 &w)
    {
        float3 ro = offsetFromPoint(pos, normal, make_float3(0.0001f, 0.0001f, 0.0001f), w);
        return Ray(ro, w);
    }

};

#endif