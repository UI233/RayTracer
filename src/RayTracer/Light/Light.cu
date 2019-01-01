#include "Light.cuh"

CUDA_FUNC PointLight::PointLight(const float3 &position, const float3 &color) : pos(position), illum(color) {}
CUDA_FUNC float3 PointLight::lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample) const 
{
    *wi = Ray(pos, ref.pos - pos);

    float t1 = length(ref.pos - pos);

    return illum / fmaxf(0.001f, dot(ref.pos - pos, ref.pos - pos));
}

DirectionalLight::DirectionalLight(const float3 &direction, const float3 &color) : dir(normalize(direction)), illum(color) {}

float3 DirectionalLight::lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample) const
{
    *wi = Ray(ref.pos - dir * 10000.0f, dir);

    return illum;
}

CUDA_FUNC float3 PointLight::getDir(float3 pos0 = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const
{
    return pos - pos0;
}

CUDA_FUNC float3 DirectionalLight::getDir(float3 pos0 = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const
{
    return dir;
}

CUDA_FUNC float3 TriangleLight::getDir(float3 pos0 = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const
{
    static float3 pos;
    pos = tri.interpolatePosition(make_float3(sample, 1.0f - sample.x - sample.y));
    return pos - pos0;
}
