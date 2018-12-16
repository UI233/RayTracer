#include "Light.cuh"


CUDA_FUNC PointLight::PointLight(const float3 &position, const float3 &color) : pos(position), illum(color) {}
CUDA_FUNC float3 PointLight::lightIllumi(Model &scene, IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const 
{
    *wi = Ray(pos, ref.pos - pos);

    float t1 = length(ref.pos - pos);

    IntersectRecord rec;

    scene.hit(*wi, rec);
    
    if (fabs(ref.t - t1) > 0.0001f)
        return make_float3(0.0f);

    return illum / dot(ref.pos - pos, ref.pos - pos);
}

DirectionalLight::DirectionalLight(const float3 &direction, const float3 &color) : dir(normalize(direction)), illum(color) {}

float3 DirectionalLight::lightIllumi(Model &scene, IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const
{
    *wi = Ray(ref.pos - dir * 10000.0f, dir);

    IntersectRecord rec;

    scene.hit(*wi, rec);

    if (length(rec.pos - ref.pos) > 0.0001f)
        return make_float3(0.0f);

    return illum;
}