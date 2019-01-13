#define _USE_MATH_DEFINES
#include "material.cuh"
#include <cmath>
#include <cstdio>
#define INV_PI 0.318309892f
#define MAXN 0xffffffffu
#define PiOver4 0.78539816339744830961f
#define PiOver2 1.57079632679489661923f
CUDA_FUNC float3 sampleHemi(const float2 &sample)
{
    float2 rs = 2.0f * sample - make_float2(1.0f, 1.0f);
    if (rs.x == 0.0f && rs.y == 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);
    float2 d;
    float r, theta;
    if (fabs(rs.x) > fabs(rs.y))
    {
        r = rs.x;
        theta = PiOver4 * (rs.y / rs.x);
    }
    else
    {
        r = rs.y;
        theta = PiOver2 - PiOver4 * (rs.y / rs.x);
    }

    d = r * make_float2(sinf(theta), cosf(theta));
    float z = sqrtf(fmaxf(0.0f, 1 - d.x * d.x - d.y * d.y));

    return make_float3(d.x, z, d.y);
}

CUDA_FUNC float3 Lambertian::f(const float3 &wo, const float3 &wi, float3 c)const
{
    if (c.x < 0.0f)
        c = color;
    return  (wo.y > 0.0f && wi.y > 0.0f) ? c * INV_PI : make_float3(0.0f, 0.0f, 0.0f);
}

CUDA_FUNC float3 Lambertian::sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample, float3 c) const
{
    if (c.x < 0.0f)
        c = color;
    if (pdf == nullptr || wi == nullptr)
        return make_float3(0.0f);


    *wi = sampleHemi(sample);
    *pdf = PDF(wo, *wi);

    return f(wo, *wi, c);
}

CUDA_FUNC bool Lambertian::isSpecular() const
{
    return false;
}


CUDA_FUNC float Lambertian::PDF(const float3 &wo, const float3 &wi) const
{
    return wi.y > +0.0f ? wi.y * INV_PI : 0.0f;
}