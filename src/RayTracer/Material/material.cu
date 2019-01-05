#define _USE_MATH_DEFINES
#include "material.cuh"
#include <cmath>
#include <cstdio>
#define INV_PI 0.318309892f
#define MAXN 0xffffffffu


CUDA_FUNC float3 Lambertian::f(const float3 &wo, const float3 &wi)const
{
    //if(wo.y >0.0f && wi.y > 0.0f)
    //    printf("%f %f\n", wo.y, wi.y);
    return  (wo.y > 0.0f && wi.y > 0.0f) ? color * INV_PI : make_float3(0.0f, 0.0f, 0.0f);
}

CUDA_FUNC float3 Lambertian::sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample) const
{
    if (pdf == nullptr || wi == nullptr)
        return make_float3(0.0f);

    float theta = sample.x * 2.0f * M_PI, phi = sample.y * M_PI_2;
    float sinPhi = sin(phi), cosPhi = cos(phi);

    *wi = make_float3(sin(theta) * sinPhi, cosPhi, cos(theta) * sinPhi);
    *pdf = PDF(wo, *wi);

    return f(wo, *wi);
}

CUDA_FUNC bool Lambertian::isSpecular() const
{
    return false;
}


CUDA_FUNC float Lambertian::PDF(const float3 &wo, const float3 &wi) const
{
    return wi.y * INV_PI;
}