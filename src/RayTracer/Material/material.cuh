#pragma once
#include "../Ray/Ray.cuh"
#include "curand.h"
#include <cmath>
#include "curand_kernel.h"
#include <cstdio>
inline CUDA_FUNC float rough2alpha(float rough)
{
    rough = fmaxf(rough, 1e-3f);
    float x = logf(rough);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x +
        0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}
//All computations happens in the local coordinate system(TBN space) where the normal vector would be (0, 1, 0)
//The computation for T vector and B vector will be posted in issue.
class BRDF
{
public:
    CUDA_FUNC BRDF() = default;
    CUDA_FUNC virtual ~BRDF() = default;
    CUDA_FUNC virtual float3 f(const float3 &wo, const float3 &wi, float3 color = make_float3(-1.0f, -1.0f, -1.0f)) const = 0;
    //emited ray, incident ray, possibility for wi, the 2-D sample points between [0, 1]
    CUDA_FUNC virtual float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample, float3 color = make_float3(-1.0f, -1.0f, -1.0f)) const = 0;
    CUDA_FUNC virtual float PDF(const float3 &wo, const float3 &wi) const = 0;
    CUDA_FUNC virtual bool isSpecular() const = 0;
};

class Lambertian : public BRDF
{
public:
    CUDA_FUNC Lambertian() = default;
    CUDA_FUNC Lambertian(const float3 &f) : color(f) {}
    CUDA_FUNC ~Lambertian() = default;
    CUDA_FUNC float3 f(const float3 &wo, const float3 &wi, float3 color = make_float3(-1.0f, -1.0f, -1.0f))const  override;
    CUDA_FUNC float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &, float3 color = make_float3(-1.0f, -1.0f, -1.0f)) const override;
    CUDA_FUNC bool isSpecular() const override;
    CUDA_FUNC float PDF(const float3 &wo, const float3 &wi) const;
private:
    float3 color;
};

//GGX BRDF
//Reference : https://schuttejoe.github.io/post/ggximportancesamplingpart1/
class GGX : public BRDF
{
public:
    CUDA_FUNC GGX() = default;
    CUDA_FUNC GGX(float3 c, float roughness) : color(c), a2(rough2alpha(roughness) * rough2alpha(roughness)){};// with parameter
    CUDA_FUNC ~GGX() = default;
    CUDA_FUNC float3 f(const float3 &wo, const float3 &wi, float3 color = make_float3(-1.0f, -1.0f, -1.0f))const  override;
    CUDA_FUNC float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &, float3 color = make_float3(-1.0f, -1.0f, -1.0f)) const override;
    
    CUDA_FUNC bool isSpecular() const override
    {
        return false;
    }
    
    CUDA_FUNC float PDF(const float3 &wo, const float3 &wi) const;
private:
    float alpha;
    float a2;
    float3 color;
    CUDA_FUNC float3 SchlickFresnel(float3 r0, float r) const;
    CUDA_FUNC float G(float3 wi, float3 wo) const;
    CUDA_FUNC float D(float3 wh) const;
};


//-----------------------------------------------------------------------------------------------------------------------------
class Oren_Nayar : public BRDF
{
public:
    CUDA_FUNC Oren_Nayar() = default;
    CUDA_FUNC Oren_Nayar(const float3 &f) : color(f) {}
    CUDA_FUNC ~Oren_Nayar() = default;
    CUDA_FUNC float3 f(const float3 &wo, const float3 &wi, float3 color = make_float3(-1.0f, -1.0f, -1.0f))const  override;
    CUDA_FUNC float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &, float3 color = make_float3(-1.0f, -1.0f, -1.0f)) const override;
    CUDA_FUNC bool isSpecular() const override;
    CUDA_FUNC float PDF(const float3 &wo, const float3 &wi) const;
private:
    float3 color;
};
//-----------------------------------------------------------------------------------------------------------------------------
class Cook_Torrance : public BRDF
{
public:
    CUDA_FUNC Cook_Torrance() = default;
    CUDA_FUNC Cook_Torrance(const float3 &f, const float & rough) : color(f), m(rough2alpha(rough)) { a2 = m * m; }
    CUDA_FUNC ~Cook_Torrance() = default;
    CUDA_FUNC float3 f(const float3 &wo, const float3 &wi, float3 color = make_float3(-1.0f, -1.0f, -1.0f))const  override;
    CUDA_FUNC float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &, float3 color = make_float3(-1.0f, -1.0f, -1.0f)) const override;
    CUDA_FUNC bool isSpecular() const override;
    CUDA_FUNC float PDF(const float3 &wo, const float3 &wi) const;
private:
    CUDA_FUNC float D(const float3 &wo) const;
    CUDA_FUNC float3 F(float3 r0, float r) const
    {
        float t = (1.0f - r);

        return r0 + (1.0f - r0) * (t * t) * (t * t) * t;
    }
    CUDA_FUNC float G(float3 wi, float3 wo) const;
    float3 color;
    float m;
    float a2;
};
//-----------------------------------------------------------------------------------------------------------------------------
class Fresnel : public BRDF
{
public:
    CUDA_FUNC Fresnel() = default;
    CUDA_FUNC Fresnel(const float3 &f, const float &e) : color(f), eta(e) {}
    CUDA_FUNC ~Fresnel() = default;
    CUDA_FUNC float3 f(const float3 &wo, const float3 &wi, float3 color = make_float3(-1.0f, -1.0f, -1.0f))const  override
    {
        return BLACK;
    }
    CUDA_FUNC float3 sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample, float3 c = make_float3(-1.0f, -1.0f, -1.0f)) const override
    {
        if (c.x < 0.0f)
            c = color;
        bool entering = wo.y > 0.0f;
        float n1 = 1.0f, n2 = eta;
        float R = (n1 - n2) / (n1 + n2);
        R *= R;

        float t = fabsf(wo.y);
        float F = R + (t * t) * (t * t) * t * (1.0f - R);
        if (sample.x < F)
        {
            *pdf = F;
            *wi = make_float3(-wo.x, wo.y, -wo.z);
            return F * c / fmaxf(fabs(wi->y), 0.0000001f);
        }
        else
        {
            *pdf = 1 - F;
            float etas;
            float sin2theta = fmaxf(0.001f, 1.0f - wo.y * wo.y);
            if (!entering)
                etas = 1.0f / eta;
            else etas = eta;
            if (sin2theta * etas * etas >= 1.0f)
                return BLACK;

            float costhetaT = sqrtf(1.0f - etas * etas *sin2theta);

            *wi = -etas * wo + (etas * wo.y - costhetaT) * make_float3(0.0f, 1.0f, 0.0f);
            *wi = normalize(*wi);
            return (1.0f - F) * c / fmaxf(fabs(wi->y), 0.0000001f);
        }
    }
    CUDA_FUNC bool isSpecular() const override
    {
        return true;
    }
    CUDA_FUNC float PDF(const float3 &wo, const float3 &wi) const
    {
        return 0.0f;
    }
private:
    float3 color;
    float eta;
};