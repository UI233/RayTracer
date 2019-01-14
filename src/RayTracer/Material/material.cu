#define _USE_MATH_DEFINES
#include "material.cuh"
#include <cmath>
#include <cstdio>
#define INV_PI 0.318309892f
#define MAXN 0xffffffffu
#define PiOver4 0.78539816339744830961f
#define PiOver2 1.57079632679489661923f
#define ex 2.718281828f

CUDA_FUNC float Clamp(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    else if (hi < v) {
        return hi;
    }
    else {
        return v;
    }
}

CUDA_FUNC inline bool SameHemisphere(const float3 &w, const float3 &wp) {
    return w.y * wp.y > 0;
}

//Lambertian Method Defination
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
        theta = PiOver2 - PiOver4 * (rs.x / rs.y);
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

//GGX method Defination
CUDA_FUNC float3 GGX::SchlickFresnel(float3 r0, float r) const
{
    float t = (1.0f - r);

    return r0 + (1.0f - r0) * (t * t) * (t * t) * t;
}

CUDA_FUNC float GGX::G(float3 wi, float3 wo) const
{
    float A = wo.y * sqrtf(a2 + (1.0f - a2) * wi.y * wi.y);
    float B = wi.y * sqrtf(a2 + (1.0f - a2) * wo.y * wo.y);

    return 2.0f * wo.y * wi.y / (A + B);
}

CUDA_FUNC float GGX::D(float3 wh) const
{
    return a2 / (M_PI * (wh.y * wh.y * (a2 - 1) + 1) *  (wh.y * wh.y * (a2 - 1) + 1));
}

CUDA_FUNC float3 GGX::f(const float3 &wo, const float3 &wi, float3 c)const
{
    if (c.x < 0.0f)
        c = color;

    float dotwin_won = fmaxf(0.0001f, fabsf(4.0f * wi.y * wo.y));
    float3 wh = normalize(wo + wi);
    
    return SchlickFresnel(c, dot(wi, wh)) * D(wh) * G(wi,wo) / dotwin_won;
}

CUDA_FUNC float3 GGX::sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample, float3 c) const
{
    if (c.x < 0.0f)
        c = color;

    float theta = acosf(sqrtf((1.0f - sample.x) / ((a2 - 1.0f) * sample.x  + 1.0f)));
    float phi = sample.y * M_PI * 2.0f;

    float3 wh = make_float3(sin(theta) * sin(phi), cos(theta), cos(phi) * sin(theta));

    *wi = 2.0f * dot(wo, wh) * wh - wo;
    *pdf = PDF(wo, *wi);

    return f(wo, *wi, c);
}

CUDA_FUNC float GGX::PDF(const float3 &wo, const float3 &wi) const
{
    float3 wh = normalize(wo + wi);

    return wi.y > 0.0f && dot(wi, wh) >0.0f  ? D(wh) * wh.y * 0.25f / wo.y : 0.0f;
}

//Oren Nayar:A modified Lambertian for rough opaque diffuse surfaces
CUDA_FUNC float3 Oren_Nayar::f(const float3 &wo, const float3 &wi, float3 c)const
{
    if (c.x < 0.0f)
        c = color;
    float sigma = 0.5;
    float A = 1 - 0.5*sigma*sigma / (sigma*sigma + 0.33);
    float B = 0.45*sigma*sigma / (sigma*sigma + 0.09);
    float CosThetaI = wi.y, CosThetaO = wo.y;
    float SinThetaI = sqrt(1 - CosThetaI * CosThetaI);
    float SinThetaO = sqrt(1 - CosThetaO * CosThetaO);

    float CosPhiI = (SinThetaI == 0) ? 1 : Clamp(wi.x / SinThetaI, -1, 1);
    float CosPhiO = (SinThetaO == 0) ? 0 : Clamp(wo.x / SinThetaO, -1, 1);
    float SinPhiI = (SinThetaI == 0) ? 1 : Clamp(wi.z / SinThetaI, -1, 1);
    float SinPhiO = (SinThetaO == 0) ? 0 : Clamp(wo.z / SinThetaI, -1, 1);
    float dCos = CosPhiI * CosPhiO + SinPhiI * SinPhiO;
    float maxCos = (dCos > 0) ? dCos : 0;

    float CosAlpha, SinAlpha, TanAlpha;
    float CosBeta, SinBeta, TanBeta;
    if (CosThetaI*CosThetaI > CosThetaO*CosThetaO) {
        SinAlpha = SinThetaO;
        TanBeta = SinThetaI * 1.0 / ((CosThetaI > 0) ? CosThetaI : (-1 * CosThetaI));
    }
    else {
        SinAlpha = SinThetaI;
        TanBeta = SinThetaI * 1.0 / ((CosThetaO > 0) ? CosThetaO : (-1 * CosThetaO));
    }

    return  (wo.y > 0.0f && wi.y > 0.0f) ? c * INV_PI * (A + B *maxCos*SinAlpha*TanBeta) : make_float3(0.0f, 0.0f, 0.0f);
}
CUDA_FUNC float3 Oren_Nayar::sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample, float3 c) const
{
    if (c.x < 0.0f)
        c = color;

    *wi = sampleHemi(sample);
    *pdf = PDF(wo, *wi);

    return f(wo, *wi, c);
}
CUDA_FUNC bool Oren_Nayar::isSpecular() const
{
    return false;
}
CUDA_FUNC float Oren_Nayar::PDF(const float3 &wo, const float3 &wi) const
{
    return wi.y > +0.0f ? wi.y * INV_PI : 0.0f;
}

//-----------------------------------------------------------------------------------------------------------------------------
//Cook_Torrance
CUDA_FUNC float Cook_Torrance::G(float3 wi, float3 wo) const
{
    float A = wo.y * sqrtf(a2 + (1.0f - a2) * wi.y * wi.y);
    float B = wi.y * sqrtf(a2 + (1.0f - a2) * wo.y * wo.y);

    return 2.0f * wo.y * wi.y / (A + B);
}

CUDA_FUNC float3 Cook_Torrance::f(const float3 &wo, const float3 &wi,  float3 c)const
{
    if (c.x < 0.0f)
        c = color;
    float dotwin_won = fmaxf(0.0001f, fabsf(4.0f * wi.y * wo.y));
    float3 wh = normalize(wo + wi);

    return F(c, dot(wi, wh)) * D(wh) * G(wi, wo) / dotwin_won;
}
CUDA_FUNC float3 Cook_Torrance::sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample,  float3 c) const
{
    if (c.x < 0.0f)
        c = color;

    if (pdf == nullptr || wi == nullptr)
        return make_float3(0.0f);
    float tan2Theta, phi;
    float logSample = log(1 - sample.x);
    if (isinf(logSample)) 
        logSample = 0;
    tan2Theta = -m * m *logSample;
    phi = sample.y * 2 * M_PI;
    float cosTheta = 1 / pow(1 + tan2Theta, 0.5);
    float sinTheta = sqrtf(fmaxf((float)0, 1 - cosTheta * cosTheta));

    float3 wh = make_float3(sinTheta * sin(phi), cosTheta, cos(phi) * sinTheta);
    if (!SameHemisphere(wo, wh)) 
        wh = -wh;

    wh = normalize(wo + *wi);
    *wi = 2.0f*dot(wo, wh)*wh - wo;
    *pdf = PDF(wo, *wi);

    return f(wo, *wi,  c);
}

CUDA_FUNC bool Cook_Torrance::isSpecular() const
{
    return false;
}

CUDA_FUNC float Cook_Torrance::D(const float3 &wo) const {
    float a = acos(wo.y);
    float c = 1.0 / (m*m*pow(wo.y, 4));
    return c * pow(ex, -(acos(wo.y)*acos(wo.y)) / (m*m));
}
CUDA_FUNC float Cook_Torrance::PDF(const float3 &wo, const float3 &wi) const
{
    if (!SameHemisphere(wo, wi)) 
        return 0;
    float3 wh = normalize(wo + wi);
    if (dot(wo, wh) == 0.0f)
        return 0.0f;
    return D(wh) / (4 * dot(wo, wh));
}
//-----------------------------------------------------------------------------------------------------------------------------
