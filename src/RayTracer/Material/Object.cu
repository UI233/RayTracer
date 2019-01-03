#include "Object.cuh"
CUDA_FUNC float3 Material::f(const float3 &wo, const float3 &wi) const
{
    if (m_type == material::LAMBERTIAN)
    {
        Lambertian tmp = *(Lambertian*)brdfs;
        return tmp.f(world2Local(wo), world2Local(wi));
    }
    else if (m_type == material::FRESNEL)
    {
        return BLACK;
    }
    else if (m_type == material::GGX)
    {

    }

    return BLACK;
}

CUDA_FUNC float3 Material::sample_f(const float3 &wo, float3 *wi, float *pdf, const float2 &sample) const
{
    float3 rwo = world2Local(wo);
    float3 res = BLACK;
    if (m_type == material::LAMBERTIAN)
    {
        Lambertian tmp = *(Lambertian*)brdfs;
        return tmp.sample_f(rwo, wi, pdf, sample);
    }
    else if (m_type == material::FRESNEL)
    {

    }
    else if (m_type == material::GGX)
    {

    }

    *wi = local2World(*wi);

    return res;
}

CUDA_FUNC float Material::PDF(const float3 &wo, const float3 &wi) const
{
    float3 rwo = world2Local(wo), rwi = world2Local(wi);

    if (m_type == material::LAMBERTIAN)
    {
        Lambertian tmp =  *(Lambertian*)brdfs;
        return tmp.PDF(rwo, rwi);
    }
    else if(m_type == material::FRESNEL)
    {
        return 0.0f;
    }
    else if(m_type == material::GGX)
    {

    }

    return 0.0f;
}

CUDA_FUNC float3 Material::world2Local(const float3 & world) const
{
    static float3 B;
    B = cross(tangent, normal);
    return dot(world, normal) * normal + dot(world, tangent) * tangent + dot(world, B) * B;
}

CUDA_FUNC float3 Material::local2World(const float3 & local) const
{
    return local.x * tangent + local.y * normal + local.z * cross(tangent, normal);
}