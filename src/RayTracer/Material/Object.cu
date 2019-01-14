#include "Object.cuh"
#include <cstdio>
CUDA_FUNC float3 Material::f(float3 normal, float3 tangent, const float3 &wo, const float3 &wi, float3 color) const
{
    float3 f = BLACK;
    if (m_type == material::LAMBERTIAN)
    {
        Lambertian tmp = *(Lambertian*)brdfs;
        f = tmp.f(world2Local(normal, tangent, wo), world2Local(normal, tangent, wi), color);
    }
    else if (m_type == material::FRESNEL)
    {
        return BLACK;
    }
    else if (m_type == material::GGX)
    {
        GGX tmp = *(GGX *)brdfs;
        f = tmp.f(world2Local(normal, tangent, wo), world2Local(normal, tangent, wi), color);
    }
    else if (m_type == material::Cook)
    {
        Cook_Torrance tmp = *(Cook_Torrance *)brdfs;
        f = tmp.f(world2Local(normal, tangent, wo), world2Local(normal, tangent, wi), color);

    }
    else if (m_type == material::Oren_Nayar)
    {
        Oren_Nayar tmp = *(Oren_Nayar *)brdfs;
        f = tmp.f(world2Local(normal, tangent, wo), world2Local(normal, tangent, wi), color);
    }
    return f;
}

CUDA_FUNC float3 Material::sample_f(float3 normal, float3 tangent, const float3 &wo, float3 *wi, float *pdf, const float2 &sample
    ,float3 color) const
{
    float3 rwo = world2Local(normal, tangent, wo);
    float3 res = BLACK;
    if (m_type == material::LAMBERTIAN)
    {
        Lambertian tmp;
        tmp = *(Lambertian*)brdfs;
        res = tmp.sample_f(rwo, wi, pdf, sample, color);
    }
    else if (m_type == material::FRESNEL)
    {
        Fresnel tmp;
        tmp = *(Fresnel*)brdfs;
        res = tmp.sample_f(rwo, wi, pdf, sample, color);
    }
    else if (m_type == material::GGX)
    {
        GGX tmp;
        tmp = *(GGX*)brdfs;
        res = tmp.sample_f(rwo, wi, pdf, sample, color);
    }
    else if (m_type == material::Cook)
    {
        Cook_Torrance tmp;
        tmp = *(Cook_Torrance*)brdfs;
        res = tmp.sample_f(rwo, wi, pdf, sample, color);
    }
    else if (m_type == material::Oren_Nayar)
    {
        Oren_Nayar tmp;
        tmp = *(Oren_Nayar*)brdfs;
        res = tmp.sample_f(rwo, wi, pdf, sample, color);
    }

    *wi = local2World(normal, tangent, *wi);
    return res;
}

CUDA_FUNC float Material::PDF(float3 normal, float3 tangent, const float3 &wo, const float3 &wi) const
{
    float3 rwo = world2Local(normal, tangent, wo), rwi = world2Local(normal, tangent, wi);

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
        GGX tmp = *(GGX*)brdfs;
        return tmp.PDF(rwo, rwi);
    }
    else if (m_type == material::Cook)
    {
        Cook_Torrance tmp = *(Cook_Torrance*)brdfs;
        return tmp.PDF(rwo, rwi);
    }
    else if (m_type == material::Oren_Nayar)
    {
        Oren_Nayar tmp = *(Oren_Nayar*)brdfs;
        return tmp.PDF(rwo, rwi);
    }

    return 0.0f;
}

//The normal and tangant for sphere
CUDA_FUNC float3 Material::world2Local(float3 normal, float3 tangent, const float3 & world) const
{
    return make_float3(dot(world, tangent) , dot(world, normal) ,dot(world, normalize(cross(tangent, normal))));
}

CUDA_FUNC float3 Material::local2World(float3 normal, float3 tangent, const float3 & local) const
{
    return local.x * tangent + local.y * normal + local.z * normalize(cross(tangent, normal));
}