#include "Light.cuh"

CUDA_FUNC PointLight::PointLight(const float3 &position, const float3 &color) : pos(position), illum(color), Light(true) {}
CUDA_FUNC float3 PointLight::lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample) const 
{
    *wi = Ray(pos, ref.pos - pos);

    //float t1 = length(ref.pos - pos);
    ref.pdf_light = 1.0f;
    return illum / fmaxf(0.001f, dot(ref.pos - pos, ref.pos - pos));
}

CUDA_FUNC float3 PointLight::getPower(float3 bound_length) const
{
    return 4.0f * M_PI * illum;
}

DirectionalLight::DirectionalLight(const float3 &direction, const float3 &color) : dir(normalize(direction)), illum(color), Light(true) {}

float3 DirectionalLight::lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample) const
{
    *wi = Ray(ref.pos - dir * 10000.0f, dir);
    ref.pdf_light = 1.0f;
    return illum;
}

CUDA_FUNC float3 DirectionalLight::getPower(float3 bound_length) const
{
    return illum * dot(bound_length, bound_length) * 0.25f * M_PI;
}

CUDA_FUNC TriangleLight::TriangleLight(const Triangle& triangle, const float3& light_color, bool two) : tri(triangle), illum(light_color), two_side(two),
Light(false){}

//Incompleted
CUDA_FUNC float3 TriangleLight::lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample) const
{
    float sq_x = sqrtf(sample.x);
    float3 tri_sample = make_float3(1 - sq_x, sample.y * sq_x, 0);
    tri_sample.z = 1.0f - tri_sample.x - tri_sample.y;
    float3 pos = tri.interpolatePosition(tri_sample);

    if (wi == nullptr)
        return BLACK;
    else *wi = Ray(pos, ref.pos - pos);

    ref.pdf_light = 1.0f / tri.area();
    return illum;
}


CUDA_FUNC bool TriangleLight::hit(Ray &r, IntersectRecord &rec)
{
    float t1 = rec.t;
    tri.hit(r, rec);
    if (rec.t > 0.0001f && rec.t < 100000.0f && rec.t < t1)
    {
        rec.isLight = true;
        rec.light = this;
        rec.light_type = (int)light::TRIANGLE_LIGHT;
        return true;
    }
    else return false;
}

CUDA_FUNC float3 TriangleLight::getPower(float3 bound_lenght) const
{
    return illum * tri.area() * M_PI * (two_side ? 2.0f : 1.0f);
}