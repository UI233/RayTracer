#include "Light.cuh"

CUDA_FUNC PointLight::PointLight(const float3 &position, const float3 &color) : pos(position), illum(color) {}
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

DirectionalLight::DirectionalLight(const float3 &direction, const float3 &color) : dir(normalize(direction)), illum(color) {}

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

CUDA_FUNC TriangleLight::TriangleLight(float3 posa, float3 posb, float3 posc, const float3& light_color, bool two) : illum(light_color), two_side(two){
    pos[0] = posa;
    pos[1] = posb;
    pos[2] = posc;
    normal = normalize(cross(pos[2] - pos[0], pos[1] - pos[0]));
}

CUDA_FUNC float3 TriangleLight::interpolatePosition(float3 tri_sample) const
{
    return tri_sample.x * pos[0] + tri_sample.y * pos[1] + tri_sample.z * pos[2];
}
//Incompleted
CUDA_FUNC float3 TriangleLight::lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample) const
{
    float sq_x = sqrtf(sample.x);
    float3 tri_sample = make_float3(1 - sq_x, sample.y * sq_x, 0);
    tri_sample.z = 1.0f - tri_sample.x - tri_sample.y;
    float3 pos = interpolatePosition(tri_sample);

    if (wi == nullptr)
        return BLACK;
    else *wi = Ray(pos, ref.pos - pos);

    float t = ref.t;
    ref.t = length(ref.pos - pos);
    //bugs here
    ref.pdf_light = PDF(ref, wi ->getDir());
    ref.t = t;
    return illum;
}


CUDA_FUNC bool TriangleLight::hit(Ray &r, IntersectRecord &rec)
{

    float t;
    float3 normal = cross(pos[0] - pos[2], pos[1] - pos[2]);
    float dot_normal_dir = dot(normal, r.getDir());
    if (fabs(dot_normal_dir) < FLOAT_EPISLON)
        return false;

    t = (-dot(r.getOrigin(), normal) + dot(pos[0], normal)) / dot_normal_dir;

    float3 rpos = r.getPos(t);

    float S = area();
    float s1 = length(cross(rpos - pos[0], rpos - pos[1]));
    float s2 = length(cross(rpos - pos[2], rpos - pos[0]));
    float s3 = length(cross(rpos - pos[2], rpos - pos[1]));


    if (fabs(s1 + s2 + s3 - S) > 0.001f)
        return false;

    float m1 = s3 / S, m2 = s2 / S, m3 = 1.0f - m1 - m2;

    if (t > FLOAT_EPISLON && t < rec.t)
    {
        rec.material = my_material;
        rec.material_type = material_type;
        rec.t = t;
        rec.normal = two_side ? (dot(normal, r.getDir()) > 0 ? -normal : normal) : normal;
        rec.pos = r.getPos(t);
        rec.isLight = true;
        rec.tangent = normalize((pos[1] - pos[0]) * m2 + (pos[2] - pos[0]) * m3);
        return true;
    }

    return false;
}

CUDA_FUNC float3 TriangleLight::getPower(float3 bound_lenght) const
{
    return illum * area() * M_PI * (two_side ? 2.0f : 1.0f);
}

CUDA_FUNC bool TriangleLight::setUpMaterial(material::MATERIAL_TYPE t, Material *mat)
{
    size_t num;
    switch (t)
    {
    case material::LAMBERTIAN:
        num = sizeof(Lambertian);
        break;
    case material::MATERIAL_NUM:
        //break;
    default:
        num = 0;
        return false;
    }
    //Todo:Bugs fucking here!
    material_type = t;
    Material tmp = *mat;
    cudaMalloc(&tmp.brdfs, num);
    cudaMemcpy(tmp.brdfs, mat->brdfs, num, cudaMemcpyHostToDevice);
    auto error = cudaMalloc(&my_material, sizeof(Material));
    error = cudaMemcpy(my_material, &tmp, sizeof(Material), cudaMemcpyHostToDevice);

    return error == cudaSuccess;
}