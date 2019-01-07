#pragma once
#include "../Model/Model.cuh"
#include "thrust/device_vector.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "helper_math.h"
#ifndef BLACK
#define BLACK make_float3(0.0f,0.0f,0.0f)
#endif // !BLACK

namespace light
{
    enum LIGHT_TYPE
    {
        POINT_LIGHT,
        DIR_LIGHT,
        TRIANGLE_LIGHT,
        TYPE_NUM
    };
}

class Light 
{
public:
    CUDA_FUNC virtual ~Light() = default;
    CUDA_FUNC Light() = default;
    CUDA_FUNC virtual float3 getPower(float3 bound_length = make_float3(0.0f, 0.0f, 0.0f)) const = 0;
    //CUDA_FUNC virtual float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const = 0;
    CUDA_FUNC virtual float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const = 0;
    CUDA_FUNC virtual float PDF(IntersectRecord rec, const float3 &wi)const { return 0.0f; };
    CUDA_FUNC virtual float3 getLe(Ray &r) const{ return BLACK; };
    CUDA_FUNC virtual float3 L(const float3 &r, IntersectRecord *rec = nullptr) const { return BLACK; }
};

class PointLight : public Light
{
public:
    CUDA_FUNC  PointLight() = default;
    CUDA_FUNC ~PointLight() = default;
    //CUDA_FUNC virtual float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const override;
    CUDA_FUNC PointLight(const float3 &position, const float3 &color);
    CUDA_FUNC float3 getPower(float3 bound_length = make_float3(0.0f, 0.0f, 0.0f)) const;
    CUDA_FUNC float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;
private:
    float3 pos, illum;
};

class DirectionalLight : public Light
{
public:
    CUDA_FUNC DirectionalLight() = default;
    CUDA_FUNC ~DirectionalLight() = default;
    CUDA_FUNC DirectionalLight(const float3 &direction, const float3 &color);
    CUDA_FUNC float3 getPower(float3 bound_length = make_float3(0.0f, 0.0f, 0.0f)) const;
    //CUDA_FUNC float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const override;
    CUDA_FUNC float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;

private:
    float3 dir, illum;
};

class TriangleLight : public Light
{
public:
    CUDA_FUNC TriangleLight() = default;
    CUDA_FUNC ~TriangleLight() = default;
    CUDA_FUNC TriangleLight(float3 posa, float3 posb, float3 posc, const float3& light_color, bool two = false);
    CUDA_FUNC float3 getPower(float3 bound_length = make_float3(0.0f, 0.0f, 0.0f)) const;
    CUDA_FUNC float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;
    CUDA_FUNC bool hit(Ray &r, IntersectRecord &rec);
    CUDA_FUNC float3 interpolatePosition(float3 tri_sample) const;
    CUDA_FUNC float area() const
    {
        return length(cross(pos[2] - pos[0], pos[1] - pos[0]));
    }

    CUDA_FUNC float PDF(IntersectRecord rec, const float3 &wi)  const{ 
        return rec.t * rec.t / (area() * fabs(dot(rec.normal , wi))); 
    };
    CUDA_FUNC float3 L(const float3 &r, IntersectRecord *rec = nullptr) const
    {
        if (two_side)
            return illum;
        else return dot(r, rec->normal) > 0 ? illum : BLACK;
    };
    __host__ bool setUpMaterial(material::MATERIAL_TYPE type, Material *mat);

private:
    float3 pos[3];
    float3 normal;
    float3 illum;
    material::MATERIAL_TYPE material_type;
    Material *my_material;
    bool two_side;
};

class EnvironmentLight : public Light
{
public:
    EnvironmentLight() {};
    ~EnvironmentLight() = default;
    CUDA_FUNC virtual float3 getPower(float3 bound_length = make_float3(0.0f, 0.0f, 0.0f)) const = 0;
    //CUDA_FUNC virtual float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const = 0;
    CUDA_FUNC float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const = 0;
    CUDA_FUNC float PDF(IntersectRecord rec, const float3 &wi)const { return 0.0f; };
    CUDA_FUNC float3 getLe(Ray &r) const { return BLACK; };
    CUDA_FUNC float3 L(const float3 &r, IntersectRecord *rec = nullptr) const { return BLACK; }
private:
    char * texture;
};
