#pragma once
#include "../Model/Model.cuh"
#include "thrust/device_vector.h"
#include "../PiecewiseConstant/PiecewiseConstant.cuh"
#define _USE_MATH_DEFINES
#include <math.h>
#define __CUDA_ARCH__
#include "texture_indirect_functions.h"
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
        ENVIRONMENT_LIGHT,
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
    __device__ virtual float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const = 0;
    __device__ virtual float PDF(IntersectRecord rec, const float3 &wi)const { return 0.0f; };
    __device__ virtual float3 getLe(Ray &r) const{ return BLACK; };
    __device__ virtual float3 L(const float3 &r, IntersectRecord *rec = nullptr) const { return BLACK; }
};

class PointLight : public Light
{
public:
    CUDA_FUNC  PointLight() = default;
    CUDA_FUNC ~PointLight() = default;
    //CUDA_FUNC virtual float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const override;
    CUDA_FUNC PointLight(const float3 &position, const float3 &color);
    CUDA_FUNC float3 getPower(float3 bound_length = make_float3(0.0f, 0.0f, 0.0f)) const;
    __device__ float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;
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
    __device__ float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;

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
    __device__ float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const override;
    CUDA_FUNC bool hit(Ray &r, IntersectRecord &rec);
    __device__ float3 interpolatePosition(float3 tri_sample) const;
    CUDA_FUNC float area() const
    {
        return length(cross(pos[2] - pos[0], pos[1] - pos[0]));
    }

    __device__ float PDF(IntersectRecord rec, const float3 &wi)  const{
		if (fabs(dot(normal, wi)) < 0.001f)
			return 0.0f;
        return rec.t * rec.t / (area() * fabs(dot(normal , wi))); 
    };
    __device__ float3 L(const float3 &r, IntersectRecord *rec = nullptr) const
    {
        if (two_side)
            return illum;
        else return dot(r, normal) > 0 ? illum : BLACK;
    };
    __host__ bool setUpMaterial(material::MATERIAL_TYPE type, Material *mat);

    __device__ float3 getLe(Ray &r) const { return BLACK; };

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
    CUDA_FUNC EnvironmentLight(float* texture, int w, int h):height(h), width(w) {
        setUp(texture, w, h);
    };
    CUDA_FUNC EnvironmentLight() = default;
    CUDA_FUNC ~EnvironmentLight() = default;
    //CUDA_FUNC virtual float3 getDir(float3 pos = make_float3(0.0f, 0.0f, 0.0f), float2 sample = make_float2(0.0f, 0.0f)) const = 0;
    __device__ float3 lightIllumi(IntersectRecord &ref, Ray *wi, float2 sample = make_float2(0.0f, 0.0f)) const;
    CUDA_FUNC float3 getPower(float3 bound_length = make_float3(0.0f, 0.0f, 0.0f)) const
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    __device__ float PDF(IntersectRecord rec, const float3 &wi)const;
    __device__ float3 getLe(Ray &r) const;
    __device__ float3 L(const float3 &r, IntersectRecord *rec = nullptr) const;
    __host__ bool setUp(float *img, int width, int height);
//    __host__ bool initialize(char *img, int height, int width,)
    cudaTextureObject_t img;
private:
    int height, width;
    cudaArray_t array;
    Distribution2D *distribution;
};
