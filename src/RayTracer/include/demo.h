#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "helper_math.h"
#define fabs(x) ((x) > 0.0f ? (x) : -(x))
#define TMIN 0.0001f
#define MAXN 0xffffffffu

class Ray
{
public:
    __host__ __device__ Ray(float3 d, float3 p);
    __host__ __device__ ~Ray() = default;
    __host__ __device__ Ray() = default;
    float3 dir;
    float3 pos;
 
    float2 padd;
};

class Camera
{
public:
    __host__ __device__ Camera(int width, int height, float3 center);
    __host__ __device__ ~Camera() = default;
    __host__ __device__ Camera() = default;
    __host__ __device__ Ray generateRay(float i, float j);
    float3 v;
private:
    float3 pos;//position of the camera
    float3 origin;//Left-button corner
    float3 u;

    float4 padd;
};

class Sphere
{
public:
    __host__ __device__  float hit(Ray &r);
    __host__ __device__ Sphere() = default;
    __host__ __device__ ~Sphere() = default;
    __host__ __device__ Sphere(float3 p, float r);
private:
    float3 pos;
    float R;
};


class Light
{
public:
    __host__ __device__ Light() = default;
    __host__ __device__ ~Light() = default;
    __host__ __device__ Light(float3 o, float3 x, float3 v);
    __host__ __device__ float hit(Ray &r);
    __device__ float3 sample(int idx, int all,curandState *state);
private:
    float3 origin;
    float3 u;
    float3 v;
    float3 normal;

    float4 padd;
};
