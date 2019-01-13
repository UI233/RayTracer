#pragma once
#include "../Light/Light.cuh"
#include "../Sampler/Sampler.cuh"

#ifndef PowerHeuristic(x,y)
#define PowerHeuristic(x,y) ((x)*(x) / ((x) * (x) + (y) * (y)))
#endif // !PowerHeuristic(x,y)
class Scene
{
public:
    CUDA_FUNC  Scene() = default;
    CUDA_FUNC ~Scene() = default;
	__device__ bool hit(Ray &r, IntersectRecord &rec) const;
    __device__ float3 sampleAllLight(IntersectRecord &rec, curandStatePhilox4_32_10_t *state) const;
    //Load the scene to GPU
    __host__ bool initializeScene(int light_size[], int model_size[],  PointLight *pointl, DirectionalLight *dril
    , TriangleLight *tril, EnvironmentLight *envl, Triangle *tri, Mesh *mesh, Quadratic *qudratic, int material_type[], Material *mat);
    __device__ float3 evaluateDirectLight(Light *light, IntersectRecord ref, float2 sample_light = make_float2(0.0f, 0.0f), 
        float2 sample_BRDF = make_float2(0.0f, 0.0f), int idx = -2, bool isDelta = false) const;

    CUDA_FUNC Light* getIdxAreaLight(int idx)
    {
        if (idx >= 0 && idx < light_sz[light::TRIANGLE_LIGHT])
            return tril + idx;
        return  nullptr;
    }

    CUDA_FUNC Light * getEnvironmentLight()
    {
        return light_sz[light::ENVIRONMENT_LIGHT] == 0 ? nullptr : el;
    }
    int light_sz_all, model_sz_all;
private:
    size_t light_sz[light::TYPE_NUM];
    size_t model_sz[model::TYPE_NUM];
    float3 light_power;

    DirectionalLight *dirl;
    PointLight *pointl;
    TriangleLight *tril;
    EnvironmentLight *el;

    Triangle *tri;
    Mesh *mesh;
    Quadratic *quad;
    float3 bound1, bound2;
};