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
    CUDA_FUNC bool hit(Ray &r, IntersectRecord &rec) const;
    __device__ float3 sampleOneLight(IntersectRecord &rec, float2 sample_light, float2 sample_surface, int sample_num) const;
    //Load the scene to GPU
    __host__ bool initializeScene(int light_size[], int model_size[],  PointLight *pointl, DirectionalLight *dril
    , TriangleLight *tril, Triangle *tri, Mesh *mesh, Quadratic *qudratic, int material_type[], Material *mat);
    CUDA_FUNC float3 evaluateDirectLight(Light *light, IntersectRecord &ref, float2 sample_light = make_float2(0.0f, 0.0f), 
        float2 sample_BRDF = make_float2(0.0f, 0.0f)) const;

private:
    int light_sz[light::TYPE_NUM];
    int model_sz[model::TYPE_NUM];
    int light_sz_all, model_sz_all;
    float3 light_power;

    DirectionalLight *dirl;
    PointLight *pointl;
    TriangleLight *tril;

    Triangle *tri;
    Mesh *mesh;
    Quadratic *quad;
    float3 bound1, bound2;
};