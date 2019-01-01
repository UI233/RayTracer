#pragma once
#include "../Light/Light.cuh"
class Scene
{
public:
    CUDA_FUNC  Scene() = default;
    CUDA_FUNC ~Scene() = default;
    CUDA_FUNC bool hit(Ray &r, IntersectRecord &rec) const;
    __device__ Light* sampleOneLight(curandState *state) const;
    //Load the scene to GPU
    __host__ bool initializeScene(int light_size[], int model_size[],  PointLight *pointl, DirectionalLight *dril
    , TriangleLight *tril, Triangle *tri, Mesh *mesh);
    CUDA_FUNC float3 getIllumi(Light *light, IntersectRecord &ref, float2 sample = make_float2(0.0f, 0.0f)) const;

private:
    int light_sz[light::TYPE_NUM];
    int model_sz[model::TYPE_NUM];
    int light_sz_all, model_sz_all;

    DirectionalLight *dirl;
    PointLight *pointl;
    TriangleLight *tril;

    Triangle *tri;
    Mesh *mesh;

};