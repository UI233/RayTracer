#ifndef SCENE_H
#define SCENE_H
#include "../Ray/Ray.cuh"
class Scene
{
public:
    CUDA_FUNC  Scene() = default;
    CUDA_FUNC ~Scene() = default;

    CUDA_FUNC bool hit(Ray &r, IntersectRecord &rec) const;

    float3 boundary_min, boundary_max;
};
#endif // !SCENE_H
