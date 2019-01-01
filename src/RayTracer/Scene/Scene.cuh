#pragma once
#include "../Light/Light.cuh"
class Scene
{
public:
    CUDA_FUNC  Scene() = default;
    CUDA_FUNC ~Scene() = default;

    CUDA_FUNC bool hit(Ray &r, IntersectRecord &rec) const;

private:
};