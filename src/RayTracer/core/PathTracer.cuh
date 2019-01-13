#pragma once
#include "../Scene/Scene.cuh"

__device__ float3 pathTracer(Ray r, Scene &scene, curandStatePhilox4_32_10_t *state, Camera *cam);