#pragma once
#include "../Scene/Scene.cuh"

__device__ float3 pathTracer(Ray r, Scene &scene, curandState *state);