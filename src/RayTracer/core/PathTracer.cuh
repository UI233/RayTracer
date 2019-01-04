#pragma once
#include "../Scene/Scene.cuh"

float3 pathTracer(Ray &r, Scene &scene, StratifiedSampler<TWO> sampler, curandState *state);