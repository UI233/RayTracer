#pragma once
#ifndef CAMERA_H
#define CAMERA_H
#include "Matrix/Matrix.cuh"
#include "../Ray/Ray.cuh"
#include "curand_kernel.h"
class Camera
{
public:
    CUDA_FUNC Camera() = default;
    CUDA_FUNC Camera(const float3 &pos, const float3 &lookat, const float &fov, const float &near, const float &far, const int2 &resolution, const float3 &u);
    CUDA_FUNC ~Camera() = default;
    CUDA_FUNC Ray generateRay(int x, int y);
    //CUDA_FUNC Ray generateDifferentialRay();
    CUDA_FUNC void update();

private:
    float3 pos;
    float3 up, front;
    mat4 raster2world;
    mat4 pers;
    int2 resolution;
};
#endif // !CAMERA_H