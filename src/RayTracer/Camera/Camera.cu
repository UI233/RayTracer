#include "Camera.cuh"
#include <cstdio>

CUDA_FUNC Camera::Camera(const float3 &pos, const float3 &lookat, const float &fov, const float &near, const float &far, const int2 &res, const float3 &u) :
    pos(pos),
    up(normalize(u)),
    front(normalize(lookat - pos)),
    pers(perspective(fov, (float)resolution.y / resolution.x, near, far)),
    resolution(res)
{
    float3 right = cross(front, up);

    mat4 world2raster(
        right.x, right.y, right.z, -pos.x,
         up.x, up.y, up.z,-pos.y,
        front.x, front.y, front.z, -pos.z,
        0.0f, 0.0f, 0.0f, 1.0f
    );

    mat4 temp = scale(make_float3((float)resolution.x, (float)resolution.y, 1.0f));

    world2raster = 
        temp * pers * world2raster;
    raster2world = inverse(world2raster);
}

CUDA_FUNC Ray Camera::generateRay(int x, int y, curandState *state)
{
    //float2 offset(sampler2D());
    //The direction of ray which computes the color of the pixel on (x,y)
    float3 sample_direction(raster2world(make_float3(x, y, 0.0f)));

    return Ray(pos, sample_direction);
}

//CUDA_FUNC Ray generateDifferentialRay();
CUDA_FUNC void Camera::update()
{
    float3 right = cross(front, up);

    mat4 world2raster(
        right.x, right.y, right.z, -pos.x,
        up.x, up.y, up.z, -pos.y,
        -front.x, -front.y, -front.z, -pos.z,
        0.0f, 0.0f, 0.0f, 1.0f
    );


    mat4 temp = scale(make_float3((float)resolution.x, (float)resolution.y, 1.0f));

    world2raster =
        temp * pers * world2raster;
    raster2world = inverse(world2raster);
}