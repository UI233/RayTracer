#include "Camera.cuh"
#include <cstdio>

CUDA_FUNC Camera::Camera(const float3 &pos, const float3 &lookat, const float &fov, const float &near, const float &far, const int2 &res, const float3 &u) :
    pos(pos),
    up(normalize(u)),
    front(normalize(lookat - pos)),
    pers(perspective(fov, (float)resolution.y / resolution.x, near, far)),
    resolution(res)
{
    n = near;
    right = normalize(cross(front, up));
    up = normalize(cross(right, front));
}

CUDA_FUNC Ray Camera::generateRay(float x, float y)
{
    //float2 offset(sampler2D());
    //The direction of ray which computes the color of the pixel on (x,y)
    static const float step = 0.25f;
    float aspect = (float)resolution.x / resolution.y; 
    float3 dir = n * front + (float)x / resolution.x * right * step * aspect+ (float)y / resolution.y * up * step;
    return Ray(pos, normalize(dir));
}

CUDA_FUNC float2 Camera::getxy(const float3 &poso)
{
    static const float step = 0.25f;
    float3 dir = normalize(poso - pos);
    
    if (dot(dir, front) == 0.0f)
        return make_float2(0.0f, 0.0f);
    float aspect = (float)resolution.x / resolution.y;
    float t = n / dot(dir, front);
    float3 rpos =  t * dir;

    float2 res;
    res.x = dot(rpos, right) * resolution.x / step / aspect;
    res.y = dot(rpos, up) * resolution.y / step;

    return res;
}
//CUDA_FUNC Ray generateDifferentialRay();