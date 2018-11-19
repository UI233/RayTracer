#include "demo.h"
__host__ __device__ Ray::Ray(float3 d, float3 p) : dir(d), pos(p) {};

__host__ __device__ Camera::Camera(int width, int height, float3 center) :
    pos(center)
{
    u = make_float3(4.0f / width, 0.0f, 0.0f);
    v = make_float3(0.0f, -3.0f / height, 0.0f);
    origin = make_float3(center.x - 2.0f, center.y + 1.5f, center.z - 1.5f);
}

__host__ __device__ Ray Camera::generateRay(float i, float j)
{
    float3 dir;
    dir.x = origin.x + i * u.x + j * v.x;
    dir.y = origin.y + i * u.y + j * v.y;
    dir.z = origin.z + i * u.z + j * v.z;

    dir.x = dir.x - pos.x;
    dir.y = dir.y - pos.y;
    dir.z = dir.z - pos.z;

    return Ray(dir, pos);
}

__host__ __device__ Light::Light(float3 o, float3 x, float3 y) : origin(o), u(x), v(y),
normal(normalize(cross(x, y))) {}
__host__ __device__ float Light::hit(Ray &r)
{
    if (fabs(dot(r.dir, normal)) < TMIN)
        return -1.0f;

    float t = dot(origin - r.pos, normal) / dot(normal, r.dir);

    if (t < TMIN)
        return -1.0f;

    float3 hit;

    hit.x = r.pos.x + r.dir.x * t - origin.x;
    hit.y = r.pos.y + r.dir.y * t - origin.y;
    hit.z = r.pos.z + r.dir.z * t - origin.z;


    float tx = dot(normalize(u), hit) / length(u);
    float ty = dot(normalize(v), hit) / length(v);

    if (tx > TMIN && tx <= 1.0f && ty > TMIN && ty <= 1.0f)
        return t;

    return -1.0f;
}

__device__ float3 Light::sample(int idx, int all, curandState *state)
{
    float3 res = origin;
    float x = (float)curand(state) / MAXN, y = (float)curand(state) / MAXN;
    x += (float)idx / all;
    y += (float)(idx % all);

    res.x += x * u.x + y * v.x;
    res.y += x * u.y + y * v.y;
    res.z += x * u.z + y * v.z;

    return res;
}


__host__ __device__ Sphere::Sphere(float3 p, float r) :pos(p), R(r) {}

__host__ __device__ float Sphere::hit(Ray &r)
{
    float3 p = r.pos;

    p.x -= pos.x;
    p.y -= pos.y;
    p.z -= pos.z;

    float a = dot(r.dir, r.dir);
    float b = 2 * dot(r.dir, p);
    float c = dot(p, p) - R * R;
    float delta = b * b - 4 * a * c;
    if (delta > 0)
    {
        float rest1 = (-b - sqrt(delta)) / 2 / a;
        if (rest1 > 200.0f)
            return -1.0f;
        if (rest1 < TMIN)
        {
            float rest2 = (-b + sqrt(delta)) / 2 / a;
            if (rest2 > 200.0f || rest2 < TMIN)
                return -1.0f;
            else
            {
                return rest2;
            }
        }
        else
            return rest1;
    }

    return -1.0f;
}