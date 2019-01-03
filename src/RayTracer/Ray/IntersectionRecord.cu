#include "IntersectionRecord.cuh"
CUDA_FUNC float findFloatUp(float v)
{
    if (v > 0.0f && v == INFINITY)
        return v;
    if (v == -0.0f)
        return +0.0f;

    unsigned int *t = (unsigned int*)&v;

    if (v >= 0.0f)
        *t++;
    else *t--;

    return v;
}

CUDA_FUNC float findFloatDown(float v)
{
    if (v < 0.0f && v == INFINITY)
        return v;
    if (v == +0.0f)
        return -0.0f;

    unsigned int *t = (unsigned int*)&v;

    if (v <= 0.0f)
        *t--;
    else *t++;

    return v;
}

CUDA_FUNC float3 offsetFromPoint(float3 origin, float3 normal, float3 error_bound, float3 d)
{
    float3 n = make_float3(fabs(normal.x), fabs(normal.y), fabs(normal.z));
    float3 offset = dot(n, error_bound) * normal;

    if (dot(d, normal) < 0)
        offset = -offset;

    float3 po = origin + offset;

    po.x = po.x > 0.0f ? findFloatUp(po.x) : findFloatDown(po.x);
    po.y = po.y > 0.0f ? findFloatUp(po.y) : findFloatDown(po.y);
    po.z = po.z > 0.0f ? findFloatUp(po.z) : findFloatDown(po.z);

    return po;
}
