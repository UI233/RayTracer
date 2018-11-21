#include "Ray.h"

CUDA_FUNC Ray::Ray(const float3 &o, const float3 &r) :origin(o),direction(normalize(r)) {}

CUDA_FUNC inline float3 Ray::getPos(float t) const
{
    return origin + t * direction;
}

CUDA_FUNC inline float3 Ray::getDir() const
{
    return direction;
}

CUDA_FUNC inline float3 Ray::getOrigin() const
{
    return origin;
}