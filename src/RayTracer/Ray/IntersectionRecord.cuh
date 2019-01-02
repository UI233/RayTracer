#include "../Material/Material.cuh"
#ifndef INFINITY
#define INFINITY 1000000.0f
#endif // !INFINITY

class IntersectRecord
{
public:
    CUDA_FUNC IntersectRecord() = default;
    CUDA_FUNC ~IntersectRecord() = default;
    CUDA_FUNC IntersectRecord(const float3 &p, const float3 &n, const Ray &r, const float &dis = INFINITY) : pos(p), normal(n), t(dis), wo(r) {};

    float3 pos, normal;
    mat4 transformation;
    float t;
    Ray wo;
    float pdf_light, pdf_surface;
    Material *material;
    int material_type;
};