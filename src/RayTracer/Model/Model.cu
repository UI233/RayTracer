#include "Model.cuh"
__host__ __device__ Triangle::Triangle(float3 p[3], float3 norm) :normal(norm) {
    pos[0] = p[0];
    pos[1] = p[1];
    pos[2] = p[2];
}

__host__ __device__ Triangle::Triangle(float3 a, float3 b, float3 c, float3 norm) :normal(norm) {
    pos[0] = a;
    pos[1] = b;
    pos[2] = c;
}

__host__ __device__  bool  Triangle::hit(Ray r, IntersectRecord &colideRec) {

    colideRec.t = -1.0f;

    float3 ab, ac, ap, norm, e, qp;
    float t;
    ab = pos[1] - pos[0];
    ac = pos[2] - pos[0];
    qp = -r.getDir();
    norm = cross(ab, ac);
    float d = dot(qp, norm);
    if (d <= 0.0f) return false;
    ap = r.getOrigin() - pos[0];
    t = dot(ap, norm);
    if (t < 0.0f) return false;
    e = cross(qp, ap);
    float v = dot(ac, e);
    if (v < 0.0f || v > d) return false;
    float w = -dot(ab, e);
    if (w < 0.0f || v + w > d) return false;
    t /= d;

    colideRec.t = t;
    colideRec.normal = norm;
    colideRec.pos = r.getPos(t);
    return true;
}