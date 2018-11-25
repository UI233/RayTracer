#pragma once
#ifndef MATRIX_H
#define MATRIX_H
#include "Ray/Ray.cuh"
class mat4
{
public:
    CUDA_FUNC mat4() = default;
    CUDA_FUNC mat4(const float &a11);
    CUDA_FUNC mat4(
        const float &a11, const float &a12, const float &a13, const float &a14,
        const float &a21, const float &a22, const float &a23, const float &a24,
        const float &a31, const float &a32, const float &a33, const float &a34,
        const float &a41, const float &a42, const float &a43, const float &a44);
    CUDA_FUNC ~mat4() = default;

    CUDA_FUNC Ray operator()(const Ray &) const;
    CUDA_FUNC float3 operator()(const float3 &) const;
    CUDA_FUNC float4 operator()(const float4 &) const;
    CUDA_FUNC mat4& operator *=(const float &);
    CUDA_FUNC mat4& operator *=(const mat4 &);
    CUDA_FUNC mat4& operator +=(const mat4 &);
    CUDA_FUNC mat4& operator -=(const mat4 &);
    float v[4][4];
};

CUDA_FUNC mat4 operator *(const mat4 &, const mat4 &);
CUDA_FUNC mat4 operator +(const mat4 &, const mat4 &);
CUDA_FUNC mat4 operator -(const mat4 &, const mat4 &);
CUDA_FUNC mat4 operator *(const float &, const mat4 &);
CUDA_FUNC mat4 operator *(const mat4 &, const float &);

CUDA_FUNC mat4 inverse(const mat4 &);
//return the matrix representing rotating about given axis
CUDA_FUNC mat4 rotation(const float &angle, const float3 &axis);
//return the matrix representing tranlating for given offset
CUDA_FUNC mat4 translation(const float3 &offset);
CUDA_FUNC mat4 scale(const float3 &offset);
//perspective projection where the field of view is fov(rad)
CUDA_FUNC mat4 perspective(float fov, float near, float far);
#endif // !MATRIX_H


