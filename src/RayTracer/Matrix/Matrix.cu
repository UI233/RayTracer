#include "Matrix.cuh"
#include <cmath>

CUDA_FUNC mat4::mat4(const float &a)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            v[i][j] = 0.0f;
    v[0][0] = v[1][1] = v[2][2] = v[3][3] = a;
}

CUDA_FUNC mat4::mat4(
    const float &a11, const float &a12, const float &a13, const float &a14,
    const float &a21, const float &a22, const float &a23, const float &a24,
    const float &a31, const float &a32, const float &a33, const float &a34,
    const float &a41, const float &a42, const float &a43, const float &a44)
{
    v[0][0] = a11; v[0][1] = a12; v[0][2] = a13; v[0][3] = a14;
    v[1][0] = a21; v[1][1] = a22; v[1][2] = a23; v[1][3] = a24;
    v[2][0] = a31; v[2][1] = a32; v[2][2] = a33; v[2][3] = a34;
    v[3][0] = a41; v[3][1] = a42; v[3][2] = a43; v[3][3] = a44;
};

CUDA_FUNC mat4& mat4::operator *= (const mat4 &b)
{
    mat4 c;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            c.v[i][j] = 0;
            for (int m = 0; m < 4; m++)
                c.v[i][j] += v[i][m] * b.v[m][j];
            v[i][j] = c.v[i][j];
        }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            v[i][j] = c.v[i][j];

    return *this;
}

CUDA_FUNC mat4& mat4::operator *= (const float &b)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            v[i][j] *= b;
        }
    return *this;
}

CUDA_FUNC mat4& mat4::operator += (const mat4 &b)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            v[i][j] += b.v[i][j];
        }
    return *this;
}

CUDA_FUNC mat4& mat4::operator -= (const mat4 &b)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            v[i][j] -= b.v[i][j];
        }
    return *this;
}

CUDA_FUNC float4 mat4::operator()(const float4 &vec4) const
{
    float4 res;

    res.x = v[0][0] * vec4.x + v[0][1] * vec4.y + v[0][2] * vec4.z + v[0][3] * vec4.w;
    res.y = v[1][0] * vec4.x + v[1][1] * vec4.y + v[1][2] * vec4.z + v[1][3] * vec4.w;
    res.z = v[2][0] * vec4.x + v[2][1] * vec4.y + v[2][2] * vec4.z + v[2][3] * vec4.w;
    res.w = v[3][0] * vec4.x + v[3][1] * vec4.y + v[3][2] * vec4.z + v[3][3] * vec4.w;

    return res;
}

CUDA_FUNC float3 mat4::operator()(const float3 &vec3) const
{
    float4 vec4(make_float4(vec3, 1.0f));
    float4 res((*this)(vec4));

    return make_float3(res.x / res.w, res.y / res.w, res.z / res.w);
}

CUDA_FUNC Ray mat4::operator()(const Ray &r) const
{
    return Ray((*this)(r.getOrigin()), (*this)(r.getDir()));
}

CUDA_FUNC mat4 operator *(const mat4 &a, const mat4 &b)
{
    mat4 res = a;
    res *= b;
    return res;
}

CUDA_FUNC mat4 operator +(const mat4 &a, const mat4 &b)
{
    mat4 res = a;
    res += b;
    return res;
}

CUDA_FUNC mat4 operator -(const mat4 &a, const mat4 &b)
{
    mat4 res = a;
    res -= b;
    return res;
}


CUDA_FUNC mat4 operator *(const float &b, const mat4 &m)
{
    mat4 res = m;
    res *= b;
    return res;
}

CUDA_FUNC mat4 operator *(const mat4 &m, const float &b)
{
    return b * m;
}

//Compute the matrix representing the rotation using Rodrigues' rotation formula 
CUDA_FUNC mat4 rotation(const float &angle, const float3 &axis)
{
    //K is the matrix which represent the cross product with axis
    mat4 k(
        0.0f, -axis.z, axis.y, 0.0f,
        axis.z, 0.0f, -axis.x, 0.0f,
        -axis.y, axis.x, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
    return mat4(1.0f) + sin(angle) * k + (1 - cos(angle)) * k * k;
}

CUDA_FUNC mat4 translation(const float3 &offset)
{
    mat4 res(1.0f);
    res.v[0][3] = offset.x;
    res.v[1][3] = offset.y;
    res.v[2][3] = offset.z;

    return res;
}


CUDA_FUNC mat4 perspective(float fov, float near, float far)
{
    float inv_tan(1.0f / tan(fov));
    float inv_length(1.0f / (far - near));

    mat4 pers(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, far * inv_length, -far * near * inv_length,
        0.0f, 0.0f, 1.0f, 0.0f
    );

    return scale(make_float3(inv_tan, inv_tan, 1.0f)) * pers;
}


CUDA_FUNC mat4 scale(const float3 &offset)
{
    mat4 res(1.0f);

    res.v[0][0] = offset.x;
    res.v[1][1] = offset.y;
    res.v[2][2] = offset.z;

    return res;
}

CUDA_FUNC mat4 inverse(const mat4 &m)
{
    float factor00 = m.v[2][2] * m.v[3][3] - m.v[3][2] * m.v[2][3];
    float factor01 = m.v[2][1] * m.v[3][3] - m.v[3][1] * m.v[2][3];
    float factor02 = m.v[2][1] * m.v[3][2] - m.v[3][1] * m.v[2][2];
    float factor03 = m.v[2][0] * m.v[3][3] - m.v[3][0] * m.v[2][3];
    float factor04 = m.v[2][0] * m.v[3][2] - m.v[3][0] * m.v[2][2];
    float factor05 = m.v[2][0] * m.v[3][1] - m.v[3][0] * m.v[2][1];
    float factor06 = m.v[1][2] * m.v[3][3] - m.v[3][2] * m.v[1][3];
    float factor07 = m.v[1][1] * m.v[3][3] - m.v[3][1] * m.v[1][3];
    float factor08 = m.v[1][1] * m.v[3][2] - m.v[3][1] * m.v[1][2];
    float factor09 = m.v[1][0] * m.v[3][3] - m.v[3][0] * m.v[1][3];
    float factor10 = m.v[1][0] * m.v[3][2] - m.v[3][0] * m.v[1][2];
    float factor11 = m.v[1][1] * m.v[3][3] - m.v[3][1] * m.v[1][3];
    float factor12 = m.v[1][0] * m.v[3][1] - m.v[3][0] * m.v[1][1];
    float factor13 = m.v[1][2] * m.v[2][3] - m.v[2][2] * m.v[1][3];
    float factor14 = m.v[1][1] * m.v[2][3] - m.v[2][1] * m.v[1][3];
    float factor15 = m.v[1][1] * m.v[2][2] - m.v[2][1] * m.v[1][2];
    float factor16 = m.v[1][0] * m.v[2][3] - m.v[2][0] * m.v[1][3];
    float factor17 = m.v[1][0] * m.v[2][2] - m.v[2][0] * m.v[1][2];
    float factor18 = m.v[1][0] * m.v[2][1] - m.v[2][0] * m.v[1][1];

    mat4 Inverse;
    Inverse.v[0][0] = +(m.v[1][1] * factor00 - m.v[1][2] * factor01 + m.v[1][3] * factor02);
    Inverse.v[0][1] = -(m.v[1][0] * factor00 - m.v[1][2] * factor03 + m.v[1][3] * factor04);
    Inverse.v[0][2] = +(m.v[1][0] * factor01 - m.v[1][1] * factor03 + m.v[1][3] * factor05);
    Inverse.v[0][3] = -(m.v[1][0] * factor02 - m.v[1][1] * factor04 + m.v[1][2] * factor05);

    Inverse.v[1][0] = -(m.v[0][1] * factor00 - m.v[0][2] * factor01 + m.v[0][3] * factor02);
    Inverse.v[1][1] = +(m.v[0][0] * factor00 - m.v[0][2] * factor03 + m.v[0][3] * factor04);
    Inverse.v[1][2] = -(m.v[0][0] * factor01 - m.v[0][1] * factor03 + m.v[0][3] * factor05);
    Inverse.v[1][3] = +(m.v[0][0] * factor02 - m.v[0][1] * factor04 + m.v[0][2] * factor05);

    Inverse.v[2][0] = +(m.v[0][1] * factor06 - m.v[0][2] * factor07 + m.v[0][3] * factor08);
    Inverse.v[2][1] = -(m.v[0][0] * factor06 - m.v[0][2] * factor09 + m.v[0][3] * factor10);
    Inverse.v[2][2] = +(m.v[0][0] * factor11 - m.v[0][1] * factor09 + m.v[0][3] * factor12);
    Inverse.v[2][3] = -(m.v[0][0] * factor08 - m.v[0][1] * factor10 + m.v[0][2] * factor12);

    Inverse.v[3][0] = -(m.v[0][1] * factor13 - m.v[0][2] * factor14 + m.v[0][3] * factor15);
    Inverse.v[3][1] = +(m.v[0][0] * factor13 - m.v[0][2] * factor16 + m.v[0][3] * factor17);
    Inverse.v[3][2] = -(m.v[0][0] * factor14 - m.v[0][1] * factor16 + m.v[0][3] * factor18);
    Inverse.v[3][3] = +(m.v[0][0] * factor15 - m.v[0][1] * factor17 + m.v[0][2] * factor18);

    float determinant =
        +m.v[0][0] * Inverse.v[0][0]
        + m.v[0][1] * Inverse.v[0][1]
        + m.v[0][2] * Inverse.v[0][2]
        + m.v[0][3] * Inverse.v[0][3];

    if (fabs(determinant) <= 1e-5f)
        return mat4(0.0f);

    Inverse *= 1.0f / determinant;

    return Inverse;
}