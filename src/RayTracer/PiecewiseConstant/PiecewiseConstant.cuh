#pragma once
#include "cuda_runtime.h"
#include "thrust/binary_search.h"
#ifndef CUDA_FUNC
#define CUDA_FUNC __host__ __device__
#endif // !CUDA_FUNC


class Distribution
{
public:
    CUDA_FUNC Distribution(int num = 0, float *value = nullptr);
    CUDA_FUNC ~Distribution() = default;
    CUDA_FUNC float PDF(float sample) const;
    CUDA_FUNC float sample(const float &u, float *pdf, int *offset) const;
    CUDA_FUNC int getCount() { return n; }
    CUDA_FUNC bool load2Device();
    CUDA_FUNC float getfunInt() { return funInt; }
private:
    int n;
    float funInt;
    float *value;
    float *cdf;
};

class Distribution2D
{
public:
    CUDA_FUNC ~Distribution2D() = default;
    CUDA_FUNC Distribution2D(float *img = nullptr, int width = 0, int height = 0);
    CUDA_FUNC float PDF(const float2 &sample) const;
    CUDA_FUNC float2 sample(const float2 &u, float *pdf) const;
    CUDA_FUNC bool load2Device();
private: 
    Distribution *hmargin;
    Distribution *condition_w;
    int height, width;
    float funInt;
};
