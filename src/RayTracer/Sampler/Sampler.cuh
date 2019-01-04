#include "curand_kernel.h"
#ifndef CUDA_FUNC
#define CUDA_FUNC __host__ __device__
#endif // !CUDA_FUNC

static const unsigned int maxn = 0xffffffffu;
enum SAMPLER_DIMENSION
{
    ONE,
    TWO,
    TWO_FOR_SHARED,
    ZERO
};

template<SAMPLER_DIMENSION>
class StratifiedSampler
{
public:
    CUDA_FUNC StratifiedSampler() = delete;
    CUDA_FUNC  StratifiedSampler(int sz, curandState *state) = delete;
    CUDA_FUNC ~StratifiedSampler() = delete;

    __device__ float operator()(int i, curandState *state) const = delete;
private:
    float *data;
    int size;
};


template<>
class StratifiedSampler<ONE>
{
public:
    CUDA_FUNC StratifiedSampler() :size(0), data(NULL) {}
    __device__ StratifiedSampler(unsigned int sz, curandState *state) : size(sz)
    {
        data = (float*)malloc(sz * sizeof(float));

        float step = 1.0f / sz;
        //Create Strarified samples
        for (int i = 0; i < sz; i++)
            data[i] = (i + (float)curand(state) / maxn) * step;
        
        int idx;
        float tmp;
        //random shuffle
        for (int i = 0; i < sz; i++)
        {
            idx = i + (int)(((float)curand(state) / maxn) * (sz - i));
            tmp = data[idx];
            data[idx] = i;
            data[i] = tmp;
        }
    }

    CUDA_FUNC ~StratifiedSampler()
    {
        if (data)
            free(data);
    }

    __device__ float operator()(int i, curandState *state = NULL) const
    {
        if (i >= 0 && i < size)
            return data[i];
        else
        {
            if (state)
                return (float)curand(state) / maxn;
            else return 0.0f;
        }
    }

private:
    float *data;
    int size;
};

template<>
class StratifiedSampler<TWO>
{
public:
    CUDA_FUNC StratifiedSampler():size(0), data(NULL) {}

    __device__ StratifiedSampler(unsigned int sz, curandState *state) : size(sz)
    {
        data = (float *)malloc(sizeof(float) * 2 * sz);

        float invSamples = 1.0f / sz;
        int idx = -1;
        float samplev;
        for (int i = 0; i < sz; i++)
        {
            samplev = (i + (float)curand(state) / maxn) * invSamples;
            data[++idx] = fminf(0.9999f, samplev);
            samplev = (i + (float)curand(state) / maxn) * invSamples;
            data[++idx] = fminf(0.9999f, samplev);
        }

        float temp;
        for (int i = 0; i < sz; i++)
        {
            idx = i + (int)(((float)curand(state) / maxn) * (sz - i));
            temp = data[2 * i];
            data[2 * i] = data[2 * idx];
            data[2 * idx] = temp;

            idx = i + (int)(((float)curand(state) / maxn) * (sz - i));
            temp = data[2 * i + 1];
            data[2 * i + 1] = data[2 * idx + 1];
            data[2 * idx + 1] = temp;
        }
    }

    CUDA_FUNC ~StratifiedSampler()
    {
        if (data)
            free(data);
    }

    __device__ float2 operator()(int i, curandState *state = NULL) const
    {
        if (i >= 0 && i < size)
            return make_float2(data[i * 2], data[i * 2 + 1]);
        else
        {
            if (state)
                return make_float2((float)curand(state) / maxn, (float)curand(state) / maxn);
            else return make_float2(0.0f, 0.0f);
        }
    }

private:
    float *data;
    int size;
};

template<>
class StratifiedSampler<TWO_FOR_SHARED>
{
public:
    CUDA_FUNC StratifiedSampler() :size(0) {}

    __device__ StratifiedSampler(unsigned int sz, curandState *state) : size(sz)
    {
        float invSamples = 1.0f / sz;
        int idx = -1;
        float samplev;
        for (int i = 0; i < sz; i++)
        {
            samplev = (i + (float)curand(state) / maxn) * invSamples;
            data[++idx] = fminf(0.9999f, samplev);
            samplev = (i + (float)curand(state) / maxn) * invSamples;
            data[++idx] = fminf(0.9999f, samplev);
        }

        float temp;
        for (int i = 0; i < sz; i++)
        {
            idx = i + (int)(((float)curand(state) / maxn) * (sz - i));
            temp = data[2 * i];
            data[2 * i] = data[2 * idx];
            data[2 * idx] = temp;

            idx = i + (int)(((float)curand(state) / maxn) * (sz - i));
            temp = data[2 * i + 1];
            data[2 * i + 1] = data[2 * idx + 1];
            data[2 * idx + 1] = temp;
        }
    }

    CUDA_FUNC ~StratifiedSampler() = default;

    __device__ float2 operator()(int i, curandState *state = NULL) const
    {
        if (i >= 0 && i < size)
            return make_float2(data[i * 2], data[i * 2 + 1]);
        else
        {
            if (state)
                return make_float2((float)curand(state) / maxn, (float)curand(state) / maxn);
            else return make_float2(0.0f, 0.0f);
        }
    }

private:
    float data[32];
    int size;
};
