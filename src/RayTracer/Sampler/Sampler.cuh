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
    CUDA_FUNC  StratifiedSampler(int sz, curandState *state) = 0;
    CUDA_FUNC ~StratifiedSampler() = delete;

    __device__ float operator()(int i, curandState *state) const = 0;

    float *data;
private:
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
            data[i] = (i + curand_uniform(state)) * step;
        
        int idx;
        float tmp;
        //random shuffle
        for (int i = 0; i < sz; i++)
        {
            idx = i + (int)((curand_uniform(state) / maxn) * (sz - i));
            idx %= sz;
            tmp = data[idx];
            data[idx] = i;
            data[i] = tmp;
        }
    }

    CUDA_FUNC ~StratifiedSampler() = default;

    __device__ float operator()(int i, curandState *state = NULL) const
    {
        if (i >= 0 && i < size)
            return data[i];
        else
        {
            if (state)
                return curand_uniform(state);
            else return 0.0f;
        }
    }

    __device__  void regenerate(curandState *state)
    {
        float step = 1.0f / size;
        //Create Strarified samples
        for (int i = 0; i < size; i++)
            data[i] = (i + curand_uniform(state)) * step;

        int idx;
        float tmp;
        //random shuffle
        for (int i = 0; i < size; i++)
        {
            idx = i + (int)(curand_uniform(state) * (size - i));
            idx %= size;
            tmp = data[idx];
            data[idx] = i;
            data[i] = tmp;
        }
    }
    float *data;
private:
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
            samplev = (i + curand_uniform(state) ) * invSamples;
            data[++idx] = fminf(0.9999f, samplev);
            samplev = (i + curand_uniform(state)) * invSamples;
            data[++idx] = fminf(0.9999f, samplev);
        }

        float temp;
        for (int i = 0; i < sz; i++)
        {
            idx = i + (int)((curand_uniform(state)) * (sz - i));
            idx %= sz;
            temp = data[2 * i];
            data[2 * i] = data[2 * idx];
            data[2 * idx] = temp;

            idx = i + (int)((curand_uniform(state) ) * (sz - i));
            idx %= sz;
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
                return make_float2(curand_uniform(state), curand_uniform(state));
            else return make_float2(0.0f, 0.0f);
        }
    }

    __device__ void regenerate(curandState *state)
    {
        float invSamples = 1.0f / size;
        int idx = -1;
        float samplev;
        for (int i = 0; i < size; i++)
        {
            samplev = (i + curand_uniform(state)) * invSamples;
            data[++idx] = fminf(0.9999f, samplev);
            samplev = (i + curand_uniform(state) ) * invSamples;
            data[++idx] = fminf(0.9999f, samplev);
        }

        float temp;
        for (int i = 0; i < size; i++)
        {
            idx = i + (int)(curand_uniform(state) * (size - i));
            idx %= size;
            temp = data[2 * i];
            data[2 * i] = data[2 * idx];
            data[2 * idx] = temp;

            idx = i + (int)((curand_uniform(state) * (size - i)));
            idx %= size;
            temp = data[2 * i + 1];
            data[2 * i + 1] = data[2 * idx + 1];
            data[2 * idx + 1] = temp;
        }
    }
    float *data;
private:
    int size;
};

template<>
class StratifiedSampler<TWO_FOR_SHARED>
{
public:
    CUDA_FUNC StratifiedSampler() :size(0) {}

    __device__ StratifiedSampler( curandState *state) : size(32)
    {
        float invSamples = 1.0f / 32.0f;
        int idx = -1;
        float samplev;
        for (int i = 0; i < 32; i++)
        {
            samplev = (i + curand_uniform(state) * invSamples);
            data[++idx] = fminf(0.9999f, samplev);
            samplev = (i + curand_uniform(state) * invSamples);
            data[++idx] = fminf(0.9999f, samplev);
        }

        float temp;
        for (int i = 0; i < 32; i++)
        {
            idx = i + (int)((curand_uniform(state) * (32 - i)));
            idx %= 32;
            temp = data[2 * i];
            data[2 * i] = data[2 * idx];
            data[2 * idx] = temp;

            idx = i + (int)((curand_uniform(state) * (32 - i)));
            idx %= 32;
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
                return make_float2(curand_uniform(state), curand_uniform(state));
            else return make_float2(0.0f, 0.0f);
        }
    }

    float data[32];
private:
    int size;
};
