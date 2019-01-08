#include "PiecewiseConstant.cuh"

inline CUDA_FUNC float rgb2y_xyz(const float3 &rgb)
{
    return rgb.x + 4.5906f * rgb.y + 0.06007 * rgb.z;
}

CUDA_FUNC Distribution::Distribution(int num, float *v) : n(num)
{
    value = new float[n];
    memcpy(value, v, sizeof(float) * n);
    cdf = new float[n + 1];
    cdf[0] = 0.0f;
    for (int i = 1; i <= n; i++)
        cdf[i] = cdf[i - 1] + value[i - 1] / n;

    funInt = cdf[n];
    for (int i = 1; i <= n; i++)
        cdf[i] = funInt == 0 ? (float) i / n  : cdf[i] / funInt;
}

CUDA_FUNC float Distribution::sample(const float &u, float *pdf, int *offset) const
{
    int l = 0, r = n;
    int mid;
    while (r > l)
    {
        mid = l + (r - l) / 2;
        if (cdf[mid] > u)
            r = mid - 1;
        else l = mid + 1;
    }

    *offset = r;
    *pdf = value[r] / funInt;
    return cdf[r + 1] == cdf[r] ? r : r + (u - cdf[r]) / (cdf[r + 1] - cdf[r]);
}

CUDA_FUNC float Distribution::PDF(float sample) const
{
    return value[(int)sample] / (funInt == 0 ? n : funInt);
}

__host__ bool Distribution::load2Device()
{
    float *tmp_v = value;
    float *tmp_cdf = cdf;
    
    value = cdf = nullptr;
    cudaMalloc(&value, sizeof(float) * n);
    cudaMalloc(&cdf, sizeof(float) * (n + 1));

    auto error = cudaMemcpy(value, tmp_v, sizeof(float) * n, cudaMemcpyHostToDevice);
    error = cudaMemcpy(cdf, tmp_cdf, sizeof(float) * (n + 1), cudaMemcpyHostToDevice);
    
    free(tmp_v);
    free(tmp_cdf);
    return error == cudaSuccess;
}


CUDA_FUNC Distribution2D::Distribution2D(float *img, int w, int h):width(w), height(h)
{
    
    funInt = 0.0f;
    float *v_tmp = (float*)malloc(sizeof(float)* height);
    float *condition_tmp = (float *)malloc(sizeof(float) * width);
    float tmp;

    condition_w = (Distribution*)malloc(sizeof(Distribution) * height);
    for (int i = 0; i < height; i++)
    {
        //P[Height]
        float sum = 0.0f;
        //width phi[0, 2PI], theta [0, PI]
        for (int j = 0; j < width; j++)
        {
            int idx = 3 * (i * width + j);
            tmp = rgb2y_xyz(make_float3(img[idx], img[idx + 1], img[idx + 2]));
            condition_tmp[j] = tmp;
            funInt += tmp;
        }
        //p(phi | theta)
        condition_w[i] = Distribution(width, condition_tmp);
        v_tmp[i] = condition_w[i].getfunInt();
    }

    funInt /= width * height;
    hmargin = (Distribution*)malloc(sizeof(Distribution));
    //p(theta)
    *hmargin = Distribution(height, v_tmp);
    free(v_tmp);
    free(condition_tmp);
}

CUDA_FUNC float Distribution2D::PDF(const float2 &sample) const
{
    //sample.x corresponding to phi
    int i1 = condition_w[0].getCount() * sample.x;
    //while sample.y ... to theta
    int i2 = hmargin -> getCount() * sample.y;

    return condition_w[i1].PDF(i2)/ hmargin->getfunInt();
}

CUDA_FUNC float2 Distribution2D::sample(const float2 &u, float *pdf) const
{
    float pdfs[2];
    int v;
    float d1 = hmargin->sample(u.y, pdfs + 1, &v);
    float d0 = condition_w[v].sample(u.x, pdfs, &v);

    *pdf = pdfs[0] * pdfs[1];

    return make_float2(d0, d1);
}

CUDA_FUNC bool Distribution2D::load2Device()
{
    bool err;
    err = hmargin->load2Device();
    for (int i = 0; i < height; i++)
        condition_w[i].load2Device();

    Distribution *a = condition_w, *b = hmargin;

    auto error = cudaMalloc(&hmargin, sizeof(Distribution));
    error = cudaMalloc(&condition_w, height * sizeof(Distribution));
    
    cudaMemcpy(hmargin, b, sizeof(Distribution), cudaMemcpyHostToDevice);
    cudaMemcpy(condition_w, a, height * sizeof(Distribution), cudaMemcpyHostToDevice);

    free(a);
    free(b);
    return err & error == cudaSuccess;
}