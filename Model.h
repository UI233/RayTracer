#pragma once
#include "helper_math.h"
#include "Ray.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

class Model {
public:
	__host__ __device__ virtual ~Model() = default;
	__device__ virtual bool hit(Ray r, float3 &colidePos, curandState *state) = 0;
};

class Triangle :public Model {
	
 public:
	__host__ __device__ Triangle() = default;
	__host__ __device__ ~Triangle() = default;
	__host__ __device__ Triangle(float3 a, float3 b, float3 c, float3 norm);
	__host__ __device__ Triangle(float3 p[3], float3 norm);
	__device__  bool hit(Ray r, float3 &colidePos, curandState *state);


private:
	float3 pos[3];
	float3 normal;
	float dummy[4];       //Make the memory cost for per Triangle a 2^n byte size
};