#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "../Ray/Ray.cu"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "vector"
#include "cstring"
#include "cstdlib"
#include "string.h"
#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include "string"
#include "fstream"
#include "sstream"
#include "algorithm"
#include "assert.h"
using namespace std;
enum {
	Empty = 998244353,
	Occupy = 23333333
};

class Model {
public:
	__host__ __device__ virtual ~Model() = default;
	__host__ __device__ virtual bool hit(Ray r, IntersectRecord &colideRec) = 0;
};

class Triangle :public Model {

public:
	__host__ __device__ Triangle() = default;
	__host__ __device__ ~Triangle() = default;
	__host__ __device__ Triangle(float3 a, float3 b, float3 c, float3 norm[3]);
	__host__ __device__ Triangle(float3 p[3], float3 norm[3]);
	__host__ __device__ Triangle(const float3 p[3], const float3 norm[3]);
	__host__ __device__  bool hit(Ray r, IntersectRecord &colideRec);
	__host__ __device__ Triangle& operator =(const Triangle& plus);

private:
	float3 pos[3];
	float3 normal[3];
	float dummy[4];       //Make the memory cost for per Triangle a 2^n byte size
};

class Mesh : public Model {

public:
	__host__ __device__ Mesh() = default;
	__host__ __device__ ~Mesh() = default;
	//__host__ friend  bool readFile(Mesh *obj, char* path);
	
	__host__ __device__  bool hit(Ray r, IntersectRecord &colideRec);
	__host__ __device__ int size() {
		return number;
	}
private:
	Triangle* meshTable;
	int number = 0;
};

bool readFile(Mesh *obj, char* path);
#endif // !MODEL_H
