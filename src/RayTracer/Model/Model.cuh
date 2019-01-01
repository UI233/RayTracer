#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "../Ray/Ray.cuh"
#include "../Matrix/Matrix.cuh"
#include "helper_math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include <cstdio>
#include "vector"
#include <fstream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#ifndef FLOAT_EPISLON
#define FLOAT_EPISLON (1e-4)
#endif
using namespace std;
//namespace model
namespace model
{    
    enum MODEL_TYPE
    {
        TRIAGNLE,
        MESH,
        SPHERE,
        CYLINDER,
        TYPE_NUM
    };
}

class Model {
public:
     virtual ~Model() = default;
	 CUDA_FUNC virtual bool hit(Ray r, IntersectRecord &colideRec) = 0;
};

class Triangle :public Model {

public:
     Triangle() = default;
     ~Triangle() = default;
	 CUDA_FUNC Triangle(float3 a, float3 b, float3 c, float3 norm[3]);
	 CUDA_FUNC Triangle(float3 p[3], float3 norm[3]);
	 CUDA_FUNC Triangle(const float3 p[3], const float3 norm[3]);
	 CUDA_FUNC bool hit(Ray r, IntersectRecord &colideRec);
     Triangle& operator =(const Triangle& plus);
     CUDA_FUNC float3 interpolatePosition(float3 sample);

private:
    float3 pos[3];
    float3 normal[3];
    float dummy[4];       //Make the memory cost for per Triangle a 2^n byte size
	mat4 transformation;
};

class Mesh : public Model {

public:
     Mesh() = default;
     ~Mesh() = default;
    __host__ bool readFile(char* path);

	CUDA_FUNC bool hit(Ray r, IntersectRecord &colideRec);
	CUDA_FUNC int size() {
        return number;
    }
private:
    Triangle* meshTable;
    int number = 0;
	mat4 transformation;
};

enum quadraticType {
	Sphere,
	Cylinder
};
class Quadratic : public Model {

public:
	Quadratic() = default;
	~Quadratic() = default;
	CUDA_FUNC Quadratic(float3 Coefficient,int Type);
	CUDA_FUNC bool setHeight(float Height);
	CUDA_FUNC bool hit(Ray r, IntersectRecord &colideRec);

private:
	float3 coefficient;
	int type;
	float height;
	mat4 transformation;
	CUDA_FUNC float3 getCenter();
	CUDA_FUNC float getRadius();
};

#endif // !MODEL_H
