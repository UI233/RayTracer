#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "../Ray/Ray.cuh"
#include "../Material/Object.cuh"
#include "../Matrix/Matrix.cuh"
#include "../Ray/IntersectionRecord.cuh"
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
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef FLOAT_EPISLON
#define FLOAT_EPISLON (1e-2f)
#endif
using namespace std;
//namespace model
namespace model
{    
    enum MODEL_TYPE
    {
        TRIAGNLE,
        MESH,
        Quadratic,
        TYPE_NUM
    };
}



class Model {
public:
     virtual ~Model() = default;
     Model() = default;
	 CUDA_FUNC virtual bool hit(Ray r, IntersectRecord &colideRec) = 0;
     CUDA_FUNC virtual float area() const = 0;
     __host__ bool setUpMaterial(material::MATERIAL_TYPE type, Material *);
	 __host__ bool setUpTransformation(mat4 Transformation) {
		 transformation = Transformation;
         return true;
	 }
protected:
    //
    material::MATERIAL_TYPE material_type;
    Material *my_material;
	mat4 transformation;
};

class Triangle :public Model {

public:
     Triangle() = default;
     ~Triangle() = default;
	 CUDA_FUNC Triangle(float3 a, float3 b, float3 c, float3 norm[3]);
	 CUDA_FUNC Triangle(float3 p[3], float3 norm[3]);
	 CUDA_FUNC Triangle(const float3 p[3], const float3 norm[3]);
	 CUDA_FUNC bool hit(Ray r, IntersectRecord &colideRec);
     CUDA_FUNC float area() const override;
     CUDA_FUNC Triangle& operator =(const Triangle& plus);
     CUDA_FUNC float3 interpolatePosition(float3 sample) const;
private:
    float3 pos[3];
    float3 normal[3];
    float dummy[4];       //Make the memory cost for per Triangle a 2^n byte size
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
    CUDA_FUNC float area() const override { return 0.0f; };
private:
    Triangle* meshTable;
    int number = 0;
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
    CUDA_FUNC float area() const override {
        return 0.0f;
    }
private:
	float3 coefficient;
	int type;
	float height;
	CUDA_FUNC float3 getCenter() const;
    CUDA_FUNC float getRadius() const;
};

#endif // !MODEL_H
