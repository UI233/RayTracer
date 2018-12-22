#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "../Ray/Ray.cuh"
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

using namespace std;
__host__ void mhr();
class Model {
public:
     virtual ~Model() = default;
     virtual bool hit(Ray r, IntersectRecord &colideRec) = 0;
};

class Triangle :public Model {

public:
     Triangle() = default;
     ~Triangle() = default;
     Triangle(float3 a, float3 b, float3 c, float3 norm[3]);
     Triangle(float3 p[3], float3 norm[3]);
     Triangle(const float3 p[3], const float3 norm[3]);
      bool hit(Ray r, IntersectRecord &colideRec);
     Triangle& operator =(const Triangle& plus);

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

      bool hit(Ray r, IntersectRecord &colideRec);
     int size() {
        return number;
    }
private:
    Triangle* meshTable;
    int number = 0;
};


#endif // !MODEL_H
