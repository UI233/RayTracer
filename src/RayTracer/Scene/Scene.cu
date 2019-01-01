#include "Scene.cuh"
//Brute force method
CUDA_FUNC bool Scene::hit(Ray &r, IntersectRecord &rec) const
{
    for (int i = 0; i < model::TYPE_NUM; i++)
    {
        switch (model::MODEL_TYPE(i))
        {
        case model::TRIAGNLE:
            for (int j = 0; j < model_sz[i]; j++)
                tri[j].hit(r, rec);
            break;
        case model::MESH:
            for (int j = 0; j < model_sz[i]; j++)
                mesh[j].hit(r, rec);
            break;
        case model::SPHERE:
            break;
        case model::CYLINDER:
            break;
        default:
            break;
        }
    }

    for (int i = light::TRIANGLE_LIGHT; i < light::TYPE_NUM; i++)
    {
        switch (light::LIGHT_TYPE(i))
        {
        case light::POINT_LIGHT:
            break;
        case light::DIR_LIGHT:
            break;
        case light::TRIANGLE_LIGHT:

            break;
        case light::TYPE_NUM:
            break;
        default:
            break;
        }
    }
    return true;
}

__device__ Light* Scene::sampleOneLight(curandState *state) const
{
    static unsigned int num, cnt;
    num = curand(state) % light_sz_all;
    cnt = 0;

    for (unsigned int i = 0; i < (unsigned int)light::TYPE_NUM; i++)
    {
        if (num >= light_sz[i])
            num -= light_sz[i];
        else
        {
            switch (light::LIGHT_TYPE(i))
            {
            case light::POINT_LIGHT:
                return pointl + num;
                break;
            case light::DIR_LIGHT:
                return dirl + num;
                break;
            case light::TRIANGLE_LIGHT:
                return tril + num;
                break;
            default:
                return nullptr;
                break;
            }
        }
    }

    return nullptr;
}

__host__ bool Scene::initializeScene(int light_size[], int model_size[], PointLight *point_light, 
    DirectionalLight *dir_light, TriangleLight *tri_light, Triangle *triangles, Mesh *meshes)
{
    cudaError_t error;
    
    error = cudaMalloc(&pointl, sizeof(PointLight) * light_size[0]);
    error = cudaMalloc(&dirl, sizeof(DirectionalLight) * light_size[1]);
    error = cudaMalloc(&pointl, sizeof(TriangleLight) * light_size[2]);
    //error = cudaMalloc(&pointl, sizeof(PointLight) * light_size[3]);
    
    error = cudaMalloc(&tri, sizeof(Triangle) * model_size[0]);
    error = cudaMalloc(&mesh, sizeof(Mesh) * model_size[1]);
      
    error = cudaMemcpy(pointl, point_light, sizeof(PointLight) * light_size[0], cudaMemcpyHostToDevice);
    error = cudaMemcpy(dirl, dir_light, sizeof(DirectionalLight) * light_size[1], cudaMemcpyHostToDevice);
    error = cudaMemcpy(tril, tri_light, sizeof(TriangleLight) * light_size[2], cudaMemcpyHostToDevice);

    error = cudaMemcpy(tri, triangles, sizeof(Triangle) * model_size[0], cudaMemcpyHostToDevice);
    error = cudaMemcpy(mesh, meshes, sizeof(Mesh) * model_size[1], cudaMemcpyHostToDevice);
    return error == cudaSuccess;
}

//Incompleted
CUDA_FUNC float3 Scene::getIllumi(Light *light, IntersectRecord &rec, float2 sample) const
{
    Ray r;
    float3 direction, color;
    float dis;
    color = light->lightIllumi(rec, &r, sample);
    dis = length(direction);
    
    return color;
}  