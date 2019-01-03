#include "Scene.cuh"

#ifndef isBlack(x)
#define isBlack(x) (length((x)) < 0.001f)
#endif // !isBlack(x)

//Brute force method
CUDA_FUNC bool Scene::hit(Ray &r, IntersectRecord &rec) const
{
    for (int i = 0; i < model::TYPE_NUM; i++)
    {
        switch (model::MODEL_TYPE(i))
        {
        case model::TRIAGNLE:
            for (int j = 0; j < model_sz[i]; j++)
            {
                Triangle a = tri[j];
                a.hit(r, rec);
            }
            break;
        case model::MESH:
            for (int j = 0; j < model_sz[i]; j++)
            { 
                Mesh t = mesh[j]; 
                t.hit(r, rec);
            }
            break;
        case model::Quadratic:
            for (int j = 0; j < model_sz[i]; j++)
            {
                Quadratic q;
                q = quad[j];
                q.hit(r, rec);
            }
            break;
        default:
            break;
        }
    }

    for (int i = light::TRIANGLE_LIGHT; i < light::TYPE_NUM; i++)
    {
        switch (light::LIGHT_TYPE(i))
        {
        case light::TRIANGLE_LIGHT:
            for (int j = 0; j < light_sz[i]; j++)
            {
                TriangleLight l = tril[j];
                l.hit(r, rec);
            }
            break;
        case light::TYPE_NUM:
            break;
        default:
            break;
        }
    }

    return rec.t > 0 && rec.t < 100000.0f;
}

__device__ float3 Scene::sampleOneLight(IntersectRecord &rec, float2 sample_light, float2 sample_surface, int sample_num) const
{
    static unsigned int num, cnt;
    num = sample_num % light_sz_all;
    cnt = 0;

    Light *obj;
    PointLight pl;
    DirectionalLight dl;
    TriangleLight trl;

    for (unsigned int i = 0; i < (unsigned int)light::TYPE_NUM; i++)
    {
        if (num >= light_sz[i])
            num -= light_sz[i];
        else
        {
            switch (light::LIGHT_TYPE(i))
            {
            case light::POINT_LIGHT:
                pl =  pointl[num];
                obj = &pl;
                break;
            case light::DIR_LIGHT:
                dl =  dirl[num];
                obj = &dl;
                break;
            case light::TRIANGLE_LIGHT:
                trl =  tril[num];
                obj = &trl;
                break;
            default:
                obj =  nullptr;
                break;
            }
            break;
        }
    }

    return obj ? light_sz_all * inv_light_power * obj->getPower(bound2 - bound1) * evaluateDirectLight(obj, rec, sample_light, sample_surface) : make_float3(0.0f, 0.0f, 0.0f);

}

__host__ bool Scene::initializeScene(int light_size[], int model_size[], PointLight *point_light, 
    DirectionalLight *dir_light, TriangleLight *tri_light, Triangle *triangles, Mesh *meshes, Quadratic *quadratic)
{
    cudaError_t error;
    
    error = cudaMalloc(&pointl, sizeof(PointLight) * light_size[0]);
    error = cudaMalloc(&dirl, sizeof(DirectionalLight) * light_size[1]);
    error = cudaMalloc(&pointl, sizeof(TriangleLight) * light_size[2]);
    //error = cudaMalloc(&pointl, sizeof(PointLight) * light_size[3]);
    
    error = cudaMalloc(&tri, sizeof(Triangle) * model_size[0]);
    error = cudaMalloc(&mesh, sizeof(Mesh) * model_size[1]);
    error = cudaMalloc(&quad, sizeof(Quadratic) * model_size[2]);
      
    error = cudaMemcpy(pointl, point_light, sizeof(PointLight) * light_size[0], cudaMemcpyHostToDevice);
    error = cudaMemcpy(dirl, dir_light, sizeof(DirectionalLight) * light_size[1], cudaMemcpyHostToDevice);
    error = cudaMemcpy(tril, tri_light, sizeof(TriangleLight) * light_size[2], cudaMemcpyHostToDevice);

    error = cudaMemcpy(tri, triangles, sizeof(Triangle) * model_size[0], cudaMemcpyHostToDevice);
    error = cudaMemcpy(mesh, meshes, sizeof(Mesh) * model_size[1], cudaMemcpyHostToDevice);
    error = cudaMemcpy(quad, quadratic, sizeof(Quadratic) * model_size[2], cudaMemcpyHostToDevice);

    model_sz_all = 0;
    for (int i = 0; i < model::TYPE_NUM; i++)
        model_sz_all += model_size[i];
    
    light_sz_all = 0;
    inv_light_power = 0.0f;
    for (unsigned int i = 0; i < (unsigned int)light::TYPE_NUM; i++)
    {
        light_sz_all += light_sz[i];
        
        int j;
        switch (light::LIGHT_TYPE(i))
        {
        case light::POINT_LIGHT:
            for (j = 0; j < light_sz[i]; j++)
                inv_light_power += point_light[j].getPower();
            break;
        case light::DIR_LIGHT:
            for (j = 0; j < light_sz[i]; j++)
                inv_light_power += dir_light[j].getPower(bound2 - bound1);
            break;
        case light::TRIANGLE_LIGHT:
            for (j = 0; j < light_sz[i]; j++)
                inv_light_power += tri_light[j].getPower();
            break;
        default:
            break;
        }
    }

    inv_light_power = 1.0f / inv_light_power;

    return error == cudaSuccess;
}

//Incompleted
CUDA_FUNC float3 Scene::evaluateDirectLight(Light *light, IntersectRecord &rec, float2 sample_light, float2 sample_BRDF) const
{
    Ray r;
    float distance;
    float3 color, res;
    bool blocked = false;

    color = light->lightIllumi(rec, &r, sample_light);
    distance = length(rec.pos - r.getOrigin());


    Material *this_material = rec.material;

    IntersectRecord light_rec;
    hit(r, light_rec);

    Ray wo;
    float3 f = this_material -> f(rec.wo.getDir(), -r.getDir());

    if (light_rec.t < distance - 0.001f)
        blocked = true;

    res = BLACK;
    if (!isBlack(f) && !blocked)
    {
        if (light->isDelta)
        {
            color = fabs(dot(rec.normal, r.getDir())) * color * f / rec.pdf_light;
        }
        else
        {
            rec.pdf_surface = this_material->PDF(wo.getDir(), r.getDir());
            color = fabs(dot(rec.normal, r.getDir())) * color * f
                *  PowerHeuristic(rec.pdf_surface, rec.pdf_light) / rec.pdf_light;
        }
        res = color;
    }
    float3 wi;
    
    if (!light->isDelta)
    {
        f = this_material->sample_f(wo.getDir(), &wi, &rec.pdf_surface, sample_BRDF);
        f *= fabs(dot(rec.normal, wi));
        r = Ray(r.getOrigin(), wi);
        //r = rec.spawnRay(r);

        rec.t = 100000.0f;
        rec.light = nullptr;
        if (!isBlack(f) && rec.pdf_surface > 0.00001f)
        {
            float pdf = light->PDF(rec, wi);
            if (fabs(pdf) > 0.00001f)
            {
                float3 l;
                if (hit(r, rec))
                {
                    if (rec.light == (void*)light)
                        l = light ->getLe(Ray(r.getOrigin(), -wi) , &rec);
                    else
                    {
                        Light *a =(Light*) rec.light;
                        l = a->getLe(r, &rec);
                    }

                    res += l * f * PowerHeuristic(rec.pdf_surface, rec.pdf_light) / rec.pdf_surface;
                }
            }

            return res;
        }
    }

    return res;
}