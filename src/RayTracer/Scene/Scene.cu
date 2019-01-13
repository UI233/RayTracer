#include "Scene.cuh"

#ifndef isBlack(x)
#define isBlack(x) (length((x)) < 0.001f)
#endif // !isBlack(x)

//Brute force method
CUDA_FUNC bool Scene::hit(Ray &r, IntersectRecord &rec) const
{
    Quadratic q;
    Triangle a;
    Mesh t;
    for (int i = 0; i < (int)model::TYPE_NUM; i++)
    {
        switch (model::MODEL_TYPE(i))
        {
        case model::TRIAGNLE:
            for (size_t j = 0; j < model_sz[i]; j++)
            {
                a = tri[j];
                a.hit(r, rec);
            }
            break;
        case model::MESH:
            for (size_t j = 0; j < model_sz[i]; j++)
            { 
                t = mesh[j]; 
                t.hit(r, rec);
            }
            break;
        case model::Quadratic:
            for (size_t j = 0; j < model_sz[i]; j++)
            { 
                q = quad[j];
                q.hit(r, rec);
            }
            break;
        default:
            break;
        }
    }

    TriangleLight l;
    for (int i = light::TRIANGLE_LIGHT; i < light::TYPE_NUM; i++)
    {
        switch (light::LIGHT_TYPE(i))
        {
        case light::TRIANGLE_LIGHT:
            for (size_t j = 0; j < light_sz[i]; j++)
            {
                l = tril[j];
                if (l.hit(r, rec))
                {
                    rec.lightidx = j;
                    rec.light_type = light::TRIANGLE_LIGHT;
                }
            }
            break;
        case light::TYPE_NUM:
            break;
        default:
            break;
        }
    }

    return rec.t > 0.001f && rec.t < INF - 0.1f;
}

__device__ float3 Scene::sampleAllLight(IntersectRecord &rec,  curandStatePhilox4_32_10_t *state) const
{
    static int cntl = 0, cnts = 0;
    float3 res = BLACK;
    Light *obj = nullptr;
    PointLight pl;
    DirectionalLight dl;
    TriangleLight trl;
    EnvironmentLight e_l;
    int idx;
    bool isDelta; 
    float2 sample_light;
    float2 sample_surface;
    float4 tmp;
    for (unsigned int i = 0; i < (unsigned int)light::TYPE_NUM; i++)
    {
        for(int j = 0; j < light_sz[i];j++)
        {   
            tmp = curand_uniform4(state);
            sample_light = make_float2(tmp.x, tmp.y);
            sample_surface = make_float2(tmp.z, tmp.w);

            switch (light::LIGHT_TYPE(i))
            {
            case light::POINT_LIGHT:
                isDelta = true;
                pl =  pointl[j];
                obj = &pl;
                break;
            case light::DIR_LIGHT:
                isDelta = true;
                dl =  dirl[j];
                obj = &dl;
                break;
            case light::TRIANGLE_LIGHT:
                isDelta = false;
                trl =  tril[j];
                obj = &trl;
                idx = j;
                break;
            case light::ENVIRONMENT_LIGHT:
                isDelta = false;
                idx = j + light_sz[light::TRIANGLE_LIGHT];
                e_l = el[j];
                obj = &e_l;
                break;
            default:
                obj =  nullptr;
                break;
            }

            res += evaluateDirectLight(obj, rec, sample_light, sample_surface, idx, isDelta);
        }
    }
    return res;
}

__host__ bool Scene::initializeScene(int light_size[], 
    int model_size[], PointLight *point_light, 
    DirectionalLight *dir_light, TriangleLight *tri_light, EnvironmentLight *envl,
    Triangle *triangles, Mesh *meshes, 
    Quadratic *quadratic, int mat_type[], Material *mat)
{
    cudaError_t error;

    error = cudaMalloc(&pointl, sizeof(PointLight) * light_size[0]);
    error = cudaMalloc(&dirl, sizeof(DirectionalLight) * light_size[1]);
    error = cudaMalloc(&tril, sizeof(TriangleLight) * light_size[2]);
    error = cudaMalloc(&el, sizeof(EnvironmentLight) * light_size[3]);
    
    error = cudaMalloc(&tri, sizeof(Triangle) * model_size[0]);
    error = cudaMalloc(&mesh, sizeof(Mesh) * model_size[1]);
    error = cudaMalloc(&quad, sizeof(Quadratic) * model_size[2]);
    
    int idx = 0;
    for (int count = 0; count < model_size[0]; count++, idx++)
        triangles[count].setUpMaterial((material::MATERIAL_TYPE)mat_type[idx], mat + idx);
    for (int count = 0; count < model_size[1]; count++, idx++)
        meshes[count].setUpMaterial((material::MATERIAL_TYPE)mat_type[idx], mat + idx);
    for (int count = 0; count < model_size[2]; count++, idx++)
        quadratic[count].setUpMaterial((material::MATERIAL_TYPE)mat_type[idx], mat + idx);

    int k = light::TRIANGLE_LIGHT;
    for(int count = 0; count < light_size[k];count++,idx++)
        tri_light[count].setUpMaterial((material::MATERIAL_TYPE)mat_type[idx], mat + idx);

    error = cudaMemcpy(pointl, point_light, sizeof(PointLight) * light_size[0], cudaMemcpyHostToDevice);
    error = cudaMemcpy(dirl, dir_light, sizeof(DirectionalLight) * light_size[1], cudaMemcpyHostToDevice);
    error = cudaMemcpy(tril, tri_light, sizeof(TriangleLight) * light_size[2], cudaMemcpyHostToDevice);
    error = cudaMemcpy(el, envl, sizeof(EnvironmentLight) * light_size[3], cudaMemcpyHostToDevice);

    error = cudaMemcpy(tri, triangles, sizeof(Triangle) * model_size[0], cudaMemcpyHostToDevice);
    error = cudaMemcpy(mesh, meshes, sizeof(Mesh) * model_size[1], cudaMemcpyHostToDevice);
    error = cudaMemcpy(quad, quadratic, sizeof(Quadratic) * model_size[2], cudaMemcpyHostToDevice);

    model_sz_all = 0;
    
    light_sz_all = 0;
    light_power = BLACK;
    for (unsigned int i = 0; i < (unsigned int)light::TYPE_NUM; i++)
    {
        light_sz_all += light_size[i];
        light_sz[i] = light_size[i];
        int j;
        switch (light::LIGHT_TYPE(i))
        {
        case light::POINT_LIGHT:
            for (j = 0; j < light_sz[i]; j++)
                light_power += point_light[j].getPower();
            break;
        case light::DIR_LIGHT:
            for (j = 0; j < light_sz[i]; j++)
                light_power += dir_light[j].getPower(bound2 - bound1);
            break;
        case light::TRIANGLE_LIGHT:
            for (j = 0; j < light_sz[i]; j++)
                light_power += tri_light[j].getPower();
            break;
        case light::ENVIRONMENT_LIGHT:
            for (j = 0; j < light_sz[i]; j++)
                light_power += envl[j].getPower();
            break;
        default:
            break;
        }
    }

    for (int i = 0; i < (unsigned int)model::TYPE_NUM; i++)
    {
        model_sz_all += model_size[i];
        model_sz[i] = model_size[i];
    }
    return error == cudaSuccess;
}


__device__ float3 Scene::evaluateDirectLight(Light *light, IntersectRecord rec, float2 sample_light, float2 sample_BRDF, int idx, bool isDelta) const
{
    Ray r;
    float3 lpos;
    float3 color, res = BLACK;
    bool blocked = false;

    color = light->lightIllumi(rec, &r, sample_light);
    lpos = r.getOrigin();

    Material *this_material = rec.material;
    IntersectRecord light_rec;

    r = rec.spawnRay(-r.getDir());
    

    //Ray wo;
    float3 f = fabsf(dot(rec.normal, r.getDir())) * rec.f(-rec.wo.getDir(), r.getDir());
    light_rec.global_cam = rec.global_cam;
    if(!isBlack(f))
        if (hit(r, light_rec))
        {
            blocked = light_rec.lightidx == idx ? false : (length(light_rec.pos - r.getOrigin()) < length(r.getOrigin() - lpos) - 0.01f);
        }

    light_rec.wo = r;
    res = BLACK;

	if (rec.pdf_light < 0.001f )
		return BLACK;

    if (!isBlack(f) && !blocked && !isBlack(color))
    {  
        if (isDelta)
        {
            color =  color * f / rec.pdf_light;
        }
        else
        {
            rec.pdf_surface = rec.PDF(-rec.wo.getDir(), r.getDir());
			float weight = (PowerHeuristic(rec.pdf_light, rec.pdf_surface));
			weight /= rec.pdf_light;
			color = color * f *  weight;
        }
        res = color;
        //if (res.y < +0.0f)
        //    printf("2\n");
    }
    float3 wi;
    //Bugs!!!
    float weight = 1.0f;
    if (!isDelta)
    {
        f = rec.sample_f(-rec.wo.getDir(), &wi, &rec.pdf_surface, sample_BRDF);
        f *= fabs(dot(rec.normal, wi));
        r = rec.spawnRay(wi);

        rec.t = INF;
        rec.lightidx = -1;
        if (!isBlack(f) && rec.pdf_surface > 0.001f)
        {
            if (!this_material->isSpecular())
            {
                float pdf = light->PDF(rec, wi);
                if (fabs(pdf) < 0.01f)
                    return res;
                weight = PowerHeuristic(rec.pdf_surface, pdf);
            }

            float3 l = BLACK;
            if (hit(r, rec) && rec.lightidx == idx)
            {
                l = light -> L( -wi, &rec);
            }
            else
                l = light->getLe(r);

            res += l * f * weight / rec.pdf_surface;
        }
    }

    return res;
}