#include "PathTracer.cuh"
#ifndef MAX_DEPTH
#define MAX_DEPTH 12
#endif // !MAX_DEPTH
#ifndef maxComp(vec)
#define maxComp(vec) ((vec).x > (vec).y ? ((vec).x > (vec).z ? (vec).x : (vec).z) : ((vec).y > (vec).z ? (vec).y : (vec).z))
#endif // !maxComp(x)


__device__ float3 pathTracer(Ray r, Scene &scene,curandState *state)
{/*
    StratifiedSampler<TWO> sampler_light(8, state);
    StratifiedSampler<TWO> sampler_scatter(8, state);*/
    IntersectRecord rec;
    bool ishit, specular_bounce = false;
    int cnt_scatter = 0, cnt_light = 0, cnt_q = 0;//Count the number of samples for sampler

    float pdf;
    float3 le = BLACK,wi;   
    float3 beta = make_float3(1.0f, 1.0f, 1.0f), rrbeta = BLACK;
    float3 res = BLACK;
    float2 sample_light, sample_scatter;//Store the samples
    float q, max_comp;
    float etaScale = 1.0f;

    for (int bounces = 0; ; bounces++)
    {
        rec.t = INF;
        rec.lightidx = -1;
        rec.isLight = false;
        ishit = scene.hit(r, rec);
        rec.wo = r; 
        if(!bounces  || specular_bounce)
        {
            if (ishit)
            {
                if (rec.isLight)
                {
                    if (rec.light_type >= light::TRIANGLE_LIGHT)
                    {
                        Light *light = scene.getIdxAreaLight(rec.lightidx);
                        if (light)
                        {
                            if (rec.light_type == light::TRIANGLE_LIGHT)
                            {
                                TriangleLight l = *((TriangleLight*)light); 
                                res += beta * l.L(-r.getDir(), &rec);
                            }
                        }
                    }
                }
                //Debug
            }
            else
            {
                Light *tmp = scene.getEnvironmentLight();
                if (tmp)
                {
                    EnvironmentLight temp = *(EnvironmentLight *)tmp;
                    res += beta * temp.getLe(r);
                }
            }
        }

        if (!ishit || bounces > MAX_DEPTH)
            break;

        specular_bounce = rec.material_type & material::SPECULAR;
        //Sample one light to light the intersection point
        //won't sample for perferctly specular surface cause only the wi direction would be accounted 
        if (!specular_bounce)
        {
            le = scene.sampleAllLight(rec,  state);
            res += beta * le;
        }
        else if(rec.material -> isTrans())
        {
            //if refraction
            float t = rec.material->eta * rec.material->eta;
            etaScale = dot(-r.getDir(), rec.normal) < 0 ? 1.0f / t : t;
        }
        //use brdf to sample new direction
        le = rec.sample_f(-r.getDir(), &wi, &pdf, sample_scatter);

        beta *= le * fabs(dot(wi, rec.normal)) / pdf;
        r = rec.spawnRay(wi);

        //Russian roulette to determine whether to terminate the routine
        rrbeta = beta * etaScale;
        max_comp = maxComp(rrbeta);
        if (bounces > 3 && max_comp < 1.0f )
        {
            q = (1.0f - max_comp) > 0.5 ? (1.0f - max_comp) : 0.5;
            if (curand_uniform(state) < q)
                break;
            beta /= 1.0f - q;
        }
    }

    return res;
}