#include "PathTracer.cuh"
#ifndef MAX_DEPTH
#define MAX_DEPTH 24
#endif // !MAX_DEPTH
#ifndef maxComp(vec)
#define maxComp(vec) ((vec).x > (vec).y ? ((vec).x > (vec).z ? (vec).x : (vec).z) : ((vec).y > (vec).z ? (vec).y : (vec).z))
#endif // !maxComp(x)


__device__ float3 pathTracer(Ray r, Scene &scene, StratifiedSampler<TWO> &sampler_scatter,  StratifiedSampler<TWO> &sampler_light,StratifiedSampler<ONE> &sampler_p,curandState *state)
{
    IntersectRecord rec;
    bool ishit, specular_bounce = false;
    int cnt_scatter = 0, cnt_light = 0, cnt_q = 0;//Count the number of samples for sampler

    float pdf;
    float3 le,wi;   
    float3 beta = make_float3(1.0f, 1.0f, 1.0f), rrbeta;
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
                    if (rec.light_type == light::TRIANGLE_LIGHT)
                    {
                        Light *light = scene.getIdxAreaLight(rec.lightidx);
                        if(light)
                            res += beta * light -> L(-r.getDir(), &rec);
                    }
                }
                //Debug
            }
            else
            {
                //Debug
                //ToDo : Add Light from environment, infinite area light for example
            }
        }

        if (!ishit || bounces > MAX_DEPTH)
            break;

        specular_bounce = rec.material_type & material::SPECULAR;
        //Sample one light to light the intersection point
        //won't sample for perferctly specular surface cause only the wi direction would be accounted 
        if (!specular_bounce)
        {
            sample_light = sampler_light(cnt_light++, state);
            sample_scatter = sampler_scatter(cnt_scatter++, state);
            le = scene.sampleOneLight(rec, sample_light, sample_scatter, sampler_p(cnt_q++) * scene.light_sz_all);
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

        beta *= le * fabs(dot(r.getDir(), rec.normal)) / pdf;
        r = rec.spawnRay(wi);

        //Russian roulette to determine whether to terminate the routine
        rrbeta = beta * etaScale;
        max_comp = maxComp(rrbeta);
        if (bounces > 3 && max_comp < 1.0f )
        {
            q = (1.0f - max_comp) > 0.5 ? (1.0f - max_comp) : 0.5;
            if (sampler_p(cnt_q++, state) < q)
                break;
            beta /= 1.0f - q;
        }
    }

    return res;
}