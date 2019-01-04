#include "PathTracer.cuh"
#ifndef MAX_DEPTH
#define MAX_DEPTH 24
#endif // !MAX_DEPTH

float3 pathTracer(Ray r, Scene &scene, StratifiedSampler<TWO> sampler_scatter,  StratifiedSampler<TWO> sampler_light,curandState *state)
{
    IntersectRecord rec;
    bool ishit, specular_bounce = false;
    int cnt_scatter = 0, cnt_light = 0;//Count the number of samples for sampler

    float pdf;
    float3 le,wi;
    float3 beta = make_float3(1.0f, 1.0f, 1.0f);
    float3 res = BLACK;
    float2 sample_light, sample_scatter;//Store the samples

    float etaScale = 1.0f;
    
    for (int bounces = 0; ; bounces++)
    {
        ishit = scene.hit(r, rec);

        if(!bounces  || specular_bounce)
        {
            if (ishit)
            {
                if (rec.isLight)
                {
                    if (rec.light_type == light::TRIANGLE_LIGHT)
                    {
                        TriangleLight trl = *(TriangleLight*)rec.light;
                        res += beta * trl.getLe(-r.getDir(), &rec);
                    }
                }
            }
        }
        else
        {
            //Add Light from environment, infinite area light for example
        }

        if (!ishit || bounces > MAX_DEPTH)
            break;
        specular_bounce = rec.material_type & material::FRESNEL;
        //Sample one light to light the intersection point
        //won't sample for perferctly specular surface cause only the wi direction would be accounted 
        if (!specular_bounce)
        {
            sample_light = sampler_light(cnt_light++, state);
            sample_scatter = sampler_scatter(cnt_scatter++, state);
            le = scene.sampleOneLight(rec, sample_light, sample_scatter, curand(state));
            res += beta * le;
        }
        //use brdf to sample new direction
        le = rec.material->sample_f(-r.getDir(), &wi, &pdf, sample_scatter);
        beta *= le * fabs(dot(r.getDir(), rec.normal)) / pdf;
        r = rec.spawnRay(wi);

        //Russian roulette to determine whether to terminate the routine
        if (bounces > 3)
        {

        }
    }

    return res;
}