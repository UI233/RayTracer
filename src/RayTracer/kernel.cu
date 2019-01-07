#define WIDTH 280
#define HEIGHT 210
#define WIDTH_PER_BLOCK  40
#define HEIGHT_PER_BLOCK  30
#define NUM 16
#define MAX_DEPTH 10
#define SAMPLE 4
#define WARP_SIZE 32
#include <glut/gl3w.h>
#include <Windows.h>
#include <stdio.h>
#include "surface_functions.h"
#include <vector_functions.hpp>
#include <cuda_gl_interop.h>
#include <device_functions.h>
#include "core/PathTracer.cuh"
#include "Camera/Camera.cuh"
#include <glut/glfw3.h>
#include "Shader/myShader.h"
#include "core/PathTracer.cuh"
#include "Ray/Ray.cuh"
#include <iostream>
#include <ctime>

__constant__ Camera globalCam;
curandState *state;
float *data_tmp;
Scene *sce;
int *cnt;
float img[3 * WIDTH * HEIGHT];

int thread_num = WIDTH * HEIGHT;
__global__ void test(cudaSurfaceObject_t surface, Scene *scene, curandState *state, float* data_tmp, int *cnt);

__global__ void initial(curandState *state, int *time)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(idx , idx, 0, state + idx);
}

__global__ void debug(cudaSurfaceObject_t surface, Scene *scene,float *data_tmp, curandState *state, int *cnt)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        ++*cnt;

    __syncthreads();
    int x = threadIdx.x, y = blockIdx.x;
    int idx = y * blockDim.x + x;
    curandState *rstate = state + idx;
    //curand_init(1234, idx, 0, rstate);//Produce many noises after initialization
    IntersectRecord rec;

    StratifiedSampler<TWO> sampler_light(16, rstate);
    StratifiedSampler<TWO> sampler_surface(16, rstate);
    StratifiedSampler<ONE> p(8, rstate);
    Ray r = globalCam.generateRay(x - WIDTH / 2, y - HEIGHT / 2);
    float3 tmp = make_float3(1.0f, 1.0f, 1.0f);
    
    //if(x == WIDTH / 2 && y == HEIGHT / 2)
    tmp = pathTracer(r, *scene,sampler_light,sampler_surface,p ,state + idx);
    float frac1 = (float)(*cnt - 1) / *cnt, frac2 = 1.0f / *cnt;
    idx *= 3;
    data_tmp[idx] = frac1 * data_tmp[idx] + frac2 * tmp.x;
    data_tmp[idx + 1] = frac1 * data_tmp[idx + 1] + frac2 * tmp.y;
    data_tmp[idx + 2] = frac1 * data_tmp[idx + 2] + frac2 * tmp.z;

    surf2Dwrite(make_float4(data_tmp[idx], data_tmp[idx + 1], data_tmp[idx + 2], 1.0f), surface, x * sizeof(float4), y);
}

void display();
__device__ void computeTexture();
bool renderScene(bool);
GLuint initGL();
GLFWwindow* glEnvironmentSetup();
bool initCUDA(GLuint glTex);
void test_for_initialize_scene();

GLuint tex;
GLuint prog;
cudaGraphicsResource *cudaTex;
cudaSurfaceObject_t texture_surface;

int main(int argc, char **argv)
{
    GLFWwindow *window = glEnvironmentSetup();
    bool changed = true, sta = true;
    tex = initGL();
    initCUDA(tex);
    test_for_initialize_scene();
    //loadModels();
    int t = time(NULL);
    int *p;
    cudaMalloc(&p, sizeof(int));
    cudaMemcpy(p, &t, sizeof(int), cudaMemcpyHostToDevice);
    initial << <HEIGHT, WIDTH >> > (state, p);
    auto error = cudaDeviceSynchronize();
    while (!glfwWindowShouldClose(window) && sta && changed)
    {
        sta = renderScene(changed);
        //changed = false;
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    //for (int i = 0; i < 1; i++)
    //{
    //    renderScene(true);
    ////}
    //cudaMemcpy(img, data_tmp, 3 * WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    //FILE * cas = fopen("test.ppm", "w");
    //fprintf(cas, "P3\n%d %d\n255\n", WIDTH, HEIGHT);

    //for (int j = HEIGHT - 1; j >= 0; j--)
    //    for (int i = 0; i < WIDTH; i++)
    //    {
    //        int idx = (WIDTH * j + i) * 3;
    //        int r = img[idx] * 255, g = img[idx + 1] * 255, b = img[idx + 2] * 255;
    //        r = min(255, r);
    //        g = min(255, g);
    //        b = min(255, b);

    //        fprintf(cas, "%d %d %d\n", r, g, b);
    //        //idx += 3;
    //    }
    //fclose(cas);
    glfwDestroyWindow(window);
    glfwTerminate();
    cudaFree(state);
    cudaFree(data_tmp);
    return 0;
}

bool renderScene(bool changed)
{
    dim3 block_size;

    block_size.x = WIDTH / WIDTH_PER_BLOCK;
    block_size.y = HEIGHT / HEIGHT_PER_BLOCK;
    block_size.z = 1;
    //ToDo: the state should be an array
    cudaError_t error;
    //if (changed)
    //{
        debug << <HEIGHT, WIDTH >> > (texture_surface, sce, data_tmp,state, cnt);
        error = cudaDeviceSynchronize();
    //}
    display();

    int idx = 0;
    

    return true;
}

GLuint initGL()
{
    //The position of the quad which covers the full screen
    static float vertices[6][2] = {
        -1.0f, 1.0f,
        -1.0f, -1.0f,
        1.0f, 1.0f,
        1.0f, 1.0f,
        -1.0f, -1.0f,
        1.0f, -1.0f
    };
    GLuint tex;
    //initialize the empty texture
    //and set the parameter for it
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLuint buffer;
    GLuint vao;
    //Push the vertices information into the vertex arrayy
    glCreateBuffers(1, &buffer);
    glCreateVertexArrays(1, &vao);

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, NULL, NULL, nullptr);

    glEnableVertexAttribArray(1);

    //Initialize the OpenGL shaders and program
    prog = glCreateProgram();
    Shader vertex, frag;

    vertex.LoadFile("./Shader/texture.vert");
    frag.LoadFile("./Shader/texture.frag");
    vertex.Load(GL_VERTEX_SHADER, prog);
    frag.Load(GL_FRAGMENT_SHADER, prog);
    glLinkProgram(prog);
    glBindTexture(GL_TEXTURE_2D, 0);

    return tex;
}

GLFWwindow* glEnvironmentSetup()
{
    glfwInit();
    

    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "test", NULL, NULL);
    glfwMakeContextCurrent(window);

    gl3wInit();

    return window;
}

bool initCUDA(GLuint glTex)
{
    auto error = cudaGLSetGLDevice(0);
    error = cudaGraphicsGLRegisterImage(&cudaTex, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    error = cudaGraphicsMapResources(1, &cudaTex, 0);

    cudaArray_t texArray;
    error = cudaGraphicsSubResourceGetMappedArray(&texArray, cudaTex, 0, 0);

    cudaResourceDesc dsc;
    dsc.resType = cudaResourceTypeArray;
    dsc.res.array.array = texArray;

    error = cudaCreateSurfaceObject(&texture_surface, &dsc);
    
    Camera cam(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, -1.0f), 2.0f, 1.00f, 100.0f,
        make_int2(WIDTH / 2, HEIGHT / 2), make_float3(0.0f, 1.0f, 0.0f));

    cudaMalloc(&state, sizeof(curandState) * thread_num);
    error = cudaMemcpyToSymbol(globalCam, &cam, sizeof(Camera));

    error = cudaMalloc(&data_tmp, sizeof(float) * HEIGHT * WIDTH * 3);
    error = cudaMemset(data_tmp, 0, sizeof(float) * HEIGHT * WIDTH * 3);

    error = cudaMalloc(&cnt, sizeof(int));
    error = cudaMemset(cnt, 0, sizeof(int));
    return error == cudaSuccess;
}

void display()
{
    glUseProgram(prog);
    glBindTexture(GL_TEXTURE_2D, tex);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}


__global__ void test(cudaSurfaceObject_t surface, Scene *scene, curandState *state, float* data_tmp, int *cnt)
{
    __shared__ StratifiedSampler<TWO_FOR_SHARED> sampler;
    //if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    //    ++*cnt;
    __syncthreads();
    int idx = (blockIdx.x * blockDim.y + blockIdx.y) * 32 + threadIdx.x;


    //To do: use time to update the seed
    curand_init(3212, idx * 3,0, &state[idx]);

    if (threadIdx.x == 0)
    {
       sampler = StratifiedSampler<TWO_FOR_SHARED>(state + idx);
    }

    curandState *rstate = state + idx;

    Ray r;
    float3 tmp;
    int stx = blockIdx.x * WIDTH_PER_BLOCK, sty = blockIdx.y * HEIGHT_PER_BLOCK;

    StratifiedSampler<TWO> sampler_light(16, rstate);
    StratifiedSampler<TWO> sampler_surface(16, rstate);
    StratifiedSampler<ONE> p(8, rstate);

    float2 offset = make_float2(0.0f, 0.0f) ;
    int x, y;
    for(int i = 0; i < WIDTH_PER_BLOCK; i++)
        for (int j = 0; j < HEIGHT_PER_BLOCK; j++)
        {
            offset = sampler(threadIdx.x);
            r = globalCam.generateRay(i + stx + offset.x - WIDTH / 2, j + sty + offset.y - HEIGHT / 2);
            tmp = pathTracer(r, *scene, sampler_surface, sampler_light, p, rstate);

            idx = (j + sty) * WIDTH + i + stx;
            idx = idx * 3;
            data_tmp[idx] += tmp.x;
            data_tmp[idx + 1] += tmp.y;
            data_tmp[idx + 2] += tmp.z;
            
            sampler_surface.regenerate(rstate);
            sampler_light.regenerate(rstate);
            p.regenerate(rstate);
        }

    //__syncthreads();

    //idx = 3 * (sty * WIDTH + stx);
    //float weight = 1.0f / (32.0f * (*cnt));
    //Write to Texture
        //for (int i = threadIdx.x; i < WIDTH_PER_BLOCK; i+=32)
        //    for (int j = 0; j < HEIGHT_PER_BLOCK; j++)
        //    {
        //        x = i + stx;
        //        y = j + sty;
        //        idx = y * WIDTH + x;
        //        idx = idx * 3;
        //        
        //        //Bug here: should take average value
        //        surf2Dwrite(make_float4(data_tmp[idx] * weight,
        //            data_tmp[idx + 1] * weight, data_tmp[idx + 2] * weight, 1.0f), surface, sizeof(float4) * x , y);
        //    }
}

void test_for_initialize_scene()
{
    Scene scene;
    int lz[light::TYPE_NUM] = {1,0,0}, ms[model::TYPE_NUM] = {0,0,2};
    int mat_type[] = { material::LAMBERTIAN , material::LAMBERTIAN, material::LAMBERTIAN };
    Lambertian lamb(make_float3(0.7f, 0.8f, 0.4f)), lamb2(make_float3(0.8f, 0.0f, 0.0f)), lamb3(make_float3(1.0f, 1.0f, 1.0f));
    Material m(&lamb, material::LAMBERTIAN), c(&lamb2, material::LAMBERTIAN), cs(&lamb3, material::LAMBERTIAN);
    Material t[] = { m,c ,cs};
/*
    Triangle tria(
        make_float3(),
        );*/
    Quadratic q(make_float3(0.3f, 0.0f, 0.0f), Sphere);
    q.setUpTransformation(
        mat4(1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f,1.0f, -10.0f,
            0.0f,0.0f,0.0f,1.0f)
    );

    Quadratic s(make_float3(1.0f, 0.0f, 0.0f), Sphere);
    s.setUpTransformation(
        mat4(1.0f, 0.0f, 0.0f, -2.0f,
            0.0f, 1.0f, 0.0f, -3.0f,
            0.0f, 0.0f, 1.0f, -5.0f,
            0.0f, 0.0f, 0.0f, 1.0f)
    );

    PointLight pl(make_float3(-8.0f, 0.0f, 0.0f), make_float3(300.0, 350.0f, 300.0f));
    Quadratic m_a[2] = { q,s};

    scene.initializeScene(
        lz, ms, &pl, nullptr, nullptr, nullptr, nullptr,
        m_a, mat_type, t
    );

    cudaMalloc(&sce, sizeof(Scene));
    cudaMemcpy(sce, &scene, sizeof(Scene), cudaMemcpyHostToDevice);
//    cas<<<1,1>>>(sce);
}