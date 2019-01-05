#define WIDTH 800
#define HEIGHT 600
#define WIDTH_PER_BLOCK  80
#define HEIGHT_PER_BLOCK  60
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

__constant__ Camera globalCam;
curandState *state;
float *data_tmp;
Scene *sce;
int *cnt;

int thread_num = 10 * 10 * 32;
__global__ void test(cudaSurfaceObject_t surface, Scene *scene, curandState *state, float* data_tmp, int *cnt);
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

    while (!glfwWindowShouldClose(window) && sta)
    {
        sta = renderScene(changed);
        //changed = false;
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

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
    test <<<block_size, 32>>> (texture_surface, sce, state, data_tmp, cnt);
    auto error = cudaDeviceSynchronize();
    //cudaDeviceSynchronize();
    display();

    //test << < WIDTH, HEIGHT >> > (texture_surface, 1.0f);
    //cudaDeviceSynchronize();
    //display();

    return error == cudaSuccess;
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

    int idx = (blockIdx.x * blockDim.y + blockIdx.y) * 32 + threadIdx.x;


    //To do: use time to update the seed
    curand_init(1774, idx,0, &state[idx]);

    if (threadIdx.x == 0)
    {
       sampler = StratifiedSampler<TWO_FOR_SHARED>(state + idx);
    }

    __syncthreads();
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

            //sampler_surface.regenerate(rstate);
           // sampler_light.regenerate(rstate);
           // p.regenerate(rstate);
        }
    __syncthreads();

    idx = 3 * (sty * WIDTH + stx);
    //Write to Texture
    if(threadIdx.x == 0)
        for (int i = 0; i < WIDTH_PER_BLOCK; i++)
            for (int j = 0; j < HEIGHT_PER_BLOCK; j++)
            {
                x = i + stx;
                y = j + sty;
                idx = y * WIDTH + x;
                idx = idx * 3;
                //Bug here: should take average value
                surf2Dwrite(make_float4(data_tmp[idx] / 32.0f, 
                    data_tmp[idx + 1] / 32.0f, data_tmp[idx + 2] / 32.0f, 1.0f), surface, sizeof(float4) * x , y);
            }


}

void test_for_initialize_scene()
{
    Scene scene;
    int lz[light::TYPE_NUM] = {1,0,0}, ms[model::TYPE_NUM] = {0,0,1};
    int mat_type[] = { material::LAMBERTIAN };
    Lambertian lamb(make_float3(0.7f, 0.8f, 0.4f));
    Material m(&lamb, material::LAMBERTIAN);
    Quadratic q(make_float3(0.3f, 0.0f, 0.0f), Sphere);
    q.setUpTransformation(
        mat4(1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f,1.0f, -10.0f,
            0.0f,0.0f,0.0f,1.0f)
    );
    PointLight pl(make_float3(-8.0f, 0.0f, 0.0f), make_float3(300.0, 350.0f, 300.0f));

    scene.initializeScene(
        lz, ms, &pl, nullptr, nullptr, nullptr, nullptr,
        &q, mat_type, &m
    );

    cudaMalloc(&sce, sizeof(Scene));
    cudaMemcpy(sce, &scene, sizeof(Scene), cudaMemcpyHostToDevice);
//    cas<<<1,1>>>(sce);
}