#define WIDTH 600
#define HEIGHT 450
#define WIDTH_PER_BLOCK  20
#define HEIGHT_PER_BLOCK  225
#define WIDTH_PER_THREADS 2
#define HEIGHT_PER_THREADS 5
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
#include "PostProcess/PostProcess.cuh"
#include <iostream>
#include <ctime>


curandState *state;
float *data_tmp;//store the rgb value for scene
int *cnt;//Store the times of rendering
GLuint tex;
GLuint prog;
cudaGraphicsResource *cudaTex;
cudaGraphExec_t exe;
cudaSurfaceObject_t texture_surface, surface_raw, surface_hdr;
cudaStream_t stream;


__constant__ Camera globalCam;

//Initialize the state of rand generator
__global__ void initial(curandState *state, int *time)
{
    int idx = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
    curand_init(idx + 332 ,idx, 0, state + idx);
}

__global__ void debug(cudaSurfaceObject_t surface, cudaSurfaceObject_t surfacew, Scene *scene, curandState *state, int *cnt)
{
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        ++(*cnt);
    __syncthreads();
    int stx = blockIdx.x * WIDTH_PER_BLOCK + threadIdx.x * WIDTH_PER_THREADS;
    int sty = blockIdx.y * HEIGHT_PER_BLOCK + threadIdx.y * HEIGHT_PER_THREADS;

    int idx = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
    curandState *rstate = state + idx;
    //curand_init(1234, idx, 0, rstate);//Produce many noises after initialization
    IntersectRecord rec;

    float3 px_color;
    Ray r;
    float frac1 = 1.0f / *cnt;
    float frac2 = frac1 * (*cnt - 1);
    float4 cas;
    for(int x = stx; x < stx +  WIDTH_PER_THREADS; x++)
        for (int y = sty; y < sty + HEIGHT_PER_THREADS; y++)
        {
            r = globalCam.generateRay(x - WIDTH / 2, y - HEIGHT / 2);
            px_color = pathTracer(r, *scene,  state + idx);
            //I don't know why this if-statement can let those strange white noises disappear...
            if (px_color.x > 1.0f && px_color.y > 1.0f && px_color.z > 1.0f)
            {
                printf("%f %f %f\n", px_color.x, px_color.y, px_color.z);
                printf("%d %d", x, y);
            }
            surf2Dread(&cas, surface, x * sizeof(float4), y);
            px_color = frac2 * make_float3(cas.x, cas.y, cas.z) + frac1 * px_color;
            surf2Dwrite(make_float4(px_color.x, px_color.y, px_color.z, 1.0f), surface, x * sizeof(float4), y);
        }
}

void display();
__device__ void computeTexture();
bool renderScene(bool);
GLuint initGL();
GLFWwindow* glEnvironmentSetup();
bool initCUDA(GLuint glTex);
void test_for_initialize_scene();
Scene *sce;

int main(int argc, char **argv)
{
    GLFWwindow *window = glEnvironmentSetup();
    bool changed = true, sta = true;
    tex = initGL();
    test_for_initialize_scene();
    initCUDA(tex);

    auto error = cudaDeviceSynchronize();
    while (!glfwWindowShouldClose(window) && sta && changed)
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
    cudaError_t error;

    cudaGraphLaunch(exe, stream);
    error = cudaStreamSynchronize(stream);

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
    cudaGraph_t renderProcess;
    //The information for kernel
    dim3 blockSize(WIDTH / WIDTH_PER_BLOCK, HEIGHT / HEIGHT_PER_BLOCK);
    dim3 threadSize(WIDTH_PER_BLOCK / WIDTH_PER_THREADS, HEIGHT_PER_BLOCK / HEIGHT_PER_THREADS);
    int thread_num = blockSize.x * blockSize.y * threadSize.x * threadSize.y;
    //Create the surface bound to OpenGL texture
    size_t heap_sz;
    auto error = cudaDeviceGetLimit(&heap_sz, cudaLimitMallocHeapSize);
    error = cudaGraphicsGLRegisterImage(&cudaTex, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    error = cudaGraphicsMapResources(1, &cudaTex, 0);

    cudaArray_t texArray;
    error = cudaGraphicsSubResourceGetMappedArray(&texArray, cudaTex, 0, 0);

    cudaResourceDesc dsc;
    dsc.resType = cudaResourceTypeArray;
    dsc.res.array.array = texArray;

    error = cudaCreateSurfaceObject(&texture_surface, &dsc);
    //Create the surface to Store temporary information
    cudaArray *arr;
    cudaChannelFormatDesc channel = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    error = cudaMallocArray(&arr, &channel, WIDTH, HEIGHT, cudaArraySurfaceLoadStore);
    dsc.res.array.array = arr;
    error = cudaCreateSurfaceObject(&surface_raw, &dsc);

    //Create the surface to Store temporary information
    channel = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    error = cudaMallocArray(&arr, &channel, WIDTH, HEIGHT, cudaArraySurfaceLoadStore);
    dsc.res.array.array = arr;
    error = cudaCreateSurfaceObject(&surface_hdr, &dsc);

    //Initialize the camera
    Camera cam(make_float3(0.0f, 0.0f, 2.0f), make_float3(0.0f, 0.3f, -1.0f), 2.0f, 0.10f, 1000.0f,
        make_int2(WIDTH / 2, HEIGHT / 2), make_float3(0.0f, 1.0f, 0.0f));
    //Malloc rand generator
    cudaMalloc(&state, sizeof(curandState) * thread_num * 2);
    int t = time(NULL);
    int *p;
    cudaMalloc(&p, sizeof(int));
    cudaMemcpy(p, &t, sizeof(int), cudaMemcpyHostToDevice);
    initial << <blockSize, threadSize >> > (state, p);
    error = cudaDeviceSynchronize();
    //Initialize camera
    error = cudaMemcpyToSymbol(globalCam, &cam, sizeof(Camera));

    /*error = cudaMalloc(&data_tmp, sizeof(float) * HEIGHT * WIDTH * 3);
    error = cudaMemset(data_tmp, 0, sizeof(float) * HEIGHT * WIDTH * 3);*/
    //Initialize the count
    error = cudaMalloc(&cnt, sizeof(int));
    error = cudaMemset(cnt, 0, sizeof(int));
    //Capture the stream to Create Graph
    cudaStreamCreate(&stream);
    //Capture the procedure for rendering to create cudaGraph
    //The procedure is render -> HDR -> filter
    cudaStreamBeginCapture(stream);
    debug <<<blockSize, threadSize, 0, stream>>> (surface_raw, texture_surface, sce, state, cnt);
    HDRKernel <<<blockSize, threadSize, 0, stream>>> (surface_hdr, surface_raw, WIDTH_PER_THREADS, HEIGHT_PER_THREADS, WIDTH_PER_BLOCK, HEIGHT_PER_BLOCK);
    filterKernel <<<blockSize, threadSize, 0, stream>>> (texture_surface, surface_hdr, WIDTH_PER_THREADS, HEIGHT_PER_THREADS, WIDTH_PER_BLOCK, HEIGHT_PER_BLOCK, WIDTH, HEIGHT);
    cudaStreamEndCapture(stream, &renderProcess);

    cudaGraphInstantiate(&exe, renderProcess, nullptr, nullptr, 0);
    return error == cudaSuccess;
}

void display()
{
    glUseProgram(prog);
    glBindTexture(GL_TEXTURE_2D, tex);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}



void test_for_initialize_scene()
{
    Scene scene;
    int lz[light::TYPE_NUM] = { 0,0,1 }, ms[model::TYPE_NUM] = { 0,0,2 };
    int mat_type[] = { material::LAMBERTIAN , material::LAMBERTIAN, material::LAMBERTIAN };
    Lambertian lamb(make_float3(0.7f, 0.8f, 0.4f)), lamb2(make_float3(1.0f, 0.0f, 0.0f)), lamb3(make_float3(1.0f, 1.0f, 1.0f));
    Material m(&lamb, material::LAMBERTIAN), c(&lamb2, material::LAMBERTIAN), cs(&lamb3, material::LAMBERTIAN);
    Material t[] = { m,c ,cs };

    TriangleLight trl(make_float3(0.0f, 10.3f, 2.0f),
        make_float3(2.0f, 0.7f, 3.0f),
        make_float3(0.0f, 0.0f, 3.0f), make_float3(8.0f, 8.0f, 8.0f), true);

    Quadratic q(make_float3(0.3f, 0.0f, 0.0f), Sphere);
    q.setUpTransformation(
        mat4(1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, -8.0f,
            0.0f, 0.0f, 0.0f, 1.0f)
    );

    Quadratic s(make_float3(1.0f, 0.0f, 0.0f), Sphere);
    s.setUpTransformation(
        mat4(1.0f, 0.0f, 0.0f, -1.0f,
            0.0f, 1.0f, 0.0f, 2.0f,
            0.0f, 0.0f, 1.0f, -4.0f,
            0.0f, 0.0f, 0.0f, 1.0f)
    );

    PointLight pl(make_float3(-8.0f, 0.0f, 0.0f), make_float3(233.7f, 33.8f, 77.7f));
    Quadratic m_a[] = { q,s };
    
    DirectionalLight disl(make_float3(0.0f, -1.0f, 0.0f), make_float3(5.0f, 5.0f, 5.0f));

    scene.initializeScene(
        lz, ms, &pl, &disl, &trl, nullptr, nullptr,
        m_a, mat_type, t
    );

    cudaMalloc(&sce, sizeof(Scene));
    auto error = cudaMemcpy(sce, &scene, sizeof(Scene), cudaMemcpyHostToDevice);
}
