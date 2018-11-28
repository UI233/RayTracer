#define HEIGHT 100
#define WIDTH 100
#define NUM 16
#define MAX_DEPTH 10
#define SAMPLE 4

#include <glut/gl3w.h>
#include <Windows.h>
#include <stdio.h>
#include "surface_functions.h"
#include <vector_functions.hpp>
#include <cuda_gl_interop.h>
#include <device_functions.h>
#include "Model/Model.cuh"
#include "Camera/Camera.cuh"
#include <glut/glfw3.h>
#include "Shader/myShader.h"

__global__ void test(cudaSurfaceObject_t surface, float time);
void display();
__device__ void computeTexture();
bool renderScene(bool);
GLuint initGL();
GLFWwindow* glEnvironmentSetup();
bool initCUDA(GLuint glTex);

GLuint tex;
GLuint prog;
cudaGraphicsResource *cudaTex;
cudaSurfaceObject_t texture_surface;

int main(int argc, char **argv)
{
    GLFWwindow *window = glEnvironmentSetup();
    bool changed = true, state = true;
    tex = initGL();
    initCUDA(tex);

    while (!glfwWindowShouldClose(window) && state)
    {
        state = renderScene(changed);
        //changed = false;
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

bool renderScene(bool changed)
{
    test <<< WIDTH, HEIGHT >>> (texture_surface, 0);
    auto error = cudaDeviceSynchronize();

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

    return error == cudaSuccess;
}

void display()
{
    static const float black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    glUseProgram(prog);
    glBindTexture(GL_TEXTURE_2D, tex);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

__global__ void test(cudaSurfaceObject_t surface, float time)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    int y = idx / WIDTH;
    int x = idx % WIDTH;

    float4 data = make_float4((float) x / WIDTH, (float) y / HEIGHT, time, 1.0f);
    surf2Dwrite(data, surface, x * sizeof(float4), y);
}

//void computeTexture()
//{
//    constexpr unsigned int block_x(2), block_y(3);
//    constexpr unsigned int threads_x(1), threads_y(1);
//
//    dim3 blocksNum(block_x, block_y);
//    dim3 threadsPerBlock(threads_x, threads_y);
//
//    curandState *state;
//    cudaError_t cudaStatus;
//    float3 *buffer;
//    cudaStatus = cudaSetDevice(0);
//    cudaStatus = cudaMalloc((void **)&state, sizeof(curandState) * 200 * 30);
//    cudaStatus = cudaMalloc((void**)&buffer, sizeof(float3) * WIDTH * HEIGHT);
//    launch << < blocksNum, threadsPerBlock >> > (state, buffer);
//
//    cudaStatus = cudaDeviceSynchronize();
//
//    //cudaStatus = cudaMemcpy(color_buffer, buffer, sizeof(float3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
//
//    //cudaFree(state);
//    //cudaFree(buffer);
//}//__device__ inline float shadowRay(float3 pos, curandState *state, int idx)
//{
//    float3 st = l1.sample(idx, SAMPLE, state);
//    Ray shadow_ray(
//        make_float3(pos.x - st.x, pos.y - st.y, pos.z - st.z),
//        st
//    );
//
//
//    float t0 = s0.hit(shadow_ray);
//    float t1 = s1.hit(shadow_ray);
//
//    if (t0 > TMIN && t0 < 1.0f - TMIN || t1 > TMIN && t1 < 1.0f - TMIN)
//        return 0.0f;
//    else return 5000.0f / (5000.0f + dot(shadow_ray.dir, shadow_ray.dir));
//}
//
//__device__ float3 getColor(Ray &r, curandState *state)
//{
//    float t0 = s0.hit(r);
//    float t1 = s1.hit(r);
//    float t2 = l1.hit(r);
//    float3 color = make_float3(1.0f, 1.0f, 1.0f);
//    
//    if (t2 > TMIN && (t2 <t1 || t1 <TMIN) && (t2 < t0 || t0 < TMIN))
//    {
//        return  color;
//    }
//    color = make_float3(0.0f, 0.0f, 0.0f);
//    float t;
//
//    if (t1 > TMIN && t0 > TMIN)
//        t = min(t0, t1);
//    else
//        t = max(t0, t1);
//
//    if (t < TMIN)
//        return color;
//    //else return make_float3(0.0f);
//
//    float3 pos = make_float3(r.pos.x + r.dir.x * t,
//        r.pos.y + r.dir.y * t,
//        r.pos.z + r.dir.z * t
//        );
//
//    float per_c = 1.0f / SAMPLE / SAMPLE;
//        for (int i = 0; i < SAMPLE * SAMPLE; ++i)
//        {
//            float pers = shadowRay(pos, state, i);
//            color.x += per_c * pers;
//            color.y += per_c * pers;
//            color.z += per_c * pers;
//        }
//
//    return color;
//}

//__global__ void launch(curandState *state, float3 *in)
//{
//    if (threadIdx.x == 0 && threadIdx.y == 0)
//    {
//        cas = Camera(WIDTH, HEIGHT, make_float3(0.25f, 3.0f, 20.5f));
//        l1 = Light(make_float3(-10.0f, 12.0f, -3.0f), make_float3(0.0f, 5.0f, 0.0f), make_float3(0.0f, 0.0f, -5.0f));
//        s0 = Sphere(make_float3(0.0f, -1006.0f, -5.0f), 1000.0f);
//        s1 = Sphere(make_float3(0.0f, 2.0f, -5.0f), 8.0f);
//        //printf("\nHello, world from the device!\n %d %d", blockIdx.x, blockIdx.y);
//    }
//
//    __syncthreads();
//
//    int i =blockIdx.x *  blockDim.x + threadIdx.x, 
//        j = blockIdx.y *  blockDim.y + threadIdx.y;
//
//    int s_id = j * 200 + i;
//    curand_init(312312,  s_id , 0, &state[s_id]);
//    i *= 8;
//    j *= 30;
//
//    int id = j * WIDTH + i;
//    Ray r;
//    float3 color;
//
//    for (int y = 0; y < 30; y++, j++, id += WIDTH)
//        for (int x = 0; x < 8; x++)
//        {
//            color = make_float3(0.0f);
//            for (int k = 0; k < SAMPLE; k++)
//            {
//                r = cas.generateRay(i + x + (float)curand(state) / MAXN, j + (float)curand(state) / MAXN);
//                float3 tmp = getColor(r, &state[s_id]);
//                color = make_float3(
//                    color.x + tmp.x / SAMPLE,
//                    color.y + tmp.y / SAMPLE,
//                    color.z + tmp.z / SAMPLE
//                );
//            }
//            in[id + x] = color;
//        }
//
//    
//    //color = make_float3(1.0f, 0.0f, 1.0f);
//}   