#define HEIGHT 200
#define WIDTH 200
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
#include "Model/Model.cuh"
#include "Camera/Camera.cuh"
#include <glut/glfw3.h>
#include "Shader/myShader.h"

__global__ void test(cudaSurfaceObject_t surface);
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
    //loadModels();

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
    test <<< WIDTH, HEIGHT >>> (texture_surface);
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
    glUseProgram(prog);
    glBindTexture(GL_TEXTURE_2D, tex);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

__global__ void test(cudaSurfaceObject_t surface)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    int y = idx / WIDTH;
    int x = idx % WIDTH;

    float4 data = make_float4((float) x / WIDTH, (float) y / HEIGHT, 0.0f, 1.0f);


    surf2Dwrite(data, surface, x * sizeof(float4), y);
}
