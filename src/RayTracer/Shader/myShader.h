#pragma once
#ifndef MY_SHADER
#define MY_SHADER

#include <fstream>
#include <glut\glcorearb.h>
#include <glut\gl3w.h>
using std::ifstream;

class Shader
{
    char *content;
    GLuint index;
public:
    Shader() :content(NULL), index(0) {}
    Shader(ifstream &in)
    {
        if (in.is_open())
        {
            in.seekg(0, in.end);
            int length = in.tellg();
            content = new char[length + 5];
            memset(content, 0, sizeof(content));
            in.seekg(0, in.beg);
            in.read(content, length);
            content[length] = 0;
        }
    }

    ~Shader()
    {
        if (content)
            delete[] content;
    }

    bool LoadFile(const char *cnt)
    {
        ifstream in(cnt);
        if (!in.is_open())
            return false;
        in.seekg(0, in.end);
        int length = in.tellg();
        in.seekg(0, in.beg);
        if (content == NULL)
            content = new char[length + 5];
        else
        {
            delete[] content;
            content = new char[length + 5];
        }
        memset(content, 0, sizeof(content));
        in.read(content, length);
        content[length] = 0;
        if (in)
        {
            in.close();
            return true;
        }
        else
        {
            in.close();
            return false;
        }
    }

    bool Load(GLenum flag, GLuint prog)
    {
        if (content == NULL || glIsProgram(prog) == GL_FALSE)
            return false;
        if (glIsShader(index))
            return false;
        index = glCreateShader(flag);
        glShaderSource(index, 1, &content, NULL);
        glCompileShader(index);
        char cas[500];
        glGetShaderInfoLog(index, 500, NULL, cas);
        printf("Complete?%s\n", cas);
        glAttachShader(prog, index);
        return true;
    }

    void Clear()
    {
        delete[] content;
        content = NULL;
        index = 0;
    }
};
#endif 