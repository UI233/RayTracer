#version 450 core
layout(location = 1) in vec2 pos;
out vec2 ppos;

void main()
{
    gl_Position = vec4(pos, 0.0f, 1.0f);
    ppos = pos / 2.0f + vec2(0.5f, 0.5f);
}