#version 450 core
uniform sampler2D tex;
in vec2 ppos;
out vec4 fragColor;

void main()
{
    fragColor = texture(tex, ppos); 
}