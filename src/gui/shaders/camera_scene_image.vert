#version 440

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

layout(std140, binding = 0) uniform ImagePlaneUniforms
{
    mat4 uMVP;
} ubuf;

layout(location = 0) out vec2 uv;

void main()
{
    uv = texCoord;
    gl_Position = ubuf.uMVP * vec4(position, 1.0);
}
