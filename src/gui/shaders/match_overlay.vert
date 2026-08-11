#version 440

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;

layout(std140, binding = 0) uniform MatchOverlayUniforms
{
    mat4 projection;
} uniforms;

layout(location = 0) out vec4 vertexColor;

void main()
{
    gl_Position = uniforms.projection * vec4(position, 0.0, 1.0);
    gl_PointSize = 8.0;
    vertexColor = color;
}
