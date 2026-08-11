#version 440

layout(location = 0) in vec2 unitPosition;
layout(location = 1) in vec2 center;
layout(location = 2) in vec4 color;

layout(std140, binding = 0) uniform MatchOverlayUniforms
{
    mat4 projection;
} uniforms;

layout(location = 0) out vec4 vertexColor;

void main()
{
    const float radius = 4.0;
    gl_Position = uniforms.projection * vec4(center + unitPosition * radius, 0.0, 1.0);
    vertexColor = color;
}
