#version 440

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

layout(std140, binding = 0) uniform SceneUniforms
{
    mat4 uMVP;
    mat4 uModelView;
    mat4 uNormalMat;
    vec4 uLightDirPointSize;
    vec4 uViewportSize;
    vec4 uRenderModeFlags;
    vec4 uScalarRange;
} ubuf;

layout(location = 0) out vec3 vColor;

void main()
{
    gl_Position = ubuf.uMVP * vec4(aPos, 1.0);
    gl_PointSize = ubuf.uLightDirPointSize.w;
    vColor = aColor;
}
