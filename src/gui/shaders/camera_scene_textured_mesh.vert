#version 440

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aTexCoord;

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

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;
layout(location = 2) out vec2 vTexCoord;

void main()
{
    gl_Position = ubuf.uMVP * vec4(aPos, 1.0);
    vNormal = mat3(ubuf.uNormalMat) * aNormal;
    vColor = aColor;
    vTexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
