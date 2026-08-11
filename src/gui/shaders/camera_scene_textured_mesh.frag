#version 440

layout(location = 0) in vec3 vNormal;
layout(location = 2) in vec2 vTexCoord;

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

layout(binding = 1) uniform sampler2D modelTexture;
layout(location = 0) out vec4 fragColor;

void main()
{
    vec3 textureColor = texture(modelTexture, vTexCoord).rgb;
    fragColor = vec4(textureColor, 1.0);
}
