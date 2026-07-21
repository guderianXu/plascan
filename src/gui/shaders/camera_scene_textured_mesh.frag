#version 440

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec2 vTexCoord;

layout(std140, binding = 0) uniform SceneUniforms
{
    mat4 uMVP;
    mat4 uModelView;
    mat4 uNormalMat;
    vec4 uLightDirPointSize;
} ubuf;

layout(binding = 1) uniform sampler2D modelTexture;
layout(location = 0) out vec4 fragColor;

vec3 srgbToLinear(vec3 c)
{
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

vec3 linearToSrgb(vec3 c)
{
    return pow(clamp(c, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
}

void main()
{
    vec4 sampled = texture(modelTexture, vTexCoord);
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(ubuf.uLightDirPointSize.xyz);
    float diff = max(dot(n, lightDir), 0.0);
    vec3 litLinear = srgbToLinear(sampled.rgb) * (0.55 + 0.75 * diff);
    fragColor = vec4(linearToSrgb(litLinear), sampled.a);
}
