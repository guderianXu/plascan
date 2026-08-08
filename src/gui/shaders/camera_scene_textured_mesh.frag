#version 440

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
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

vec3 srgbToLinear(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2));
}

vec3 linearToSrgb(vec3 color)
{
    return pow(clamp(color, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
}

void main()
{
    vec3 textureColor = texture(modelTexture, vTexCoord).rgb;
    float hasVertexColor = step(0.0, min(vColor.r, min(vColor.g, vColor.b)));
    vec3 textureLinear = srgbToLinear(textureColor);
    vec3 vertexLinear = srgbToLinear(max(vColor, vec3(0.0)));
    float disagreement = length(textureLinear - vertexLinear);
    float textureWeight = mix(
        0.65,
        0.15,
        smoothstep(0.08, 0.35, disagreement));
    vec3 stabilizedLinear = mix(vertexLinear, textureLinear, textureWeight);
    vec3 outputLinear = mix(textureLinear, stabilizedLinear, hasVertexColor);
    fragColor = vec4(linearToSrgb(outputLinear), 1.0);
}
