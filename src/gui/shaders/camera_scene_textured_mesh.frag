#version 440

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vTexCoord;
layout(location = 3) flat in float vVertexColorFallback;
layout(location = 4) in vec3 vViewPosition;

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
    vec3 vertexColor = clamp(vColor, vec3(0.0), vec3(1.0));
    float fallbackWeight = step(0.5, vVertexColorFallback);
    vec3 baseColor = mix(textureColor, vertexColor, fallbackWeight);
    int mode = int(ubuf.uRenderModeFlags.x + 0.5);
    if (mode == 1)
    {
        float normalLengthSquared = dot(vNormal, vNormal);
        vec3 normal = normalLengthSquared > 1.0e-20
            ? vNormal * inversesqrt(normalLengthSquared)
            : vec3(0.0, 0.0, 1.0);
        vec3 viewDirection = normalize(-vViewPosition);
        if (dot(normal, viewDirection) < 0.0)
        {
            normal = -normal;
        }
        vec3 lightDirection = normalize(ubuf.uLightDirPointSize.xyz);
        float headDiffuse = max(dot(normal, viewDirection), 0.0);
        float keyDiffuse = max(dot(normal, lightDirection), 0.0);
        float shape = 0.72 + 0.20 * keyDiffuse + 0.08 * headDiffuse;
        baseColor = linearToSrgb(srgbToLinear(baseColor) * shape);
    }
    fragColor = vec4(baseColor, 1.0);
}
