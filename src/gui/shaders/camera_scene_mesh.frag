#version 440

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec3 vViewPosition;

layout(std140, binding = 0) uniform SceneUniforms
{
    mat4 uMVP;
    mat4 uModelView;
    mat4 uNormalMat;
    vec4 uLightDirPointSize;
} ubuf;

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
    float normalLengthSquared = dot(vNormal, vNormal);
    vec3 n = normalLengthSquared > 1.0e-20
        ? vNormal * inversesqrt(normalLengthSquared)
        : vec3(0.0, 0.0, 1.0);
    vec3 lightDir = normalize(ubuf.uLightDirPointSize.xyz);
    vec3 fillDir = normalize(vec3(0.60, -0.15, 0.55));
    vec3 viewDir = normalize(-vViewPosition);
    if (dot(n, viewDir) < 0.0)
    {
        n = -n;
    }
    vec3 halfDir = normalize(lightDir + viewDir);
    float headDiffuse = max(dot(n, viewDir), 0.0);
    float keyDiffuse = max(dot(n, lightDir), 0.0);
    float diffuse = 0.15 * headDiffuse + 0.85 * keyDiffuse;
    float fillDiffuse = max(dot(n, fillDir), 0.0);
    float specular = pow(max(dot(n, halfDir), 0.0), 40.0);
    vec3 baseLinear = srgbToLinear(vColor);
    vec3 litLinear =
        baseLinear * (0.18 + 0.72 * diffuse + 0.08 * fillDiffuse)
        + vec3(0.025 * specular);
    fragColor = vec4(linearToSrgb(litLinear), 1.0);
}
