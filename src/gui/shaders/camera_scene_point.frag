#version 440

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec3 vViewPosition;
layout(location = 3) in vec2 vPointOffset;
layout(location = 4) flat in float vSelected;

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
    if (dot(vPointOffset, vPointOffset) > 1.0)
    {
        discard;
    }
    if (ubuf.uRenderModeFlags.z > 0.5)
    {
        if (vSelected < 0.5)
        {
            discard;
        }
        fragColor = vec4(vColor, 0.92);
        return;
    }

    float normalLengthSquared = dot(vNormal, vNormal);
    if (!(normalLengthSquared > 1.0e-20) || normalLengthSquared > 1.0e20)
    {
        fragColor = vec4(vColor, 1.0);
        return;
    }

    vec3 normal = vNormal * inversesqrt(normalLengthSquared);
    vec3 viewDirection = normalize(-vViewPosition);
    if (dot(normal, viewDirection) < 0.0)
    {
        normal = -normal;
    }
    vec3 lightDirection = normalize(ubuf.uLightDirPointSize.xyz);
    float headDiffuse = max(dot(normal, viewDirection), 0.0);
    float keyDiffuse = max(dot(normal, lightDirection), 0.0);
    float lightScale = 0.25 + 0.55 * keyDiffuse + 0.20 * headDiffuse;
    fragColor = vec4(linearToSrgb(srgbToLinear(vColor) * lightScale), 1.0);
}
