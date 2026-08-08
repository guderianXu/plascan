#version 440

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec3 vViewPosition;
layout(location = 3) in float vElevation;

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

vec3 hsvToRgb(vec3 hsv)
{
    vec3 p = abs(fract(hsv.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return hsv.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), hsv.y);
}

vec3 scalarRamp(float value, vec2 range)
{
    float span = range.y - range.x;
    float normalized = span > 0.0
        ? clamp((value - range.x) / span, 0.0, 1.0)
        : 0.5;
    return hsvToRgb(vec3((1.0 - normalized) * (2.0 / 3.0), 1.0, 0.92));
}

void main()
{
    int mode = int(ubuf.uRenderModeFlags.x + 0.5);
    if (mode == 3)
    {
        fragColor = vec4(vec3(54.0, 50.0, 94.0) / 255.0, 1.0);
        return;
    }

    vec3 baseColor = vColor;
    if (mode == 2)
    {
        baseColor = vec3(160.0, 156.0, 205.0) / 255.0;
    }
    else if (mode == 4)
    {
        baseColor = scalarRamp(vElevation, ubuf.uScalarRange.xy);
    }
    else if (ubuf.uRenderModeFlags.y < 0.5)
    {
        baseColor = vec3(239.0, 236.0, 224.0) / 255.0;
    }

    vec3 surfaceNormal = vNormal;
    if (mode == 2)
    {
        surfaceNormal = cross(dFdx(vViewPosition), dFdy(vViewPosition));
    }
    float normalLengthSquared = dot(surfaceNormal, surfaceNormal);
    vec3 normal = normalLengthSquared > 1.0e-20
        ? surfaceNormal * inversesqrt(normalLengthSquared)
        : vec3(0.0, 0.0, 1.0);
    vec3 lightDirection = normalize(ubuf.uLightDirPointSize.xyz);
    vec3 viewDirection = normalize(-vViewPosition);
    if (dot(normal, viewDirection) < 0.0)
    {
        normal = -normal;
    }
    float headDiffuse = max(dot(normal, viewDirection), 0.0);
    float keyDiffuse = max(dot(normal, lightDirection), 0.0);
    vec3 baseLinear = srgbToLinear(baseColor);
    vec3 shadedPalette = vec3(239.0, 236.0, 224.0) / 255.0;
    vec3 solidPalette = vec3(160.0, 156.0, 205.0) / 255.0;
    float neutralDistance = min(distance(baseColor, shadedPalette),
                                distance(baseColor, solidPalette));
    float isNeutralSurface = 1.0 - step(0.008, neutralDistance);
    float photoShape = 0.86 + 0.10 * keyDiffuse + 0.04 * headDiffuse;
    float neutralShape = 0.38 + 0.42 * keyDiffuse + 0.20 * headDiffuse;
    float shape = mix(photoShape, neutralShape, isNeutralSurface);
    fragColor = vec4(linearToSrgb(baseLinear * shape), 1.0);
}
