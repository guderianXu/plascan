#version 440

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in float aImageCount;

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
layout(location = 2) out vec3 vViewPosition;
layout(location = 3) out vec2 vPointOffset;
layout(location = 4) flat out float vSelected;

vec3 hsvToRgb(vec3 hsv)
{
    vec3 p = abs(fract(hsv.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return hsv.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), hsv.y);
}

vec3 scalarRamp(float value, vec2 range, bool reverse)
{
    float span = range.y - range.x;
    float normalized = span > 0.0
        ? clamp((value - range.x) / span, 0.0, 1.0)
        : 0.5;
    if (reverse)
    {
        normalized = 1.0 - normalized;
    }
    return hsvToRgb(vec3((1.0 - normalized) * (2.0 / 3.0), 1.0, 0.92));
}

void main()
{
    const vec2 corners[6] = vec2[](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0));
    vec2 corner = corners[gl_VertexIndex];
    vec4 clipPosition = ubuf.uMVP * vec4(aPos, 1.0);
    vec2 ndc = clipPosition.xy / max(clipPosition.w, 1.0e-20);
    vec4 selectionRect = ubuf.uScalarRange;
    if (ubuf.uRenderModeFlags.w > 0.5)
    {
        vSelected = 1.0;
    }
    else
    {
        vSelected = clipPosition.w > 0.0
            && ndc.x >= selectionRect.x && ndc.x <= selectionRect.z
            && ndc.y >= selectionRect.y && ndc.y <= selectionRect.w
            ? 1.0 : 0.0;
    }

    vec2 viewportSize = max(ubuf.uViewportSize.xy, vec2(1.0));
    float selectionScale = ubuf.uRenderModeFlags.z > 0.5
        ? (ubuf.uRenderModeFlags.w > 1.5 ? 1.50 : 1.35)
        : 1.0;
    clipPosition.xy += corner * ubuf.uLightDirPointSize.w * selectionScale
        / viewportSize * clipPosition.w;
    gl_Position = clipPosition;
    vNormal = mat3(ubuf.uNormalMat) * aNormal;

    int mode = int(ubuf.uRenderModeFlags.x + 0.5);
    if (ubuf.uRenderModeFlags.z > 0.5)
    {
        vColor = ubuf.uRenderModeFlags.w > 1.5
            ? vec3(1.0, 0.18, 0.38)
            : vec3(1.0, 0.90, 0.20);
    }
    else if (mode == 1)
    {
        vColor = scalarRamp(aPos.z, ubuf.uScalarRange.xy, false);
    }
    else if (mode == 2 && ubuf.uRenderModeFlags.y > 0.5)
    {
        vColor = scalarRamp(aImageCount, ubuf.uScalarRange.xy, true);
    }
    else
    {
        vColor = aColor;
    }
    vViewPosition = (ubuf.uModelView * vec4(aPos, 1.0)).xyz;
    vPointOffset = corner;
}
