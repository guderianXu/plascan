#version 440

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

layout(std140, binding = 0) uniform SceneUniforms
{
    mat4 uMVP;
    mat4 uModelView;
    mat4 uNormalMat;
    vec4 uLightDirPointSize;
    vec4 uViewportSize;
} ubuf;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;
layout(location = 2) out vec3 vViewPosition;
layout(location = 3) out vec2 vPointOffset;

void main()
{
    const vec2 corners[6] = vec2[](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0));
    vec2 corner = corners[gl_VertexIndex];
    vec4 clipPosition = ubuf.uMVP * vec4(aPos, 1.0);
    vec2 viewportSize = max(ubuf.uViewportSize.xy, vec2(1.0));
    clipPosition.xy += corner * ubuf.uLightDirPointSize.w / viewportSize * clipPosition.w;
    gl_Position = clipPosition;
    vNormal = mat3(ubuf.uNormalMat) * aNormal;
    vColor = aColor;
    vViewPosition = (ubuf.uModelView * vec4(aPos, 1.0)).xyz;
    vPointOffset = corner;
}
