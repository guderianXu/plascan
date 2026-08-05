#version 440

layout(location = 0) in vec3 instanceCenter;
layout(location = 1) in vec3 instanceRight;
layout(location = 2) in vec3 instanceUp;
layout(location = 3) in vec3 instanceForward;
layout(location = 4) in vec4 instanceUvRect;

layout(std140, binding = 0) uniform CameraPlaneUniforms
{
    mat4 uMVP;
    mat4 uModelView;
    vec4 uViewportZoom;
} ubuf;

layout(location = 0) out vec2 uv;

float cameraHalfExtent()
{
    float depth = -(ubuf.uModelView * vec4(instanceCenter, 1.0)).z;
    float viewportHeight = max(ubuf.uViewportZoom.x, 1.0);
    float halfExtentPixels = ubuf.uViewportZoom.z;
    float perspectiveScale = 2.0 * tan(radians(22.5));
    return depth > 0.000001
        ? depth * perspectiveScale * halfExtentPixels / viewportHeight
        : ubuf.uViewportZoom.w;
}

void main()
{
    const vec2 corners[6] = vec2[6](
        vec2( 1.0,  1.0),
        vec2(-1.0,  1.0),
        vec2(-1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0));
    vec2 corner = corners[gl_VertexIndex];
    float halfExtent = cameraHalfExtent();
    vec3 worldPosition = instanceCenter
        + instanceRight * corner.x * halfExtent
        + instanceUp * corner.y * halfExtent * 0.68;
    uv = vec2(
        corner.x > 0.0 ? instanceUvRect.z : instanceUvRect.x,
        corner.y > 0.0 ? instanceUvRect.y : instanceUvRect.w);
    gl_Position = ubuf.uMVP * vec4(worldPosition, 1.0);
}
