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

layout(location = 0) out vec3 vColor;

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
    vec3 direction = normalize(instanceForward);
    vec3 worldPosition = gl_VertexIndex == 0
        ? instanceCenter
        : instanceCenter + direction * cameraHalfExtent() * instanceUvRect.w;
    vColor = instanceUvRect.xyz;
    gl_Position = ubuf.uMVP * vec4(worldPosition, 1.0);
}
