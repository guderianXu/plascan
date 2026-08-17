#version 440

layout(location = 0) in vec3 position;

layout(std140, binding = 0) uniform ProjectedImageUniforms
{
    mat4 uMVP;
    mat4 uSourceView;
    vec4 intrinsics;
    vec4 imageGeometry;
    vec4 composition;
} ubuf;

layout(location = 0) out vec3 sourceCameraPosition;

void main()
{
    gl_Position = ubuf.uMVP * vec4(position, 1.0);
    sourceCameraPosition = (ubuf.uSourceView * vec4(position, 1.0)).xyz;
}
