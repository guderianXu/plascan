#version 440

layout(binding = 1) uniform sampler2D imageTexture;

layout(std140, binding = 0) uniform ImagePlaneUniforms
{
    mat4 uMVP;
    vec4 composition;
} ubuf;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

void main()
{
    vec4 imageColor = texture(imageTexture, uv);
    fragColor = vec4(imageColor.rgb, imageColor.a * ubuf.composition.x);
}
