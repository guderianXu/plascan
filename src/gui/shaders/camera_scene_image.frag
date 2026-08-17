#version 440

layout(binding = 1) uniform sampler2D imageTexture;

layout(std140, binding = 0) uniform ProjectedImageUniforms
{
    mat4 uMVP;
    mat4 uSourceView;
    vec4 intrinsics;
    vec4 imageGeometry;
    vec4 composition;
} ubuf;

layout(location = 0) in vec3 sourceCameraPosition;
layout(location = 0) out vec4 fragColor;

void main()
{
    float depth = -sourceCameraPosition.z;
    if (depth <= 1.0e-6)
    {
        discard;
    }

    float pixelX = ubuf.intrinsics.z
        + ubuf.imageGeometry.z * ubuf.intrinsics.x
            * sourceCameraPosition.x / depth;
    float pixelY = ubuf.intrinsics.w
        + ubuf.imageGeometry.w * ubuf.intrinsics.y
            * sourceCameraPosition.y / depth;
    vec2 uv = vec2(pixelX / ubuf.imageGeometry.x,
                   pixelY / ubuf.imageGeometry.y);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))))
    {
        discard;
    }

    vec4 imageColor = texture(imageTexture, uv);
    fragColor = vec4(imageColor.rgb, imageColor.a * ubuf.composition.x);
}
