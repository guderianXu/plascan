#version 440

layout(binding = 1) uniform sampler2D cameraTexture;

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 cardUv;
layout(location = 0) out vec4 fragColor;

void main()
{
    vec4 imageColor = texture(cameraTexture, uv);
    float edgeDistance = min(
        min(cardUv.x, 1.0 - cardUv.x),
        min(cardUv.y, 1.0 - cardUv.y));
    float borderMask = 1.0 - smoothstep(0.018, 0.035, edgeDistance);
    vec3 borderColor = vec3(0.08, 0.31, 0.62);
    fragColor = vec4(mix(imageColor.rgb, borderColor, borderMask), 1.0);
}
