#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec3 world_position;
layout(location = 0) out float camera_depth;
layout(set = 0, binding = 0, std430) readonly buffer Camera {
    uint camera_words[];
};
layout(set = 0, binding = 1, std430) readonly buffer NearFar {
    float near_depth;
    float far_depth;
};
layout(push_constant) uniform Push {
    int raster_downscale;
    int reserved;
} push_data;

#include "recovered_camera.glsl"

void main() {
    vec2 pixel;
    float depth;
    bool valid = camera_project(world_position, pixel, depth);
    if (!valid) {
        pixel = vec2(float(camera_int(524)) * 0.5,
                     float(camera_int(528)) * 0.5);
        depth = -1000000000.0;
    }
    camera_depth = depth;
    float normalized_depth;
    if (near_depth == 3.402823466e+38 || far_depth == -3.402823466e+38)
        normalized_depth = 1.001;
    else if (near_depth == far_depth)
        normalized_depth = 1.0;
    else
        normalized_depth = (depth - near_depth) / (far_depth - near_depth);
    vec2 raster_size = vec2(
        float((camera_int(524) + push_data.raster_downscale - 1) /
              push_data.raster_downscale),
        float((camera_int(528) + push_data.raster_downscale - 1) /
              push_data.raster_downscale));
    vec2 ndc = ((pixel / float(push_data.raster_downscale)) /
                raster_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, normalized_depth, 1.0);
}
