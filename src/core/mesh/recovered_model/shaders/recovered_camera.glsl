// Clean-room ordinary-perspective subset of the 688-byte target camera ABI.
// The caller must reject non-perspective cameras, non-zero auxiliary records,
// and unsupported calibration branches before dispatch.

float camera_float(uint byte_offset) {
    return uintBitsToFloat(camera_words[byte_offset >> 2]);
}

int camera_int(uint byte_offset) {
    return int(camera_words[byte_offset >> 2]);
}

vec3 camera_local_point(vec3 world) {
    vec3 delta = world - vec3(
        camera_float(48), camera_float(52), camera_float(56));
    mat3 camera_to_world = mat3(
        vec3(camera_float(0), camera_float(4), camera_float(8)),
        vec3(camera_float(16), camera_float(20), camera_float(24)),
        vec3(camera_float(32), camera_float(36), camera_float(40)));
    return (transpose(camera_to_world) * delta) /
           vec3(camera_float(60));
}

vec2 camera_apply_image_matrix(vec2 pixel) {
    mat3 matrix = mat3(
        vec3(camera_float(592), camera_float(596), camera_float(600)),
        vec3(camera_float(608), camera_float(612), camera_float(616)),
        vec3(camera_float(624), camera_float(628), camera_float(632)));
    vec3 transformed = matrix * vec3(pixel, 1.0);
    return transformed.xy / transformed.z;
}

bool camera_project(vec3 world, out vec2 pixel, out float depth) {
    vec3 local = camera_local_point(world);
    depth = local.z;
    if (camera_int(520) != 1 || !(depth > 0.0)) return false;

    float inverse_depth = 1.0 / depth;
    float x = local.x * inverse_depth;
    float y = local.y * inverse_depth;
    vec2 distorted = vec2(x, y);
    if (camera_int(552) != 0) {
        float x2 = x * x;
        float y2 = y * y;
        float radius2 = x2 + y2;
        float maximum_radius2 = camera_float(96);
        bool outside = radius2 > maximum_radius2;
        float tangential_scale = outside ? maximum_radius2 / radius2 : 1.0;
        float limited_radius2 = outside ? maximum_radius2 : radius2;
        float radius4 = limited_radius2 * limited_radius2;
        float radial = camera_float(64) * limited_radius2 +
                       camera_float(68) * radius4 +
                       camera_float(72) * limited_radius2 * radius4 +
                       camera_float(76) * radius4 * radius4;
        float tangential_radial = 1.0 +
            camera_float(88) * limited_radius2 +
            camera_float(92) * radius4;
        float p1 = camera_float(80);
        float p2 = camera_float(84);
        vec2 correction;
        correction.x = x * radial +
            (p1 * (y2 + 3.0 * x2) + 2.0 * p2 * x * y) *
                tangential_radial * tangential_scale;
        correction.y = y * radial +
            (p2 * (x2 + 3.0 * y2) + 2.0 * p1 * x * y) *
                tangential_radial * tangential_scale;
        distorted = vec2(x, y) + correction;
    }

    float focal = camera_float(532);
    vec2 calibrated;
    calibrated.x = distorted.x * (focal + camera_float(544)) +
                   camera_float(548) * distorted.y;
    calibrated.y = focal * distorted.y;
    calibrated += vec2(
        camera_float(536) + float(camera_int(524)) * 0.5,
        camera_float(540) + float(camera_int(528)) * 0.5);
    pixel = camera_apply_image_matrix(calibrated);
    return true;
}
