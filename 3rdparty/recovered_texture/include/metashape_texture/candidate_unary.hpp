#pragma once

#include <cstdint>
#include <vector>

namespace metashape_texture::recovered {

// Native FaceCameraOption payload after its uint32 face-rank field.
struct FaceCameraQuality {
    float sharpness = -128.0F;
    float photoconsistency = -128.0F;
    float resolution = -128.0F;
    float nadirness = -128.0F;
};

// Context bytes +0x68..+0x6b, in native order.
struct FaceCameraCostFlags {
    bool sharpness = false;
    bool photoconsistency = true;
    bool nadirness = true;
    bool resolution = true;
};

// Exact MSVC/SSE binary32 Record20 -> unary conversion for one face.  The
// input contains only present camera candidates, in camera order.
std::vector<std::int32_t> build_face_camera_unary(
    const std::vector<FaceCameraQuality> &options,
    float face_weight,
    FaceCameraCostFlags flags);

}  // namespace metashape_texture::recovered
