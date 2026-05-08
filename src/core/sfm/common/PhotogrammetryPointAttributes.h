#pragma once

namespace xjw {
namespace sfm {

struct PhotogrammetryPointAttributes
{
    int pointId = -1;
    int trackLength = 0;
    float reprojectionError = 0.0f;
    float confidence = 1.0f;
    bool isControlPoint = false;
    bool isValid = true;
};

} // namespace sfm
} // namespace xjw
