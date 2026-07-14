#pragma once

#include "MarkerDetector.h"

namespace xjw::control_points
{

enum class AprilTagFamily
{
    Tag16h5,
    Tag25h9,
    Tag36h10,
    Tag36h11,
    Circle21h7,
    Standard41h12,
    Standard52h13
};

QVector<AprilTagFamily> supportedAprilTagFamilies();
MarkerTargetFamily markerTargetFamily(AprilTagFamily family);
QString aprilTagFamilyName(AprilTagFamily family);

class AprilTagDetector final : public MarkerDetector
{
public:
    explicit AprilTagDetector(AprilTagFamily family);

    QVector<MarkerDetection> detect(const QImage &image,
                                     const QImage &mask,
                                     const MarkerDetectionOptions &options) const override;

    AprilTagFamily family() const noexcept;

private:
    AprilTagFamily _family;
};

} // namespace xjw::control_points
