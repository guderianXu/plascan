#pragma once

#include "MarkerDetector.h"

namespace xjw::control_points
{

enum class NonCodedTargetType
{
    Circle,
    FourQuadrant
};

class NonCodedTargetDetector final : public MarkerDetector
{
public:
    explicit NonCodedTargetDetector(NonCodedTargetType type);

    QVector<MarkerDetection> detect(const QImage &image,
                                     const QImage &mask,
                                     const MarkerDetectionOptions &options) const override;

private:
    NonCodedTargetType _type;
};

} // namespace xjw::control_points
