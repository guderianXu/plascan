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

        QVector<MarkerDetection>
        detect(const cv::Mat& image, const cv::Mat& mask, const MarkerDetectionOptions& options) const override;

    private:
        NonCodedTargetType _type;
    };

} // namespace xjw::control_points
