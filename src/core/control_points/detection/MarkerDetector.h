#pragma once

#include <opencv2/core/mat.hpp>
#include <QPointF>
#include <QString>
#include <QVector>

#include <atomic>

namespace xjw::control_points
{

    enum class MarkerTargetFamily
    {
        AprilTag16h5,
        AprilTag25h9,
        AprilTag36h10,
        AprilTag36h11,
        AprilTagCircle21h7,
        AprilTagStandard41h12,
        AprilTagStandard52h13,
        Circular12Bit,
        Circular14Bit,
        Circular16Bit,
        Circular20Bit,
        NonCodedCircle,
        NonCodedFourQuadrant
    };

    struct MarkerDetection
    {
        MarkerTargetFamily family = MarkerTargetFamily::AprilTag36h11;
        int targetId = -1;
        QPointF center;
        QVector<QPointF> corners;
        double confidence = 0.0;
        double centerSigmaPx = 1.0;
        double decisionMargin = 0.0;
        int hamming = 0;
        double sizePx = 0.0;
        double rotationDegrees = 0.0;
        QString source;
    };

    struct MarkerDetectionOptions
    {
        int threadCount = 0;
        double quadDecimate = 1.0;
        double quadSigma = 0.0;
        bool refineEdges = true;
        int maxHamming = 2;
        double minDecisionMargin = 0.0;
        const std::atomic_bool* cancelRequested = nullptr;
    };

    // Both inputs are two-dimensional CV_8UC1 matrices. An empty mask means no exclusions;
    // any nonzero mask pixel excludes that location. Row padding and ROI views are supported.
    // Storage is borrowed only for the duration of the synchronous detect() call.
    void validateMarkerImages(const cv::Mat& image, const cv::Mat& mask);

    class MarkerDetector
    {
    public:
        virtual ~MarkerDetector() = default;

        virtual QVector<MarkerDetection>
        detect(const cv::Mat& image, const cv::Mat& mask, const MarkerDetectionOptions& options) const = 0;
    };

    QString markerTargetFamilyName(MarkerTargetFamily family);

} // namespace xjw::control_points
