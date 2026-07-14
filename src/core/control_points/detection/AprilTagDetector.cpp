#include "AprilTagDetector.h"

#include <apriltag/apriltag.h>
#include <apriltag/common/image_u8.h>
#include <apriltag/common/zarray.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag25h9.h>
#include <apriltag/tag36h10.h>
#include <apriltag/tag36h11.h>
#include <apriltag/tagCircle21h7.h>
#include <apriltag/tagStandard41h12.h>
#include <apriltag/tagStandard52h13.h>

#include <QLineF>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <thread>

namespace xjw::control_points
{
namespace
{

using FamilyDestroy = void (*)(apriltag_family_t *);
using FamilyHandle = std::unique_ptr<apriltag_family_t, FamilyDestroy>;
using DetectorHandle = std::unique_ptr<apriltag_detector_t, decltype(&apriltag_detector_destroy)>;
using DetectionsHandle = std::unique_ptr<zarray_t, decltype(&apriltag_detections_destroy)>;

FamilyHandle createFamily(AprilTagFamily family)
{
    switch (family)
    {
    case AprilTagFamily::Tag16h5:
        return FamilyHandle(tag16h5_create(), tag16h5_destroy);
    case AprilTagFamily::Tag25h9:
        return FamilyHandle(tag25h9_create(), tag25h9_destroy);
    case AprilTagFamily::Tag36h10:
        return FamilyHandle(tag36h10_create(), tag36h10_destroy);
    case AprilTagFamily::Tag36h11:
        return FamilyHandle(tag36h11_create(), tag36h11_destroy);
    case AprilTagFamily::Circle21h7:
        return FamilyHandle(tagCircle21h7_create(), tagCircle21h7_destroy);
    case AprilTagFamily::Standard41h12:
        return FamilyHandle(tagStandard41h12_create(), tagStandard41h12_destroy);
    case AprilTagFamily::Standard52h13:
        return FamilyHandle(tagStandard52h13_create(), tagStandard52h13_destroy);
    }
    throw std::invalid_argument("Unsupported AprilTag family");
}

bool isCancelled(const MarkerDetectionOptions &options)
{
    return options.cancelRequested != nullptr && options.cancelRequested->load(std::memory_order_relaxed);
}

bool isMasked(const QImage &mask, const QPointF &point)
{
    if (mask.isNull())
    {
        return false;
    }

    const int x = qRound(point.x());
    const int y = qRound(point.y());
    if (x < 0 || y < 0 || x >= mask.width() || y >= mask.height())
    {
        return true;
    }
    return qGray(mask.pixel(x, y)) != 0;
}

bool detectionIsMasked(const QImage &mask, const MarkerDetection &detection)
{
    if (isMasked(mask, detection.center))
    {
        return true;
    }
    for (const QPointF &corner : detection.corners)
    {
        if (isMasked(mask, corner))
        {
            return true;
        }
    }
    return false;
}

MarkerDetection copyDetection(const apriltag_detection_t &source, AprilTagFamily family)
{
    MarkerDetection detection;
    detection.family = markerTargetFamily(family);
    detection.targetId = source.id;
    detection.center = QPointF(source.c[0], source.c[1]);
    for (const auto &corner : source.p)
    {
        detection.corners.push_back(QPointF(corner[0], corner[1]));
    }
    detection.decisionMargin = source.decision_margin;
    detection.hamming = source.hamming;
    detection.confidence = std::clamp(source.decision_margin / 100.0, 0.0, 1.0);
    detection.centerSigmaPx = std::clamp(10.0 / std::max(1.0f, source.decision_margin), 0.05, 2.0);
    detection.sizePx = 0.25 * (QLineF(detection.corners[0], detection.corners[1]).length()
                               + QLineF(detection.corners[1], detection.corners[2]).length()
                               + QLineF(detection.corners[2], detection.corners[3]).length()
                               + QLineF(detection.corners[3], detection.corners[0]).length());
    detection.rotationDegrees = QLineF(detection.corners[0], detection.corners[1]).angle();
    detection.source = QStringLiteral("apriltag:%1").arg(aprilTagFamilyName(family));
    return detection;
}

} // namespace

QVector<AprilTagFamily> supportedAprilTagFamilies()
{
    return {
        AprilTagFamily::Tag16h5,
        AprilTagFamily::Tag25h9,
        AprilTagFamily::Tag36h10,
        AprilTagFamily::Tag36h11,
        AprilTagFamily::Circle21h7,
        AprilTagFamily::Standard41h12,
        AprilTagFamily::Standard52h13,
    };
}

MarkerTargetFamily markerTargetFamily(AprilTagFamily family)
{
    switch (family)
    {
    case AprilTagFamily::Tag16h5:
        return MarkerTargetFamily::AprilTag16h5;
    case AprilTagFamily::Tag25h9:
        return MarkerTargetFamily::AprilTag25h9;
    case AprilTagFamily::Tag36h10:
        return MarkerTargetFamily::AprilTag36h10;
    case AprilTagFamily::Tag36h11:
        return MarkerTargetFamily::AprilTag36h11;
    case AprilTagFamily::Circle21h7:
        return MarkerTargetFamily::AprilTagCircle21h7;
    case AprilTagFamily::Standard41h12:
        return MarkerTargetFamily::AprilTagStandard41h12;
    case AprilTagFamily::Standard52h13:
        return MarkerTargetFamily::AprilTagStandard52h13;
    }
    throw std::invalid_argument("Unsupported AprilTag family");
}

QString aprilTagFamilyName(AprilTagFamily family)
{
    return markerTargetFamilyName(markerTargetFamily(family));
}

AprilTagDetector::AprilTagDetector(AprilTagFamily family)
    : _family(family)
{
}

QVector<MarkerDetection> AprilTagDetector::detect(const QImage &image,
                                                   const QImage &mask,
                                                   const MarkerDetectionOptions &options) const
{
    if (image.isNull() || isCancelled(options))
    {
        return {};
    }
    if (!mask.isNull() && mask.size() != image.size())
    {
        throw std::invalid_argument("AprilTag mask size must match the source image");
    }

    const QImage gray = image.convertToFormat(QImage::Format_Grayscale8);
    image_u8_t input = {
        static_cast<int32_t>(gray.width()),
        static_cast<int32_t>(gray.height()),
        static_cast<int32_t>(gray.bytesPerLine()),
        const_cast<uint8_t *>(gray.constBits()),
    };

    FamilyHandle family = createFamily(_family);
    DetectorHandle detector(apriltag_detector_create(), apriltag_detector_destroy);
    if (!family || !detector)
    {
        throw std::runtime_error("Failed to allocate AprilTag detector resources");
    }

    const unsigned int hardware_threads = std::max(1u, std::thread::hardware_concurrency());
    detector->nthreads = options.threadCount > 0
                             ? options.threadCount
                             : static_cast<int>(hardware_threads);
    detector->quad_decimate = static_cast<float>(std::max(0.25, options.quadDecimate));
    detector->quad_sigma = static_cast<float>(options.quadSigma);
    detector->refine_edges = options.refineEdges;
    apriltag_detector_add_family_bits(detector.get(), family.get(), std::clamp(options.maxHamming, 0, 2));

    DetectionsHandle native_detections(
        apriltag_detector_detect(detector.get(), &input),
        apriltag_detections_destroy);
    if (!native_detections || isCancelled(options))
    {
        return {};
    }

    QVector<MarkerDetection> result;
    result.reserve(zarray_size(native_detections.get()));
    for (int index = 0; index < zarray_size(native_detections.get()); ++index)
    {
        if (isCancelled(options))
        {
            return {};
        }

        apriltag_detection_t *native_detection = nullptr;
        zarray_get(native_detections.get(), index, &native_detection);
        MarkerDetection detection = copyDetection(*native_detection, _family);
        if (detection.decisionMargin < options.minDecisionMargin || detectionIsMasked(mask, detection))
        {
            continue;
        }
        result.push_back(std::move(detection));
    }
    return result;
}

AprilTagFamily AprilTagDetector::family() const noexcept
{
    return _family;
}

QString markerTargetFamilyName(MarkerTargetFamily family)
{
    switch (family)
    {
    case MarkerTargetFamily::AprilTag16h5:
        return QStringLiteral("tag16h5");
    case MarkerTargetFamily::AprilTag25h9:
        return QStringLiteral("tag25h9");
    case MarkerTargetFamily::AprilTag36h10:
        return QStringLiteral("tag36h10");
    case MarkerTargetFamily::AprilTag36h11:
        return QStringLiteral("tag36h11");
    case MarkerTargetFamily::AprilTagCircle21h7:
        return QStringLiteral("tagCircle21h7");
    case MarkerTargetFamily::AprilTagStandard41h12:
        return QStringLiteral("tagStandard41h12");
    case MarkerTargetFamily::AprilTagStandard52h13:
        return QStringLiteral("tagStandard52h13");
    case MarkerTargetFamily::Circular12Bit:
        return QStringLiteral("circular12");
    case MarkerTargetFamily::Circular14Bit:
        return QStringLiteral("circular14");
    case MarkerTargetFamily::Circular16Bit:
        return QStringLiteral("circular16");
    case MarkerTargetFamily::Circular20Bit:
        return QStringLiteral("circular20");
    case MarkerTargetFamily::NonCodedCircle:
        return QStringLiteral("noncoded-circle");
    case MarkerTargetFamily::NonCodedFourQuadrant:
        return QStringLiteral("noncoded-four-quadrant");
    }
    return QStringLiteral("unknown");
}

} // namespace xjw::control_points
