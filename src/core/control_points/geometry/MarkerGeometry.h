#pragma once

#include "model/MarkerTypes.h"

#include <QHash>
#include <QSize>

#include <opencv2/core.hpp>

#include <functional>

namespace xjw::control_points
{

struct MarkerCamera
{
    QString imageId;
    QString imagePath;
    cv::Matx33d intrinsics = cv::Matx33d::eye();
    cv::Matx33d rotation = cv::Matx33d::eye();
    cv::Vec3d translation{0.0, 0.0, 0.0};
    QSize imageSize;
    std::function<bool(const QPointF &)> acceptsPixel;

    cv::Matx34d projectionMatrix() const;
    cv::Point3d center() const;
    double depth(const cv::Point3d &point) const;
    QPointF project(const cv::Point3d &point) const;
    bool contains(const QPointF &pixel) const;
};

struct MarkerTriangulationOptions
{
    double maximumReprojectionErrorPx = 4.0;
    double minimumIntersectionAngleDegrees = 0.5;
};

struct MarkerTriangulation
{
    bool success = false;
    cv::Point3d point;
    double rmsReprojectionPx = std::numeric_limits<double>::quiet_NaN();
    double minimumIntersectionAngleDegrees = 0.0;
    QHash<QString, double> residualByImage;
    QStringList usedImageIds;
    QString error;
};

struct EpipolarBand
{
    bool valid = false;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double halfWidthPx = 2.0;

    double distanceTo(const QPointF &pixel) const;
    bool contains(const QPointF &pixel) const;
};

MarkerTriangulation triangulateMarker(
    const Marker &marker,
    const QVector<MarkerCamera> &cameras,
    const MarkerTriangulationOptions &options = {});

EpipolarBand epipolarSearchBand(const QPointF &sourcePixel,
                                const MarkerCamera &sourceCamera,
                                const MarkerCamera &targetCamera,
                                double halfWidthPx = 2.0);

} // namespace xjw::control_points
