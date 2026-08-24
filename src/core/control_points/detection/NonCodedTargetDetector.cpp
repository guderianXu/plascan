#include "NonCodedTargetDetector.h"

#include "DetectionMerger.h"
#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

#include <QLineF>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xjw::control_points
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

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
    return x < 0 || y < 0 || x >= mask.width() || y >= mask.height() || qGray(mask.pixel(x, y)) != 0;
}

cv::Mat grayscaleMat(const QImage &image)
{
    const QImage gray = image.convertToFormat(QImage::Format_Grayscale8);
    return cv::Mat(gray.height(),
                   gray.width(),
                   CV_8UC1,
                   const_cast<uchar *>(gray.constBits()),
                   gray.bytesPerLine()).clone();
}

double bilinearSample(const cv::Mat &image, double x, double y)
{
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= image.cols || y0 + 1 >= image.rows)
    {
        return 255.0;
    }
    const double dx = x - x0;
    const double dy = y - y0;
    const double top = (1.0 - dx) * image.at<uchar>(y0, x0) + dx * image.at<uchar>(y0, x0 + 1);
    const double bottom = (1.0 - dx) * image.at<uchar>(y0 + 1, x0)
                          + dx * image.at<uchar>(y0 + 1, x0 + 1);
    return (1.0 - dy) * top + dy * bottom;
}

QPolygonF ellipseBounds(const cv::RotatedRect &ellipse)
{
    cv::Point2f vertices[4];
    ellipse.points(vertices);
    QPolygonF polygon;
    for (const cv::Point2f &vertex : vertices)
    {
        polygon.push_back(QPointF(vertex.x, vertex.y));
    }
    return polygon;
}

QPointF refineDarkCenter(const cv::Mat &gray, const QPointF &initial, double radius)
{
    const int min_x = std::max(0, static_cast<int>(std::floor(initial.x() - radius)));
    const int max_x = std::min(gray.cols - 1, static_cast<int>(std::ceil(initial.x() + radius)));
    const int min_y = std::max(0, static_cast<int>(std::floor(initial.y() - radius)));
    const int max_y = std::min(gray.rows - 1, static_cast<int>(std::ceil(initial.y() + radius)));

    double weight_sum = 0.0;
    double weighted_x = 0.0;
    double weighted_y = 0.0;
    const double radius_squared = radius * radius;
    for (int y = min_y; y <= max_y; ++y)
    {
        for (int x = min_x; x <= max_x; ++x)
        {
            const double dx = x - initial.x();
            const double dy = y - initial.y();
            if (dx * dx + dy * dy > radius_squared)
            {
                continue;
            }
            const double weight = 255.0 - gray.at<uchar>(y, x);
            weight_sum += weight;
            weighted_x += weight * x;
            weighted_y += weight * y;
        }
    }
    if (weight_sum <= 1.0)
    {
        return initial;
    }
    return QPointF(weighted_x / weight_sum, weighted_y / weight_sum);
}

QVector<MarkerDetection> detectCircles(const cv::Mat &gray,
                                        const QImage &mask,
                                        const MarkerDetectionOptions &options)
{
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.7);
    cv::Mat binary;
    cv::threshold(blurred, binary, 0.0, 255.0, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    QVector<MarkerDetection> detections;
    const double image_area = static_cast<double>(gray.rows) * gray.cols;
    for (const std::vector<cv::Point> &contour : contours)
    {
        if (isCancelled(options))
        {
            return {};
        }
        if (contour.size() < 5)
        {
            continue;
        }

        const double area = std::abs(cv::contourArea(contour));
        const double perimeter = cv::arcLength(contour, true);
        if (area < 50.0 || area > image_area * 0.25 || perimeter <= 0.0)
        {
            continue;
        }
        const double circularity = 4.0 * kPi * area / (perimeter * perimeter);
        const cv::RotatedRect ellipse = cv::fitEllipse(contour);
        const double major = std::max(ellipse.size.width, ellipse.size.height);
        const double minor = std::min(ellipse.size.width, ellipse.size.height);
        if (circularity < 0.72 || minor < 6.0 || minor / major < 0.72)
        {
            continue;
        }

        const QPointF center = refineDarkCenter(gray,
                                                QPointF(ellipse.center.x, ellipse.center.y),
                                                0.55 * major);
        if (isMasked(mask, center))
        {
            continue;
        }

        cv::Mat inner = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::Mat outer = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::ellipse(inner, ellipse, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
        cv::RotatedRect expanded = ellipse;
        expanded.size.width *= 1.45f;
        expanded.size.height *= 1.45f;
        cv::ellipse(outer, expanded, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
        outer.setTo(0, inner);
        const double inner_mean = cv::mean(gray, inner)[0];
        const double outer_mean = cv::mean(gray, outer)[0];
        const double contrast = (outer_mean - inner_mean) / 255.0;
        if (contrast < 0.25)
        {
            continue;
        }

        MarkerDetection detection;
        detection.family = MarkerTargetFamily::NonCodedCircle;
        detection.center = center;
        detection.corners = ellipseBounds(ellipse);
        detection.confidence = std::clamp(0.55 * circularity + 0.65 * contrast, 0.0, 1.0);
        detection.centerSigmaPx = std::clamp(0.5 / std::max(contrast, 0.1), 0.1, 2.0);
        detection.sizePx = 0.5 * (major + minor);
        detection.rotationDegrees = ellipse.angle;
        detection.source = QStringLiteral("noncoded:circle");
        detections.push_back(detection);
    }
    return DetectionMerger::merge(detections, 2.0);
}

struct HarmonicScore
{
    double score = 0.0;
    double radius = 0.0;
    double angle = 0.0;
};

HarmonicScore fourQuadrantScore(const cv::Mat &gray, const cv::Point2f &center)
{
    constexpr int sample_count = 64;
    QVector<HarmonicScore> scores;
    for (int radius = 5; radius <= 36; ++radius)
    {
        if (center.x - radius < 1 || center.y - radius < 1
            || center.x + radius >= gray.cols - 1 || center.y + radius >= gray.rows - 1)
        {
            continue;
        }

        double cosine = 0.0;
        double sine = 0.0;
        double mean = 0.0;
        for (int index = 0; index < sample_count; ++index)
        {
            const double theta = 2.0 * kPi * index / sample_count;
            const double value = bilinearSample(gray,
                                                center.x + radius * std::cos(theta),
                                                center.y + radius * std::sin(theta));
            mean += value;
            cosine += value * std::cos(2.0 * theta);
            sine += value * std::sin(2.0 * theta);
        }
        mean /= sample_count;
        const double amplitude = 2.0 * std::hypot(cosine, sine) / (sample_count * 255.0);
        const double balanced = 1.0 - std::min(1.0, std::abs(mean - 127.5) / 127.5);
        const double score = amplitude * balanced;
        scores.push_back({score,
                          static_cast<double>(radius),
                          0.5 * std::atan2(sine, cosine) * 180.0 / kPi});
    }
    if (scores.isEmpty())
    {
        return {};
    }

    const auto maximum = std::max_element(scores.cbegin(), scores.cend(), [](const HarmonicScore &left,
                                                                             const HarmonicScore &right)
    {
        return left.score < right.score;
    });
    HarmonicScore best = *maximum;
    for (const HarmonicScore &candidate : scores)
    {
        if (candidate.score >= 0.92 * maximum->score && candidate.radius > best.radius)
        {
            best = candidate;
        }
    }
    return best;
}

QVector<MarkerDetection> detectFourQuadrant(const cv::Mat &gray,
                                             const QImage &mask,
                                             const MarkerDetectionOptions &options)
{
    std::vector<cv::Point2f> candidates;
    cv::goodFeaturesToTrack(gray, candidates, 128, 0.02, 6.0, cv::noArray(), 7, false, 0.04);
    if (!candidates.empty())
    {
        cv::cornerSubPix(gray,
                         candidates,
                         cv::Size(5, 5),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01));
    }

    QVector<MarkerDetection> detections;
    for (const cv::Point2f &candidate : candidates)
    {
        if (isCancelled(options))
        {
            return {};
        }
        const HarmonicScore harmonic = fourQuadrantScore(gray, candidate);
        if (harmonic.score < 0.35 || harmonic.radius < 5.0)
        {
            continue;
        }
        const QPointF center = refineDarkCenter(gray,
                                                QPointF(candidate.x, candidate.y),
                                                1.05 * harmonic.radius);
        if (isMasked(mask, center))
        {
            continue;
        }

        MarkerDetection detection;
        detection.family = MarkerTargetFamily::NonCodedFourQuadrant;
        detection.center = center;
        const double radius = harmonic.radius;
        detection.corners = QPolygonF({
            QPointF(center.x() - radius, center.y() - radius),
            QPointF(center.x() + radius, center.y() - radius),
            QPointF(center.x() + radius, center.y() + radius),
            QPointF(center.x() - radius, center.y() + radius),
        });
        detection.confidence = std::clamp(harmonic.score / 0.65, 0.0, 1.0);
        detection.centerSigmaPx = std::clamp(0.35 / std::max(harmonic.score, 0.1), 0.1, 2.0);
        detection.sizePx = 2.0 * radius;
        detection.rotationDegrees = harmonic.angle;
        detection.source = QStringLiteral("noncoded:four-quadrant");
        detections.push_back(detection);
    }
    return DetectionMerger::merge(detections, 4.0);
}

} // namespace

NonCodedTargetDetector::NonCodedTargetDetector(NonCodedTargetType type)
    : _type(type)
{
}

QVector<MarkerDetection> NonCodedTargetDetector::detect(const QImage &image,
                                                         const QImage &mask,
                                                         const MarkerDetectionOptions &options) const
{
    if (image.isNull() || isCancelled(options))
    {
        return {};
    }
    if (!mask.isNull() && mask.size() != image.size())
    {
        throw std::invalid_argument("Non-coded marker mask size must match the source image");
    }

    const cv::Mat gray = grayscaleMat(image);
    if (_type == NonCodedTargetType::Circle)
    {
        return detectCircles(gray, mask, options);
    }
    return detectFourQuadrant(gray, mask, options);
}

} // namespace xjw::control_points
