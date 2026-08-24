#include "InteractiveMaskAlgorithms.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::mask
{
    namespace
    {

        cv::Mat toBgr8(const cv::Mat& image)
        {
            if (image.empty())
            {
                return {};
            }

            cv::Mat normalized;
            if (image.depth() == CV_8U)
            {
                normalized = image;
            }
            else
            {
                cv::normalize(image, normalized, 0, 255, cv::NORM_MINMAX);
                normalized.convertTo(normalized, CV_8U);
            }

            cv::Mat bgr;
            if (normalized.channels() == 1)
            {
                cv::cvtColor(normalized, bgr, cv::COLOR_GRAY2BGR);
            }
            else if (normalized.channels() == 4)
            {
                cv::cvtColor(normalized, bgr, cv::COLOR_BGRA2BGR);
            }
            else if (normalized.channels() == 3)
            {
                bgr = normalized;
            }
            else
            {
                return {};
            }
            return bgr;
        }

        cv::Point boundedPoint(const cv::Point& point, const cv::Size& size)
        {
            return cv::Point(std::clamp(point.x, 0, std::max(0, size.width - 1)),
                             std::clamp(point.y, 0, std::max(0, size.height - 1)));
        }

    } // namespace

    cv::Mat rectangleSelection(const cv::Size& imageSize, const cv::Rect& rectangle)
    {
        if (imageSize.width <= 0 || imageSize.height <= 0)
        {
            return {};
        }

        cv::Mat selection = cv::Mat::zeros(imageSize, CV_8UC1);
        const cv::Rect bounds(0, 0, imageSize.width, imageSize.height);
        const cv::Rect clipped = rectangle & bounds;
        if (clipped.area() > 0)
        {
            selection(clipped).setTo(255);
        }
        return selection;
    }

    cv::Mat magicWandSelection(const cv::Mat& image, const cv::Point& seed, int colorTolerance)
    {
        const cv::Mat bgr = toBgr8(image);
        if (bgr.empty() || !cv::Rect(0, 0, bgr.cols, bgr.rows).contains(seed))
        {
            return {};
        }

        cv::Mat floodMask = cv::Mat::zeros(bgr.rows + 2, bgr.cols + 2, CV_8UC1);
        cv::Mat scratch = bgr.clone();
        const int tolerance = std::clamp(colorTolerance, 0, 255);
        const cv::Scalar difference(tolerance, tolerance, tolerance);
        cv::floodFill(scratch,
                      floodMask,
                      seed,
                      cv::Scalar(),
                      nullptr,
                      difference,
                      difference,
                      4 | cv::FLOODFILL_FIXED_RANGE | cv::FLOODFILL_MASK_ONLY | (255 << 8));
        return floodMask(cv::Rect(1, 1, bgr.cols, bgr.rows)).clone();
    }

    cv::Mat
    smartBrushSelection(const cv::Mat& image, const std::vector<cv::Point>& stroke, int radius, int colorTolerance)
    {
        const cv::Mat bgr = toBgr8(image);
        if (bgr.empty() || stroke.empty())
        {
            return {};
        }

        cv::Mat selection = cv::Mat::zeros(bgr.size(), CV_8UC1);
        const int effectiveRadius = std::clamp(radius, 1, 512);
        const int tolerance = std::clamp(colorTolerance, 0, 255);
        const int maximumDistanceSquared = 3 * tolerance * tolerance;

        for (const cv::Point& rawSeed : stroke)
        {
            const cv::Point seed = boundedPoint(rawSeed, bgr.size());
            const cv::Vec3b seedColor = bgr.at<cv::Vec3b>(seed);
            const cv::Rect roi(std::max(0, seed.x - effectiveRadius),
                               std::max(0, seed.y - effectiveRadius),
                               std::min(bgr.cols - std::max(0, seed.x - effectiveRadius), effectiveRadius * 2 + 1),
                               std::min(bgr.rows - std::max(0, seed.y - effectiveRadius), effectiveRadius * 2 + 1));
            for (int y = roi.y; y < roi.y + roi.height; ++y)
            {
                const cv::Vec3b* source = bgr.ptr<cv::Vec3b>(y);
                unsigned char* target = selection.ptr<unsigned char>(y);
                for (int x = roi.x; x < roi.x + roi.width; ++x)
                {
                    const int dx = x - seed.x;
                    const int dy = y - seed.y;
                    if (dx * dx + dy * dy > effectiveRadius * effectiveRadius)
                    {
                        continue;
                    }
                    int distanceSquared = 0;
                    for (int channel = 0; channel < 3; ++channel)
                    {
                        const int difference =
                            static_cast<int>(source[x][channel]) - static_cast<int>(seedColor[channel]);
                        distanceSquared += difference * difference;
                    }
                    if (distanceSquared <= maximumDistanceSquared)
                    {
                        target[x] = 255;
                    }
                }
            }
        }
        return selection;
    }

    std::vector<cv::Point>
    edgeSnappedPath(const cv::Mat& image, const cv::Point& start, const cv::Point& end, int searchRadius)
    {
        const cv::Mat bgr = toBgr8(image);
        if (bgr.empty())
        {
            return {};
        }

        const cv::Point boundedStart = boundedPoint(start, bgr.size());
        const cv::Point boundedEnd = boundedPoint(end, bgr.size());
        const double dx = static_cast<double>(boundedEnd.x - boundedStart.x);
        const double dy = static_cast<double>(boundedEnd.y - boundedStart.y);
        const int steps = std::max(std::abs(boundedEnd.x - boundedStart.x), std::abs(boundedEnd.y - boundedStart.y));
        if (steps <= 1)
        {
            return {boundedStart, boundedEnd};
        }

        const int radius = std::clamp(searchRadius, 1, 48);
        const int padding = radius + 2;
        const cv::Rect imageBounds(0, 0, bgr.cols, bgr.rows);
        const cv::Rect edgeRegion(std::min(boundedStart.x, boundedEnd.x) - padding,
                                  std::min(boundedStart.y, boundedEnd.y) - padding,
                                  std::abs(boundedEnd.x - boundedStart.x) + padding * 2 + 1,
                                  std::abs(boundedEnd.y - boundedStart.y) + padding * 2 + 1);
        const cv::Rect clippedEdgeRegion = edgeRegion & imageBounds;

        cv::Mat gray;
        cv::cvtColor(bgr(clippedEdgeRegion), gray, cv::COLOR_BGR2GRAY);
        cv::Mat gradientX;
        cv::Mat gradientY;
        cv::Sobel(gray, gradientX, CV_32F, 1, 0, 3);
        cv::Sobel(gray, gradientY, CV_32F, 0, 1, 3);
        cv::Mat gradient;
        cv::magnitude(gradientX, gradientY, gradient);

        const int stateCount = radius * 2 + 1;
        const double length = std::hypot(dx, dy);
        const double normalX = -dy / length;
        const double normalY = dx / length;
        constexpr float Infinity = std::numeric_limits<float>::max() / 8.0f;
        std::vector<float> previous(static_cast<size_t>(stateCount), Infinity);
        std::vector<float> current(static_cast<size_t>(stateCount), Infinity);
        std::vector<std::vector<short>> parents(static_cast<size_t>(steps + 1),
                                                std::vector<short>(static_cast<size_t>(stateCount), 0));
        previous[static_cast<size_t>(radius)] = 0.0f;

        auto candidatePoint = [&](int step, int offset)
        {
            const double amount = static_cast<double>(step) / static_cast<double>(steps);
            return boundedPoint(cv::Point(cvRound(boundedStart.x + dx * amount + normalX * offset),
                                          cvRound(boundedStart.y + dy * amount + normalY * offset)),
                                bgr.size());
        };

        for (int step = 1; step <= steps; ++step)
        {
            std::fill(current.begin(), current.end(), Infinity);
            for (int state = 0; state < stateCount; ++state)
            {
                const int offset = state - radius;
                const cv::Point point = candidatePoint(step, offset);
                const cv::Point localPoint = point - clippedEdgeRegion.tl();
                const float edgeCost = 255.0f - std::min(255.0f, gradient.at<float>(localPoint));
                const float centerCost = 0.18f * static_cast<float>(std::abs(offset));
                for (int previousState = std::max(0, state - 3); previousState <= std::min(stateCount - 1, state + 3);
                     ++previousState)
                {
                    const float smoothness = 2.5f * std::abs(state - previousState);
                    const float cost =
                        previous[static_cast<size_t>(previousState)] + edgeCost + centerCost + smoothness;
                    if (cost < current[static_cast<size_t>(state)])
                    {
                        current[static_cast<size_t>(state)] = cost;
                        parents[static_cast<size_t>(step)][static_cast<size_t>(state)] =
                            static_cast<short>(previousState);
                    }
                }
            }
            previous.swap(current);
        }

        int state = radius;
        std::vector<cv::Point> reversed;
        reversed.reserve(static_cast<size_t>(steps + 1));
        reversed.push_back(boundedEnd);
        for (int step = steps; step > 0; --step)
        {
            if (step != steps)
            {
                reversed.push_back(candidatePoint(step, state - radius));
            }
            state = parents[static_cast<size_t>(step)][static_cast<size_t>(state)];
        }
        reversed.push_back(boundedStart);
        std::reverse(reversed.begin(), reversed.end());
        reversed.erase(std::unique(reversed.begin(), reversed.end()), reversed.end());
        return reversed;
    }

    void applySelectionToMask(cv::Mat* mask, const cv::Mat& selection, bool exclude)
    {
        if (!mask || selection.empty())
        {
            return;
        }
        if (mask->empty() || mask->size() != selection.size() || mask->type() != CV_8UC1)
        {
            *mask = cv::Mat::zeros(selection.size(), CV_8UC1);
        }
        mask->setTo(exclude ? 255 : 0, selection);
    }

} // namespace xjw::mask
