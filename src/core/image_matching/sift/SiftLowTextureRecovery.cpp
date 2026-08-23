#include "SiftLowTextureRecovery.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xjw::image_matching
{
    namespace
    {

        constexpr int kGridColumns = 8;
        constexpr int kGridRows = 8;
        constexpr int kGridCellCount = kGridColumns * kGridRows;

        int adaptiveBorder(const ImageFeatureInput& input,
                           const ImageMatchingRuntimeConfig& runtime)
        {
            const int requested = std::max(0, runtime.removeBorders);
            if (!runtime.adaptiveSift)
            {
                return requested;
            }
            const int minimumSide = std::min(input.grayImage.cols, input.grayImage.rows);
            return std::min(requested, std::max(4, minimumSide / 64));
        }

        cv::Rect gridCellRect(const cv::Size& size, int row, int column)
        {
            const int left = column * size.width / kGridColumns;
            const int right = (column + 1) * size.width / kGridColumns;
            const int top = row * size.height / kGridRows;
            const int bottom = (row + 1) * size.height / kGridRows;
            return cv::Rect(left, top, right - left, bottom - top);
        }

        int cellIndex(const cv::Point2f& point, const cv::Size& size)
        {
            const int column = std::clamp(
                static_cast<int>(point.x * kGridColumns / std::max(1, size.width)),
                0,
                kGridColumns - 1);
            const int row = std::clamp(
                static_cast<int>(point.y * kGridRows / std::max(1, size.height)),
                0,
                kGridRows - 1);
            return row * kGridColumns + column;
        }

        std::uint64_t quantizedPointKey(const cv::Point2f& point)
        {
            const int x = std::max(0, static_cast<int>(std::lround(point.x / 3.0f)));
            const int y = std::max(0, static_cast<int>(std::lround(point.y / 3.0f)));
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(y)) << 32U) |
                static_cast<std::uint32_t>(x);
        }

        cv::Mat validPhotometricMask(const ImageFeatureInput& input,
                                     const ImageMatchingRuntimeConfig& runtime)
        {
            const int minimum = std::clamp(
                static_cast<int>(std::ceil(std::clamp(runtime.grayscaleMin, 0.0f, 1.0f) * 255.0f)),
                0,
                255);
            const int maximum = std::clamp(
                static_cast<int>(std::floor(std::clamp(runtime.grayscaleMax, 0.0f, 1.0f) * 255.0f)),
                0,
                255);
            cv::Mat mask;
            cv::inRange(input.grayImage, cv::Scalar(minimum), cv::Scalar(maximum), mask);
            if (!input.validMask.empty())
            {
                cv::bitwise_and(mask, input.validMask, mask);
            }

            const int border = adaptiveBorder(input, runtime);
            if (border > 0 && border * 2 < mask.cols && border * 2 < mask.rows)
            {
                mask.rowRange(0, border).setTo(0);
                mask.rowRange(mask.rows - border, mask.rows).setTo(0);
                mask.colRange(0, border).setTo(0);
                mask.colRange(mask.cols - border, mask.cols).setTo(0);
            }
            return mask;
        }

        cv::Mat robustGrayStretch(const cv::Mat& image, const cv::Mat& validMask)
        {
            std::array<int, 256> histogram{};
            int sampleCount = 0;
            for (int row = 0; row < image.rows; ++row)
            {
                const unsigned char* imageRow = image.ptr<unsigned char>(row);
                const unsigned char* maskRow = validMask.ptr<unsigned char>(row);
                for (int column = 0; column < image.cols; ++column)
                {
                    if (maskRow[column] == 0)
                    {
                        continue;
                    }
                    ++histogram[imageRow[column]];
                    ++sampleCount;
                }
            }
            if (sampleCount <= 0)
            {
                return image;
            }

            const int lowerTarget = std::max(1, sampleCount / 50);
            const int upperTarget = sampleCount - lowerTarget;
            int cumulative = 0;
            int lower = 0;
            int upper = 255;
            for (int value = 0; value < 256; ++value)
            {
                cumulative += histogram[static_cast<std::size_t>(value)];
                if (cumulative >= lowerTarget)
                {
                    lower = value;
                    break;
                }
            }
            cumulative = 0;
            for (int value = 0; value < 256; ++value)
            {
                cumulative += histogram[static_cast<std::size_t>(value)];
                if (cumulative >= upperTarget)
                {
                    upper = value;
                    break;
                }
            }
            if (upper - lower < 16)
            {
                return image;
            }

            cv::Mat lookup(1, 256, CV_8UC1);
            for (int value = 0; value < 256; ++value)
            {
                lookup.at<unsigned char>(value) = static_cast<unsigned char>(std::clamp(
                    (value - lower) * 255 / (upper - lower), 0, 255));
            }
            cv::Mat stretched;
            cv::LUT(image, lookup, stretched);
            return stretched;
        }

    } // namespace

    bool SiftLowTextureRecoveryPlan::isValid() const
    {
        return !enhancedImage.empty() && !recoveryMask.empty() &&
            maximumFeatures > 0 && targetFeatures > 0;
    }

    std::optional<SiftLowTextureRecoveryPlan> planSiftLowTextureRecovery(
        const ImageFeatureInput& input,
        const ImageMatchingRuntimeConfig& runtime,
        const std::vector<cv::KeyPoint>& baseKeypoints,
        int targetFeatureCount)
    {
        if (!runtime.lowTextureRecovery || input.grayImage.empty() ||
            input.grayImage.type() != CV_8UC1 || targetFeatureCount <= 0)
        {
            return std::nullopt;
        }

        const cv::Mat validMask = validPhotometricMask(input, runtime);
        std::vector<int> cellCounts(kGridCellCount, 0);
        for (const cv::KeyPoint& keypoint : baseKeypoints)
        {
            const int x = static_cast<int>(std::lround(keypoint.pt.x));
            const int y = static_cast<int>(std::lround(keypoint.pt.y));
            if (x >= 0 && y >= 0 && x < validMask.cols && y < validMask.rows &&
                validMask.at<unsigned char>(y, x) != 0)
            {
                ++cellCounts[static_cast<std::size_t>(cellIndex(keypoint.pt, validMask.size()))];
            }
        }

        const int desiredPerCell = std::clamp(
            (targetFeatureCount + kGridCellCount * 2 - 1) / (kGridCellCount * 2),
            4,
            16);
        cv::Mat recoveryMask = cv::Mat::zeros(input.grayImage.size(), CV_8UC1);
        int recoveryCellCount = 0;
        for (int row = 0; row < kGridRows; ++row)
        {
            for (int column = 0; column < kGridColumns; ++column)
            {
                const int index = row * kGridColumns + column;
                const cv::Rect cell = gridCellRect(input.grayImage.size(), row, column);
                const int minimumValidPixels = std::max(64, cell.area() / 5);
                if (cellCounts[static_cast<std::size_t>(index)] >= desiredPerCell ||
                    cv::countNonZero(validMask(cell)) < minimumValidPixels)
                {
                    continue;
                }
                validMask(cell).copyTo(recoveryMask(cell));
                ++recoveryCellCount;
            }
        }
        if (recoveryCellCount == 0)
        {
            return std::nullopt;
        }

        SiftLowTextureRecoveryPlan plan;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(kGridColumns, kGridRows));
        clahe->apply(robustGrayStretch(input.grayImage, validMask), plan.enhancedImage);
        plan.recoveryMask = std::move(recoveryMask);
        plan.maximumFeatures = runtime.maxKeypoints > 0
            ? std::clamp(runtime.maxKeypoints / 2, 512, 4096)
            : 4096;
        plan.targetFeatures = std::min(
            plan.maximumFeatures, std::max(256, targetFeatureCount / 2));
        return plan;
    }

    SiftRawFeatures filterRecoveredSiftFeatures(
        const SiftRawFeatures& base,
        const SiftRawFeatures& recovered,
        const cv::Mat& recoveryMask)
    {
        SiftRawFeatures filtered;
        if (recoveryMask.empty() || recoveryMask.type() != CV_8UC1 ||
            recovered.descriptors.empty() ||
            recovered.descriptors.rows != static_cast<int>(recovered.keypoints.size()))
        {
            return filtered;
        }

        std::unordered_set<std::uint64_t> occupied;
        occupied.reserve(base.keypoints.size() + recovered.keypoints.size());
        for (const cv::KeyPoint& keypoint : base.keypoints)
        {
            const int x = static_cast<int>(std::lround(keypoint.pt.x));
            const int y = static_cast<int>(std::lround(keypoint.pt.y));
            if (x >= 0 && y >= 0 && x < recoveryMask.cols && y < recoveryMask.rows &&
                recoveryMask.at<unsigned char>(y, x) != 0)
            {
                occupied.insert(quantizedPointKey(keypoint.pt));
            }
        }

        for (int index = 0; index < static_cast<int>(recovered.keypoints.size()); ++index)
        {
            cv::KeyPoint keypoint = recovered.keypoints[static_cast<std::size_t>(index)];
            const int x = static_cast<int>(std::lround(keypoint.pt.x));
            const int y = static_cast<int>(std::lround(keypoint.pt.y));
            if (x < 0 || y < 0 || x >= recoveryMask.cols || y >= recoveryMask.rows ||
                recoveryMask.at<unsigned char>(y, x) == 0 ||
                !occupied.insert(quantizedPointKey(keypoint.pt)).second)
            {
                continue;
            }
            keypoint.class_id = 1;
            keypoint.response *= 0.5f;
            filtered.keypoints.push_back(keypoint);
            filtered.descriptors.push_back(recovered.descriptors.row(index));
        }
        return filtered;
    }

} // namespace xjw::image_matching
