#include "SiftFeatureExtractor.h"

#include "SiftComputeBackend.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace xjw::image_matching
{
    namespace
    {

        int adaptiveBorder(const ImageFeatureInput& input, const ImageMatchingRuntimeConfig& runtime)
        {
            const int requested = std::max(0, runtime.removeBorders);
            if (!runtime.adaptiveSift)
            {
                return requested;
            }
            const int minimumSide = std::min(input.grayImage.cols, input.grayImage.rows);
            return std::min(requested, std::max(4, minimumSide / 64));
        }

        bool pointIsAllowed(const ImageFeatureInput& input,
                            const cv::KeyPoint& keypoint,
                            const ImageMatchingRuntimeConfig& runtime)
        {
            const int x = static_cast<int>(std::lround(keypoint.pt.x));
            const int y = static_cast<int>(std::lround(keypoint.pt.y));
            if (x < 0 || y < 0 || x >= input.grayImage.cols || y >= input.grayImage.rows)
            {
                return false;
            }
            const int border = adaptiveBorder(input, runtime);
            if (x < border || y < border || x >= input.grayImage.cols - border || y >= input.grayImage.rows - border)
            {
                return false;
            }

            const float gray = static_cast<float>(input.grayImage.at<unsigned char>(y, x)) / 255.0f;
            if (gray < runtime.grayscaleMin || gray > runtime.grayscaleMax)
            {
                return false;
            }
            return input.validMask.empty() || input.validMask.at<unsigned char>(y, x) != 0;
        }

        void
        transformKeypoints(std::vector<cv::KeyPoint>* keypoints, double inverseScale, const cv::Point2f& offset = {})
        {
            if (!keypoints)
            {
                return;
            }
            const float scale = static_cast<float>(inverseScale);
            for (cv::KeyPoint& keypoint : *keypoints)
            {
                keypoint.pt.x = keypoint.pt.x * scale + offset.x;
                keypoint.pt.y = keypoint.pt.y * scale + offset.y;
                keypoint.size *= scale;
            }
        }

        void appendRawFeatures(SiftRawFeatures* destination,
                               SiftRawFeatures source,
                               double inverseScale,
                               const cv::Point2f& offset = {})
        {
            if (!destination || source.keypoints.empty() || source.descriptors.empty())
            {
                return;
            }
            transformKeypoints(&source.keypoints, inverseScale, offset);
            destination->keypoints.insert(
                destination->keypoints.end(), source.keypoints.begin(), source.keypoints.end());
            destination->descriptors.push_back(source.descriptors);
        }

        cv::Mat scaledImage(const cv::Mat& image, double scale, int interpolation)
        {
            if (image.empty() || std::abs(scale - 1.0) < 1e-6)
            {
                return image;
            }
            cv::Mat result;
            cv::resize(image, result, cv::Size(), scale, scale, interpolation);
            return result;
        }

        int detectorLimit(int requested)
        {
            return requested > 0 ? std::max(256, requested * 2) : 0;
        }

        SiftRawFeatures
        extractCpuPass(const cv::Mat& image, const cv::Mat& validMask, int requested, double contrastThreshold)
        {
            SiftRawFeatures result;
            cv::Ptr<cv::SIFT> sift = cv::SIFT::create(
                detectorLimit(requested), 3, std::clamp(contrastThreshold, 0.001, 0.20), 10.0, 1.6, true);
            sift->detectAndCompute(image,
                                   validMask.empty() ? cv::noArray() : cv::InputArray(validMask),
                                   result.keypoints,
                                   result.descriptors,
                                   false);
            return result;
        }

        int adaptiveTarget(const cv::Size& size, int maximumKeypoints)
        {
            const double megapixels = static_cast<double>(size.area()) / 1'000'000.0;
            const int densityTarget = std::max(1000, static_cast<int>(std::ceil(megapixels * 1800.0)));
            return maximumKeypoints > 0 ? std::min(maximumKeypoints, densityTarget) : densityTarget;
        }

        template <typename ExtractPass>
        SiftRawFeatures adaptivePass(const cv::Mat& image,
                                     const cv::Mat& mask,
                                     int requested,
                                     int target,
                                     float initialThreshold,
                                     ExtractPass extractPass)
        {
            SiftRawFeatures best;
            float threshold = initialThreshold;
            const int attempts = target > 0 ? 4 : 1;
            for (int attempt = 0; attempt < attempts; ++attempt)
            {
                SiftRawFeatures current = extractPass(image, mask, requested, threshold);
                if (current.keypoints.size() > best.keypoints.size())
                {
                    best = std::move(current);
                }
                if (static_cast<int>(best.keypoints.size()) >= target)
                {
                    break;
                }
                threshold *= 0.5f;
            }
            return best;
        }

        SiftRawFeatures extractAdaptive(const ImageFeatureInput& input,
                                        const ImageMatchingRuntimeConfig& runtime,
                                        SiftComputeBackend backend)
        {
            const int maximumSide = std::max(input.grayImage.cols, input.grayImage.rows);
            const int minimumSide = std::min(input.grayImage.cols, input.grayImage.rows);
            const int maxDimension =
                runtime.maxImageDimension > 0 ? std::max(512, runtime.maxImageDimension) : maximumSide;
            const int target = adaptiveTarget(input.grayImage.size(), runtime.maxKeypoints);
            const auto runPass = [&](const cv::Mat& image, const cv::Mat& mask, int requested, int passTarget)
            {
                if (backend != SiftComputeBackend::Cpu)
                {
                    const float initialThreshold = backend == SiftComputeBackend::Cuda ? runtime.siftDetectionThreshold
                                                                                       : runtime.siftContrastThreshold;
                    return adaptivePass(image,
                                        mask,
                                        requested,
                                        passTarget,
                                        initialThreshold,
                                        [&](const cv::Mat& passImage, const cv::Mat&, int limit, float threshold)
                                        {
                                            SiftExtractionRequest request;
                                            request.image = passImage;
                                            request.maximumFeatures = std::max(1024, detectorLimit(limit));
                                            request.contrastThreshold = threshold;
                                            request.deviceIndex = runtime.cudaDevice;
                                            return extractSiftOnGpu(backend, request);
                                        });
                }
                return adaptivePass(image,
                                    mask,
                                    requested,
                                    passTarget,
                                    runtime.siftContrastThreshold,
                                    [](const cv::Mat& passImage, const cv::Mat& passMask, int limit, float threshold)
                                    { return extractCpuPass(passImage, passMask, limit, threshold); });
            };

            SiftRawFeatures combined;
            if (minimumSide < 800)
            {
                constexpr double scale = 2.0;
                const cv::Mat upscaled = scaledImage(input.grayImage, scale, cv::INTER_CUBIC);
                const cv::Mat mask = scaledImage(input.validMask, scale, cv::INTER_NEAREST);
                appendRawFeatures(&combined, runPass(upscaled, mask, runtime.maxKeypoints, target), 1.0 / scale);
                return combined;
            }

            if (maximumSide <= maxDimension)
            {
                appendRawFeatures(
                    &combined, runPass(input.grayImage, input.validMask, runtime.maxKeypoints, target), 1.0);
                return combined;
            }

            const double coarseScale = static_cast<double>(maxDimension) / static_cast<double>(maximumSide);
            const int coarseBudget = runtime.maxKeypoints > 0 ? std::max(1024, runtime.maxKeypoints * 2 / 3) : 0;
            const cv::Mat coarseImage = scaledImage(input.grayImage, coarseScale, cv::INTER_AREA);
            const cv::Mat coarseMask = scaledImage(input.validMask, coarseScale, cv::INTER_NEAREST);
            appendRawFeatures(&combined,
                              runPass(coarseImage, coarseMask, coarseBudget, std::max(800, target * 2 / 3)),
                              1.0 / coarseScale);

            // 全图粗尺度建立稳定覆盖，原分辨率重叠瓦片补回整体缩放会丢失的细节。
            const int tileSize = maxDimension;
            const int overlap = std::clamp(tileSize / 16, 64, 256);
            const int step = std::max(256, tileSize - overlap);
            const int columns = std::max(1, (input.grayImage.cols - overlap + step - 1) / step);
            const int rows = std::max(1, (input.grayImage.rows - overlap + step - 1) / step);
            const int tileCount = columns * rows;
            const int fineBudget =
                runtime.maxKeypoints > 0 ? std::max(512, runtime.maxKeypoints / std::max(1, tileCount)) : 0;
            for (int row = 0; row < rows; ++row)
            {
                for (int column = 0; column < columns; ++column)
                {
                    const int x = std::min(column * step, std::max(0, input.grayImage.cols - tileSize));
                    const int y = std::min(row * step, std::max(0, input.grayImage.rows - tileSize));
                    const cv::Rect roi(x,
                                       y,
                                       std::min(tileSize, input.grayImage.cols - x),
                                       std::min(tileSize, input.grayImage.rows - y));
                    const cv::Mat tileMask = input.validMask.empty() ? cv::Mat() : input.validMask(roi);
                    appendRawFeatures(
                        &combined,
                        runPass(input.grayImage(roi), tileMask, fineBudget, std::max(256, fineBudget / 2)),
                        1.0,
                        cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
                }
            }
            return combined;
        }

        SiftRawFeatures extractSinglePass(const ImageFeatureInput& input,
                                          const ImageMatchingRuntimeConfig& runtime,
                                          SiftComputeBackend backend)
        {
            if (backend != SiftComputeBackend::Cpu)
            {
                SiftExtractionRequest request;
                request.image = input.grayImage;
                request.maximumFeatures = std::max(1024, detectorLimit(runtime.maxKeypoints));
                request.contrastThreshold = backend == SiftComputeBackend::Cuda ? runtime.siftDetectionThreshold
                                                                                : runtime.siftContrastThreshold;
                request.deviceIndex = runtime.cudaDevice;
                return extractSiftOnGpu(backend, request);
            }
            return extractCpuPass(
                input.grayImage, input.validMask, runtime.maxKeypoints, runtime.siftContrastThreshold);
        }

        void rootNormalize(cv::Mat row)
        {
            const double l1 = cv::norm(row, cv::NORM_L1);
            if (l1 <= 1e-12)
            {
                return;
            }
            row /= l1;
            cv::sqrt(row, row);
            const double l2 = cv::norm(row, cv::NORM_L2);
            if (l2 > 1e-12)
            {
                row /= l2;
            }
        }

        FeatureSet selectAndConvert(const ImageFeatureInput& input,
                                    const ImageMatchingRuntimeConfig& runtime,
                                    const SiftRawFeatures& raw,
                                    SiftComputeBackend backend)
        {
            std::vector<int> ranked;
            ranked.reserve(raw.keypoints.size());
            for (int index = 0; index < static_cast<int>(raw.keypoints.size()); ++index)
            {
                if (pointIsAllowed(input, raw.keypoints[static_cast<std::size_t>(index)], runtime))
                {
                    ranked.push_back(index);
                }
            }
            std::stable_sort(ranked.begin(),
                             ranked.end(),
                             [&](int left, int right)
                             {
                                 const float leftResponse = raw.keypoints[static_cast<std::size_t>(left)].response;
                                 const float rightResponse = raw.keypoints[static_cast<std::size_t>(right)].response;
                                 return leftResponse == rightResponse ? left < right : leftResponse > rightResponse;
                             });

            std::vector<int> selected;
            selected.reserve(ranked.size());
            const int limit = runtime.maxKeypoints > 0 ? runtime.maxKeypoints : static_cast<int>(ranked.size());
            const int gridColumns = runtime.adaptiveSift ? 8 : 1;
            const int gridRows = runtime.adaptiveSift ? 8 : 1;
            const int perCell = std::max(1, (limit + gridColumns * gridRows - 1) / (gridColumns * gridRows));
            std::vector<int> cellCounts(static_cast<std::size_t>(gridColumns * gridRows), 0);
            std::unordered_set<std::uint64_t> occupied;
            std::vector<int> deferred;
            for (const int index : ranked)
            {
                const cv::Point2f point = raw.keypoints[static_cast<std::size_t>(index)].pt;
                const int qx = std::max(0, static_cast<int>(std::lround(point.x / 3.0f)));
                const int qy = std::max(0, static_cast<int>(std::lround(point.y / 3.0f)));
                const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(qy)) << 32U) |
                                          static_cast<std::uint32_t>(qx);
                if (!occupied.insert(key).second)
                {
                    continue;
                }
                const int column = std::clamp(
                    static_cast<int>(point.x * gridColumns / std::max(1, input.grayImage.cols)), 0, gridColumns - 1);
                const int row = std::clamp(
                    static_cast<int>(point.y * gridRows / std::max(1, input.grayImage.rows)), 0, gridRows - 1);
                const int cell = row * gridColumns + column;
                if (cellCounts[static_cast<std::size_t>(cell)] < perCell)
                {
                    selected.push_back(index);
                    ++cellCounts[static_cast<std::size_t>(cell)];
                }
                else
                {
                    deferred.push_back(index);
                }
                if (static_cast<int>(selected.size()) >= limit)
                {
                    break;
                }
            }
            for (const int index : deferred)
            {
                if (static_cast<int>(selected.size()) >= limit)
                {
                    break;
                }
                selected.push_back(index);
            }

            FeatureSet features;
            features.computeBackend = siftBackendName(backend);
            features.imageWidth = input.originalWidth > 0 ? input.originalWidth : input.grayImage.cols;
            features.imageHeight = input.originalHeight > 0 ? input.originalHeight : input.grayImage.rows;
            features.keypoints.reserve(selected.size());
            features.scores.reserve(selected.size());
            if (!raw.descriptors.empty())
            {
                features.descriptors.create(static_cast<int>(selected.size()), raw.descriptors.cols, CV_32F);
            }
            for (int outputIndex = 0; outputIndex < static_cast<int>(selected.size()); ++outputIndex)
            {
                const int sourceIndex = selected[static_cast<std::size_t>(outputIndex)];
                cv::KeyPoint keypoint = raw.keypoints[static_cast<std::size_t>(sourceIndex)];
                if (input.coordinateScale > 0.0 && input.coordinateScale != 1.0)
                {
                    const float scale = static_cast<float>(input.coordinateScale);
                    keypoint.pt *= scale;
                    keypoint.size *= scale;
                }
                features.keypoints.push_back(keypoint);
                features.scores.push_back(keypoint.response);
                if (!features.descriptors.empty())
                {
                    raw.descriptors.row(sourceIndex).convertTo(features.descriptors.row(outputIndex), CV_32F);
                    if (runtime.rootSift)
                    {
                        rootNormalize(features.descriptors.row(outputIndex));
                    }
                }
            }
            return features;
        }

    } // namespace

    bool SiftFeatureExtractor::isBackendAvailable(SiftComputeBackend backend, int deviceIndex)
    {
        return isSiftBackendAvailable(backend, deviceIndex);
    }

    SiftComputeBackend SiftFeatureExtractor::resolveBackend(SiftComputeBackend requested, int deviceIndex)
    {
        return resolveSiftBackend(requested, deviceIndex);
    }

    FeatureSet SiftFeatureExtractor::extract(const ImageFeatureInput& input, const ImageMatchingRuntimeConfig& runtime)
    {
        if (input.grayImage.empty() || input.grayImage.type() != CV_8U)
        {
            throw std::invalid_argument("SIFT requires a non-empty CV_8U grayscale image");
        }
        if (!input.validMask.empty() &&
            (input.validMask.type() != CV_8U || input.validMask.size() != input.grayImage.size()))
        {
            throw std::invalid_argument("SIFT valid mask must be CV_8U and match the input image size");
        }

        const SiftComputeBackend backend = resolveBackend(runtime.siftBackend, runtime.cudaDevice);
        const SiftRawFeatures raw = runtime.adaptiveSift ? extractAdaptive(input, runtime, backend)
                                                         : extractSinglePass(input, runtime, backend);
        return selectAndConvert(input, runtime, raw, backend);
    }

} // namespace xjw::image_matching
