// =============================================================================
// 文件: SgmMatcher.cpp
// 功能: 半全局/多全局立体匹配器 CPU 实现
// =============================================================================
#include "SgmMatcher.h"
#include "DenseMatchBackend.h"
#include "SubpixelRefiner.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw::dense_match
{

    namespace
    {

        constexpr std::array<SgmDirection, 8> kSgmDirections = {
            {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}};

        bool isInside(int x, int y, int width, int height)
        {
            return x >= 0 && x < width && y >= 0 && y < height;
        }

        bool isUsablePathCost(float cost)
        {
            return std::isfinite(cost) && cost < kInvalidCost;
        }

        void aggregateDirection(
            CostVolume& aggregate, const CostVolume& costVolume, float p1, float p2, SgmDirection direction)
        {
            const int width = costVolume[0].cols;
            const int height = costVolume[0].rows;
            const int numDisparities = static_cast<int>(costVolume.size());
            std::vector<float> previous(static_cast<std::size_t>(numDisparities), kInvalidCost);
            std::vector<float> current(static_cast<std::size_t>(numDisparities), kInvalidCost);

            for (int startY = 0; startY < height; ++startY)
            {
                for (int startX = 0; startX < width; ++startX)
                {
                    const int predecessorX = startX - direction.x;
                    const int predecessorY = startY - direction.y;
                    if (isInside(predecessorX, predecessorY, width, height))
                    {
                        continue;
                    }

                    std::fill(previous.begin(), previous.end(), kInvalidCost);
                    bool hasPreviousPixel = false;
                    int x = startX;
                    int y = startY;
                    while (isInside(x, y, width, height))
                    {
                        float minimumPrevious = kInvalidCost;
                        if (hasPreviousPixel)
                        {
                            for (float cost : previous)
                            {
                                if (isUsablePathCost(cost))
                                {
                                    minimumPrevious = std::min(minimumPrevious, cost);
                                }
                            }
                        }

                        std::fill(current.begin(), current.end(), kInvalidCost);
                        for (int disparityIndex = 0; disparityIndex < numDisparities; ++disparityIndex)
                        {
                            const std::size_t index = static_cast<std::size_t>(disparityIndex);
                            if (!costVolume.isValid(index, y, x))
                            {
                                continue;
                            }

                            const float matchingCost = costVolume[index].at<float>(y, x);
                            if (!isUsablePathCost(matchingCost))
                            {
                                aggregate[index].at<float>(y, x) = kInvalidCost;
                                continue;
                            }

                            float pathCost = matchingCost;
                            if (hasPreviousPixel && isUsablePathCost(minimumPrevious))
                            {
                                float transitionCost = minimumPrevious + p2;
                                if (isUsablePathCost(previous[index]))
                                {
                                    transitionCost = std::min(transitionCost, previous[index]);
                                }
                                if (disparityIndex > 0 && isUsablePathCost(previous[index - 1]))
                                {
                                    transitionCost = std::min(transitionCost, previous[index - 1] + p1);
                                }
                                if (disparityIndex + 1 < numDisparities && isUsablePathCost(previous[index + 1]))
                                {
                                    transitionCost = std::min(transitionCost, previous[index + 1] + p1);
                                }
                                pathCost += transitionCost - minimumPrevious;
                            }

                            current[index] = pathCost;
                            float& aggregatedCost = aggregate[index].at<float>(y, x);
                            if (isUsablePathCost(aggregatedCost))
                            {
                                aggregatedCost += pathCost;
                            }
                        }

                        previous.swap(current);
                        hasPreviousPixel = true;
                        x += direction.x;
                        y += direction.y;
                    }
                }
            }
        }

    } // namespace

    CostVolume
    aggregateSgmCostVolume(const CostVolume& costVolume, int p1, int p2, const std::vector<SgmDirection>& directions)
    {
        if (costVolume.empty())
        {
            return {};
        }

        CostVolume aggregate(costVolume.minDisparity(), costVolume.maxDisparity(), costVolume[0].size());
        for (std::size_t disparityIndex = 0; disparityIndex < aggregate.size(); ++disparityIndex)
        {
            aggregate[disparityIndex].setTo(0.0f, aggregate.hypothesisValidMask(disparityIndex));
        }

        const float smallPenalty = static_cast<float>(std::max(0, p1));
        const float largePenalty = static_cast<float>(std::max(std::max(0, p2), p1));
        for (const SgmDirection direction : directions)
        {
            if ((direction.x == 0 && direction.y == 0) || std::abs(direction.x) > 1 || std::abs(direction.y) > 1)
            {
                continue;
            }
            aggregateDirection(aggregate, costVolume, smallPenalty, largePenalty, direction);
        }
        return aggregate;
    }

    SgmMatcher::SgmMatcher(const DenseMatchConfig& cfg) : _config(cfg)
    {
    }

    DisparityResult SgmMatcher::compute(const cv::Mat& left, const cv::Mat& right)
    {
        CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
        CV_Assert(left.size() == right.size());

        int imgW = left.cols;
        int imgH = left.rows;
        if (_config.maxDisparity <= _config.minDisparity)
        {
            DisparityResult empty;
            empty.disparity = cv::Mat();
            empty.confidence = cv::Mat();
            empty.validMask = cv::Mat();
            return empty;
        }
        const int numDisp =
            checkedCostVolumeBufferLayout(imgW, imgH, _config.minDisparity, _config.maxDisparity).numDisparities;

        const DenseMatchComputeBackend backend = resolveDenseMatchComputeBackend(_config);
        CostVolume C;
#ifdef DM_ENABLE_CUDA
        if (backend == DenseMatchComputeBackend::Cuda)
        {
            C = computeCostVolumeCUDA(left,
                                      right,
                                      _config.minDisparity,
                                      _config.maxDisparity,
                                      _config.corrKernelW,
                                      _config.corrKernelH,
                                      _config.costFunc,
                                      _config.cudaDevice);
        }
#endif
#ifdef DM_ENABLE_OPENCL
        if (backend == DenseMatchComputeBackend::OpenCl)
        {
            C = computeCostVolumeOpenCL(left,
                                        right,
                                        _config.minDisparity,
                                        _config.maxDisparity,
                                        _config.corrKernelW,
                                        _config.corrKernelH,
                                        _config.costFunc,
                                        _config.openClDevice);
        }
#endif
        if (backend == DenseMatchComputeBackend::Cpu)
        {
            C = computeCostVolume(left,
                                  right,
                                  _config.minDisparity,
                                  _config.maxDisparity,
                                  _config.corrKernelW,
                                  _config.corrKernelH,
                                  _config.costFunc,
                                  _config.numThreads);
        }

        int numDirs = _config.sgmDirections;
        if (numDirs < 1)
        {
            numDirs = 1;
        }
        if (numDirs > 8)
        {
            numDirs = 8;
        }

        std::vector<SgmDirection> directions;
        directions.reserve(static_cast<std::size_t>(numDirs));
        for (int directionIndex = 0; directionIndex < numDirs; ++directionIndex)
        {
            directions.push_back(kSgmDirections[static_cast<std::size_t>(directionIndex)]);
        }
        CostVolume aggregate = aggregateSgmCostVolume(C, _config.p1, _config.p2, directions);

#ifdef DM_ENABLE_CUDA
        if (backend == DenseMatchComputeBackend::Cuda)
        {
            return selectCostVolumeCUDA(aggregate, _config.subpixel, _config.cudaDevice);
        }
#endif
#ifdef DM_ENABLE_OPENCL
        if (backend == DenseMatchComputeBackend::OpenCl)
        {
            return selectCostVolumeOpenCL(aggregate, _config.subpixel, _config.openClDevice);
        }
#endif

        DisparityResult result;
        result.disparity = cv::Mat(imgH, imgW, CV_32FC1);
        result.confidence = cv::Mat(imgH, imgW, CV_32FC1);
        result.validMask = cv::Mat(imgH, imgW, CV_8UC1);
        const int threadCount = std::max(1, _config.numThreads);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threadCount)
#endif
        for (int y = 0; y < imgH; ++y)
        {
            for (int x = 0; x < imgW; ++x)
            {
                const BestDisparity selection = selectBestDisparity(aggregate, y, x);
                if (!selection.valid)
                {
                    result.disparity.at<float>(y, x) = 0.0f;
                    result.confidence.at<float>(y, x) = 0.0f;
                    result.validMask.at<uchar>(y, x) = 0;
                    continue;
                }

                result.disparity.at<float>(y, x) = static_cast<float>(selection.disparity);
                result.confidence.at<float>(y, x) = selection.confidence;
                result.validMask.at<uchar>(y, x) = 1;
            }
        }

        if (_config.subpixel != SubpixelMode::None)
        {
            SubpixelRefiner refiner(_config);
            result.disparity = refiner.refine(
                result.disparity, aggregate, _config.minDisparity, _config.maxDisparity, result.validMask);
        }

        return result;
    }

} // namespace xjw::dense_match
