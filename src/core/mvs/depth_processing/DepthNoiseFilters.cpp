#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "DepthPostprocessor.h"
#include "Logger.h"

namespace xjw::mvs
{

    void DepthPostprocessor::applySparseSupportPrior(cv::Mat& depthMap,
                                                     cv::Mat& confidenceMap,
                                                     const cv::Mat& supportMask,
                                                     int refIdx)
    {
        if (depthMap.empty() || supportMask.empty())
        {
            return;
        }

        cv::Mat support;
        if (supportMask.type() == CV_8U)
        {
            support = supportMask;
        }
        else
        {
            supportMask.convertTo(support, CV_8U);
        }

        if (support.size() != depthMap.size())
        {
            cv::resize(support, support, depthMap.size(), 0, 0, cv::INTER_NEAREST);
        }

        const cv::Mat validDepth = depthMap > 0;
        const int beforeValid = cv::countNonZero(validDepth);
        if (beforeValid <= 0)
        {
            return;
        }

        cv::Mat unsupportedMask;
        cv::bitwise_and(validDepth, support == 0, unsupportedMask);
        const int unsupportedValid = cv::countNonZero(unsupportedMask);
        if (unsupportedValid <= 0)
        {
            return;
        }

        if (confidenceMap.empty() || confidenceMap.size() != depthMap.size() || confidenceMap.type() != CV_32F)
        {
            LOG_DEBUG("[MVS][帧 %d][稀疏支撑] support 外=%d/%d，置信图不可用，深度保持不变",
                      refIdx,
                      unsupportedValid,
                      beforeValid);
            return;
        }

        constexpr float kUnsupportedConfidenceScale = 0.75f;
        for (int y = 0; y < confidenceMap.rows; ++y)
        {
            float* confRow = confidenceMap.ptr<float>(y);
            const uint8_t* maskRow = unsupportedMask.ptr<uint8_t>(y);
            for (int x = 0; x < confidenceMap.cols; ++x)
            {
                if (maskRow[x] != 0)
                {
                    confRow[x] *= kUnsupportedConfidenceScale;
                }
            }
        }

        LOG_DEBUG("[MVS][帧 %d][稀疏支撑] support 外=%d/%d，置信度缩放=%.2f，深度保持不变",
                  refIdx,
                  unsupportedValid,
                  beforeValid,
                  kUnsupportedConfidenceScale);
    }

    int DepthPostprocessor::removeLocalDepthOutliers(cv::Mat& depthMap,
                                                     cv::Mat& confidenceMap,
                                                     int kernelSize,
                                                     float relDepthThreshold,
                                                     float maxRemovalRatio,
                                                     int refIdx,
                                                     int sameLayerRadiusPixels)
    {
        if (depthMap.empty() || depthMap.type() != CV_32F)
        {
            return 0;
        }
        if (kernelSize < 3 || relDepthThreshold <= 0.0f || maxRemovalRatio <= 0.0f)
        {
            return 0;
        }

        const cv::Mat validMask = depthMap > 0.0f;
        const int validBefore = cv::countNonZero(validMask);
        if (validBefore <= 0)
        {
            return 0;
        }

        const int medianKernel = std::clamp(kernelSize | 1, 3, 5);
        cv::Mat localMedian;
        cv::medianBlur(depthMap, localMedian, medianKernel);

        cv::Mat outlierMask = cv::Mat::zeros(depthMap.size(), CV_8U);
#if defined(HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int y = 0; y < depthMap.rows; ++y)
        {
            const float* depthRow = depthMap.ptr<float>(y);
            const float* medianRow = localMedian.ptr<float>(y);
            uint8_t* maskRow = outlierMask.ptr<uint8_t>(y);
            for (int x = 0; x < depthMap.cols; ++x)
            {
                const float depth = depthRow[x];
                const float medianDepth = medianRow[x];
                if (depth <= 0.0f || medianDepth <= 0.0f)
                {
                    continue;
                }

                const float relDiff = std::fabs(depth - medianDepth) / std::max(medianDepth, 1e-6f);
                if (relDiff > relDepthThreshold)
                {
                    // A real occlusion edge or thin silhouette has neighbours on
                    // its own depth layer even when the local median belongs to
                    // the other layer. An isolated spike does not. Preserve the
                    // former and remove only unsupported one-pixel hypotheses.
                    const int same_layer_radius = std::clamp(sameLayerRadiusPixels, 0, 2);
                    int same_layer_neighbors = 0;
                    for (int delta_y = -same_layer_radius; delta_y <= same_layer_radius; ++delta_y)
                    {
                        const int neighbor_y = y + delta_y;
                        if (neighbor_y < 0 || neighbor_y >= depthMap.rows)
                        {
                            continue;
                        }
                        const float* neighbor_row = depthMap.ptr<float>(neighbor_y);
                        for (int delta_x = -same_layer_radius; delta_x <= same_layer_radius; ++delta_x)
                        {
                            if (delta_x == 0 && delta_y == 0)
                            {
                                continue;
                            }
                            const int neighbor_x = x + delta_x;
                            if (neighbor_x < 0 || neighbor_x >= depthMap.cols)
                            {
                                continue;
                            }
                            const float neighbor_depth = neighbor_row[neighbor_x];
                            if (neighbor_depth <= 0.0f)
                            {
                                continue;
                            }
                            const float neighbor_difference =
                                std::fabs(neighbor_depth - depth) / std::max(depth, 1.0e-6f);
                            if (neighbor_difference <= relDepthThreshold)
                            {
                                ++same_layer_neighbors;
                            }
                        }
                    }
                    const int minimum_same_layer_neighbors = same_layer_radius > 0 ? 2 : 1;
                    if (same_layer_neighbors < minimum_same_layer_neighbors)
                    {
                        maskRow[x] = 255;
                    }
                }
            }
        }

        const int candidateCount = cv::countNonZero(outlierMask);
        if (candidateCount <= 0)
        {
            return 0;
        }

        const float removalRatio = static_cast<float>(candidateCount) / static_cast<float>(validBefore);
        if (removalRatio > maxRemovalRatio)
        {
            LOG_DEBUG("[MVS][帧 %d][后处理] 局部离群候选过多 %d/%d (%.1f%% > %.1f%%)，跳过",
                      refIdx,
                      candidateCount,
                      validBefore,
                      removalRatio * 100.0f,
                      maxRemovalRatio * 100.0f);
            return 0;
        }

        depthMap.setTo(0.0f, outlierMask);
        if (!confidenceMap.empty() && confidenceMap.size() == depthMap.size() && confidenceMap.type() == CV_32F)
        {
            confidenceMap.setTo(0.0f, outlierMask);
        }

        LOG_DEBUG("[MVS][帧 %d][后处理] 局部离群移除=%d/%d kernel=%d relative_threshold=%.2f",
                  refIdx,
                  candidateCount,
                  validBefore,
                  medianKernel,
                  relDepthThreshold);
        return candidateCount;
    }

    int DepthPostprocessor::removeSmallDepthComponents(
        cv::Mat& depthMap, cv::Mat& confidenceMap, int minComponentArea, float maxRemovalRatio, int refIdx)
    {
        if (depthMap.empty() || depthMap.type() != CV_32F)
        {
            return 0;
        }
        if (minComponentArea <= 1 || maxRemovalRatio <= 0.0f)
        {
            return 0;
        }

        const cv::Mat validMask = depthMap > 0.0f;
        const int validBefore = cv::countNonZero(validMask);
        if (validBefore <= 0)
        {
            return 0;
        }

        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int componentCount = cv::connectedComponentsWithStats(validMask, labels, stats, centroids, 8, CV_32S);

        std::vector<unsigned char> remove_labels(static_cast<std::size_t>(componentCount), 0);
        int candidateCount = 0;
        for (int label = 1; label < componentCount; ++label)
        {
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area <= 0 || area >= minComponentArea)
            {
                continue;
            }

            candidateCount += area;
            remove_labels[static_cast<std::size_t>(label)] = 1;
        }

        if (candidateCount <= 0)
        {
            return 0;
        }

        const float removalRatio = static_cast<float>(candidateCount) / static_cast<float>(validBefore);
        if (removalRatio > maxRemovalRatio)
        {
            LOG_DEBUG("[MVS][帧 %d][后处理] 小连通域候选过多 %d/%d (%.1f%% > %.1f%%)，跳过",
                      refIdx,
                      candidateCount,
                      validBefore,
                      removalRatio * 100.0f,
                      maxRemovalRatio * 100.0f);
            return 0;
        }

        cv::Mat removeMask = cv::Mat::zeros(depthMap.size(), CV_8U);
        cv::parallel_for_(cv::Range(0, labels.rows),
                          [&](const cv::Range& range)
                          {
                              for (int y = range.start; y < range.end; ++y)
                              {
                                  const int* label_row = labels.ptr<int>(y);
                                  unsigned char* mask_row = removeMask.ptr<unsigned char>(y);
                                  for (int x = 0; x < labels.cols; ++x)
                                  {
                                      const int label = label_row[x];
                                      if (label > 0 && remove_labels[static_cast<std::size_t>(label)] != 0)
                                      {
                                          mask_row[x] = 255;
                                      }
                                  }
                              }
                          });

        depthMap.setTo(0.0f, removeMask);
        if (!confidenceMap.empty() && confidenceMap.size() == depthMap.size() && confidenceMap.type() == CV_32F)
        {
            confidenceMap.setTo(0.0f, removeMask);
        }

        LOG_DEBUG("[MVS][帧 %d][后处理] 小连通域移除=%d/%d min_area=%d",
                  refIdx,
                  candidateCount,
                  validBefore,
                  minComponentArea);
        return candidateCount;
    }

} // namespace xjw::mvs
