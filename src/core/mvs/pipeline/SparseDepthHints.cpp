#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    // =============================================================================
    cv::Mat MvsPipelineService::buildHintDepth(int refIdx, int W, int H, const std::vector<int>& sourceIndices) const
    {
        const int minSourceViews = sourceIndices.empty() ? 0 : 1;
        std::vector<size_t> visiblePointIndices =
            visibleSparsePointIndicesForFrame(refIdx, sourceIndices, minSourceViews);
        if (visiblePointIndices.empty() && minSourceViews > 0)
        {
            visiblePointIndices = visibleSparsePointIndicesForFrame(refIdx, {}, 0);
            LOG_DEBUG("[MVS][帧 %d][稀疏引导] 共视点为空，回退参考帧可见点 (%zu)", refIdx, visiblePointIndices.size());
        }

        return buildHintDepthFromVisiblePoints(refIdx, W, H, visiblePointIndices);
    }

    cv::Mat MvsPipelineService::buildHintDepthFromVisiblePoints(int refIdx,
                                                                int W,
                                                                int H,
                                                                const std::vector<size_t>& visiblePointIndices) const
    {
        if (refIdx < 0 || refIdx >= static_cast<int>(_views.size()))
        {
            return cv::Mat();
        }

        return buildHintDepthForCamera(refIdx, mvsPinholeCamera(_views[refIdx].camera), W, H, visiblePointIndices);
    }

    cv::Mat MvsPipelineService::buildHintDepthForCamera(int refIdx,
                                                        const FramePinholeCamera& camera,
                                                        int W,
                                                        int H,
                                                        const std::vector<size_t>& visiblePointIndices) const
    {
        const std::vector<ProjectedSparseDepthSample> samples =
            collectProjectedSparseDepthSamples(_sparse, camera, W, H, visiblePointIndices);
        return buildHintDepthFromProjectedSamples(refIdx, W, H, samples);
    }

    std::vector<ProjectedSparseDepthSample>
    MvsPipelineService::collectProjectedSparseDepthSamples(const SparseCloud& sparse,
                                                           const FramePinholeCamera& camera,
                                                           int imageWidth,
                                                           int imageHeight,
                                                           const std::vector<size_t>& visiblePointIndices)
    {
        std::vector<ProjectedSparseDepthSample> samples;
        if (sparse.points.empty() || imageWidth <= 0 || imageHeight <= 0 || !camera.isValid())
        {
            return samples;
        }

        std::vector<ProjectedSparseDepthSample> projectedCandidates;
        projectedCandidates.reserve(visiblePointIndices.size());
        std::vector<float> depthQuantileSamples;
        depthQuantileSamples.reserve(std::min(visiblePointIndices.size(), kMaxProjectedDepthQuantileSamples));
        const size_t quantileSampleStride = visiblePointIndices.size() > kMaxProjectedDepthQuantileSamples
                                                ? (visiblePointIndices.size() + kMaxProjectedDepthQuantileSamples - 1) /
                                                      kMaxProjectedDepthQuantileSamples
                                                : 1;
        size_t validDepthOrdinal = 0;
        for (size_t pointIndex : visiblePointIndices)
        {
            if (pointIndex >= sparse.points.size())
            {
                continue;
            }
            const auto& pt = sparse.points[pointIndex];
            const double world[3] = {pt[0], pt[1], pt[2]};
            double pixel[2] = {0.0, 0.0};
            double depth = 0.0;
            if (!camera.projectWorldPointWithDepth(world, pixel, depth) || !std::isfinite(depth) || pixel[0] < 0.0 ||
                pixel[0] >= static_cast<double>(imageWidth) || pixel[1] < 0.0 ||
                pixel[1] >= static_cast<double>(imageHeight))
            {
                continue;
            }

            ProjectedSparseDepthSample candidate;
            candidate.uNorm = static_cast<float>(pixel[0] / static_cast<double>(imageWidth));
            candidate.vNorm = static_cast<float>(pixel[1] / static_cast<double>(imageHeight));
            candidate.depth = static_cast<float>(depth);
            projectedCandidates.push_back(candidate);

            if ((validDepthOrdinal % quantileSampleStride) == 0 &&
                depthQuantileSamples.size() < kMaxProjectedDepthQuantileSamples)
            {
                depthQuantileSamples.push_back(candidate.depth);
            }
            ++validDepthOrdinal;
        }

        float depthLo = 0.f, depthHi = 1e30f;
        if (depthQuantileSamples.size() >= 4)
        {
            auto q1It = depthQuantileSamples.begin() + static_cast<std::ptrdiff_t>(depthQuantileSamples.size() / 4);
            std::nth_element(depthQuantileSamples.begin(), q1It, depthQuantileSamples.end());
            const float Q1 = *q1It;

            auto q3It = depthQuantileSamples.begin() + static_cast<std::ptrdiff_t>(depthQuantileSamples.size() * 3 / 4);
            std::nth_element(depthQuantileSamples.begin(), q3It, depthQuantileSamples.end());
            const float Q3 = *q3It;

            float IQR = Q3 - Q1;
            depthLo = Q1 - 1.5f * IQR;
            depthHi = Q3 + 1.5f * IQR;
        }

        samples.reserve(projectedCandidates.size());
        for (const ProjectedSparseDepthSample& candidate : projectedCandidates)
        {
            if (candidate.depth < depthLo || candidate.depth > depthHi || !std::isfinite(candidate.depth))
            {
                continue;
            }

            samples.push_back(candidate);
        }

        return samples;
    }

    cv::Mat MvsPipelineService::buildHintDepthFromProjectedSamples(
        int refIdx, int W, int H, const std::vector<ProjectedSparseDepthSample>& samples)
    {
        if (samples.empty() || W <= 0 || H <= 0)
        {
            return cv::Mat();
        }

        cv::Mat hint = buildSparseSeedDepthFromProjectedSamples(refIdx, W, H, samples);
        if (hint.empty())
        {
            return cv::Mat();
        }

        // 第二步：限距离膨胀——仅将稀疏种子传播到 maxHintRadius 像素范围内
        // 不做全图扫线传播，以免把远离稀疏点的像素也强制初始化为"错误 hint"：
        //   GPU 初始化对有 hint 的像素使用 hint±30% 的窄范围，
        //   若 hint 覆盖了距离真实深度很远的区域，PatchMatch 将无法逃脱。
        // 超出 maxHintRadius 的像素保持 hint=0 → GPU 用全范围随机初始化。
        const int seedHintCnt = cv::countNonZero(hint > 0);
        if (seedHintCnt <= 0)
        {
            LOG_DEBUG("[MVS][帧 %d][稀疏引导] 可见点=%zu，无有效种子，跳过传播", refIdx, samples.size());
            return cv::Mat();
        }

        const int adaptiveHintRadius =
            seedHintCnt > 0
                ? std::clamp(static_cast<int>(std::sqrt(static_cast<float>(W * H) / seedHintCnt) * 0.5f), 16, 48)
                : 0;
        const int maxHintRadius = adaptiveHintRadius;
        // 距离变换限距离膨胀：用 OpenCV 的优化扫描求每个像素最近的稀疏 seed，
        // 避免手写多轮 at<> 全图扫描。只在 maxHintRadius 内传播，远处仍保留 0。
        {
            cv::Mat seedDistanceMask(H, W, CV_8U, cv::Scalar(255));
            seedDistanceMask.setTo(0, hint > 0);

            cv::Mat distanceMap;
            cv::Mat nearestSeedLabels;
            cv::distanceTransform(
                seedDistanceMask, distanceMap, nearestSeedLabels, cv::DIST_L1, 3, cv::DIST_LABEL_PIXEL);

            double maxLabelValue = 0.0;
            cv::minMaxLoc(nearestSeedLabels, nullptr, &maxLabelValue);
            std::vector<float> labelDepths(static_cast<size_t>(std::max(0.0, maxLabelValue)) + 1, 0.0f);
            for (int row = 0; row < H; ++row)
            {
                const float* hintRow = hint.ptr<float>(row);
                const int* labelRow = nearestSeedLabels.ptr<int>(row);
                for (int col = 0; col < W; ++col)
                {
                    const float depth = hintRow[col];
                    const int label = labelRow[col];
                    if (depth <= 0.0f || label <= 0)
                    {
                        continue;
                    }

                    float& labelDepth = labelDepths[static_cast<size_t>(label)];
                    if (labelDepth == 0.0f || depth < labelDepth)
                    {
                        labelDepth = depth;
                    }
                }
            }

            for (int row = 0; row < H; ++row)
            {
                float* hintRow = hint.ptr<float>(row);
                const float* distanceRow = distanceMap.ptr<float>(row);
                const int* labelRow = nearestSeedLabels.ptr<int>(row);
                for (int col = 0; col < W; ++col)
                {
                    if (hintRow[col] > 0.0f || distanceRow[col] > static_cast<float>(maxHintRadius))
                    {
                        continue;
                    }

                    const int label = labelRow[col];
                    if (label <= 0 || static_cast<size_t>(label) >= labelDepths.size())
                    {
                        continue;
                    }

                    const float depth = labelDepths[static_cast<size_t>(label)];
                    if (depth > 0.0f)
                    {
                        hintRow[col] = depth;
                    }
                }
            }
        }

        int hintCnt = cv::countNonZero(hint > 0);
        LOG_DEBUG("[MVS][帧 %d][稀疏引导] visible=%zu seeds=%d radius=%d coverage=%d/%d (%.1f%%)",
                  refIdx,
                  samples.size(),
                  seedHintCnt,
                  maxHintRadius,
                  hintCnt,
                  W * H,
                  100.0f * hintCnt / (W * H));
        return hint;
    }
} // namespace xjw::mvs
