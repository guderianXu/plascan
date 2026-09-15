#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    // =============================================================================
    bool MvsPipelineService::estimateDepthRange(int refIdx,
                                                float& zNear,
                                                float& zFar,
                                                const std::vector<int>& sourceIndices) const
    {
        const int minSourceViews = sourceIndices.empty() ? 0 : 1;
        std::vector<size_t> visiblePointIndices =
            visibleSparsePointIndicesForFrame(refIdx, sourceIndices, minSourceViews);
        if (visiblePointIndices.size() < 5 && minSourceViews > 0)
        {
            visiblePointIndices = visibleSparsePointIndicesForFrame(refIdx, {}, 0);
            LOG_DEBUG(
                "[MVS][帧 %d][深度范围] 共视稀疏点不足，回退参考帧可见点 (%zu)", refIdx, visiblePointIndices.size());
        }
        return estimateDepthRangeFromVisiblePoints(refIdx, visiblePointIndices, zNear, zFar);
    }

    bool MvsPipelineService::estimateDepthRangeFromVisiblePoints(int refIdx,
                                                                 const std::vector<size_t>& visiblePointIndices,
                                                                 float& zNear,
                                                                 float& zFar) const
    {
        const CameraView& ref = _views[refIdx];
        const FramePinholeCamera cam = mvsPinholeCamera(ref.camera);

        std::vector<float> depths;
        depths.reserve(visiblePointIndices.size());

        for (size_t pointIndex : visiblePointIndices)
        {
            if (pointIndex >= _sparse.points.size())
            {
                continue;
            }
            const auto& pt = _sparse.points[pointIndex];
            const double world[3] = {pt[0], pt[1], pt[2]};
            double camera_point[3] = {0.0, 0.0, 0.0};
            cam.worldToCamera(world, camera_point);
            if (camera_point[2] > 0.0)
            {
                depths.push_back(static_cast<float>(camera_point[2]));
            }
        }

        if (depths.size() < 5)
        {
            // 没有足够的稀疏点——使用全局最大相机基线估算深度范围。
            // 关键：使用「全局最大基线」（所有相机对之间的最大距离），
            // 而非 per-camera 最大基线，以保证所有帧使用一致的 zNear/zFar，
            // 防止深度图均值差异过大导致融合一致性检查全部失败。
            float maxBaseline = 0.f;
            const int NVall = static_cast<int>(_views.size());
            for (int ia = 0; ia < NVall; ++ia)
            {
                for (int ib = ia + 1; ib < NVall; ++ib)
                {
                    const CameraBaseline baseline = CameraBaseline::evaluate(_views[ia].camera, _views[ib].camera);
                    if (baseline.isValid())
                    {
                        maxBaseline = std::max(maxBaseline, static_cast<float>(baseline.length()));
                    }
                }
            }
            if (maxBaseline > 1e-3f)
            {
                // 航空摄影测量：典型场景深度 ≈ 基线的 0.5× ~ 100×
                zNear = maxBaseline * 0.1f;
                zFar = maxBaseline * 100.f;
                LOG_DEBUG("[MVS][深度范围] 回退全局基线: baseline=%.4f zNear=%.4f zFar=%.4f", maxBaseline, zNear, zFar);
            }
            else
            {
                // 实在无法估计
                float dx = _sparse.maxPt[0] - _sparse.minPt[0];
                float dy = _sparse.maxPt[1] - _sparse.minPt[1];
                float dz = _sparse.maxPt[2] - _sparse.minPt[2];
                float diag = std::sqrt(dx * dx + dy * dy + dz * dz);
                zNear = 0.1f;
                zFar = diag > 0 ? diag * 3.f : 100.f;
                LOG_DEBUG("[MVS][深度范围] 回退 AABB: diag=%.4f zNear=%.4f zFar=%.4f", diag, zNear, zFar);
            }
            return true;
        }

        std::sort(depths.begin(), depths.end());
        size_t n = depths.size();

        // 使用 IQR (四分位距) 剔除离群深度值，比固定百分位更鲁棒
        float Q1 = depths[n / 4];
        float Q3 = depths[n * 3 / 4];
        float IQR = Q3 - Q1;
        float lowerFence = Q1 - 1.5f * IQR;
        float upperFence = Q3 + 1.5f * IQR;

        // 在 fence 范围内重新取 5%/95% 分位
        std::vector<float> inlierDepths;
        inlierDepths.reserve(n);
        for (float d : depths)
        {
            if (d >= lowerFence && d <= upperFence)
                inlierDepths.push_back(d);
        }
        if (inlierDepths.size() < 3)
            inlierDepths = depths; // fallback

        size_t ni = inlierDepths.size();
        zNear = inlierDepths[static_cast<size_t>(ni * 0.02f)] * _config.zNearScale;
        zFar = inlierDepths[static_cast<size_t>(ni * 0.98f)] * _config.zFarScale;
        zNear = std::max(zNear, 0.01f);
        zFar = std::max(zFar, zNear + 0.1f);

        LOG_DEBUG("[MVS][帧 %d][深度范围] Q1=%.4f Q3=%.4f IQR=%.4f "
                  "fence=[%.4f,%.4f] inliers=%zu/%zu visible=%zu",
                  refIdx,
                  Q1,
                  Q3,
                  IQR,
                  lowerFence,
                  upperFence,
                  inlierDepths.size(),
                  n,
                  visiblePointIndices.size());
        return true;
    }
} // namespace xjw::mvs
