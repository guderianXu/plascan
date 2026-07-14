#pragma once

#include "MvsTypes.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace xjw
{
namespace mvs
{

struct MvsSourceViewScore
{
    int viewIndex = -1;
    int commonVisiblePoints = 0;
    float medianTriangulationAngleDeg = 0.f;
    float score = 0.f;
};

inline int effectiveMvsViewWidth(const CameraView &view)
{
    if (view.imageWidth > 0)
    {
        return view.imageWidth;
    }
    return std::max(1, static_cast<int>(std::round(view.camera.principalX() * 2.0)));
}

inline int effectiveMvsViewHeight(const CameraView &view)
{
    if (view.imageHeight > 0)
    {
        return view.imageHeight;
    }
    return std::max(1, static_cast<int>(std::round(view.camera.principalY() * 2.0)));
}

inline bool isMvsSparsePointVisibleInView(const CameraView &view,
                                          const std::array<float, 3> &point,
                                          int overrideWidth = -1,
                                          int overrideHeight = -1,
                                          float *depthOut = nullptr)
{
    if (!view.camera.isValid())
    {
        return false;
    }

    const double world[3] = {point[0], point[1], point[2]};
    double pixel[2] = {0.0, 0.0};
    double positive_depth = 0.0;
    if (!view.camera.projectWorldPointWithDepth(world, pixel, positive_depth))
    {
        return false;
    }

    const int width = overrideWidth > 0 ? overrideWidth : effectiveMvsViewWidth(view);
    const int height = overrideHeight > 0 ? overrideHeight : effectiveMvsViewHeight(view);
    if (pixel[0] < 0.0 || pixel[1] < 0.0
        || pixel[0] >= static_cast<double>(width) || pixel[1] >= static_cast<double>(height))
    {
        return false;
    }

    if (depthOut)
    {
        *depthOut = static_cast<float>(positive_depth);
    }
    return true;
}

inline float mvsTriangulationAngleDeg(const CameraView &a,
                                      const CameraView &b,
                                      const std::array<float, 3> &point)
{
    const std::array<double, 3> ca = a.camera.cameraCenter();
    const std::array<double, 3> cb = b.camera.cameraCenter();

    const float ax = point[0] - static_cast<float>(ca[0]);
    const float ay = point[1] - static_cast<float>(ca[1]);
    const float az = point[2] - static_cast<float>(ca[2]);
    const float bx = point[0] - static_cast<float>(cb[0]);
    const float by = point[1] - static_cast<float>(cb[1]);
    const float bz = point[2] - static_cast<float>(cb[2]);
    const float an = std::sqrt(ax * ax + ay * ay + az * az);
    const float bn = std::sqrt(bx * bx + by * by + bz * bz);
    if (an <= 1e-6f || bn <= 1e-6f)
    {
        return 0.f;
    }
    float c = (ax * bx + ay * by + az * bz) / (an * bn);
    c = std::max(-1.f, std::min(1.f, c));
    constexpr float kPi = 3.14159265358979323846f;
    return std::acos(c) * 180.0f / kPi;
}

inline std::vector<size_t> collectMvsVisibleSparsePointIndices(const std::vector<CameraView> &views,
                                                               const SparseCloud &sparse,
                                                               int refIdx,
                                                               const std::vector<int> &sourceIndices,
                                                               int minSourceViews)
{
    std::vector<size_t> indices;
    if (refIdx < 0 || refIdx >= static_cast<int>(views.size()) || sparse.points.empty())
    {
        return indices;
    }

    indices.reserve(sparse.points.size());
    for (size_t pi = 0; pi < sparse.points.size(); ++pi)
    {
        const auto &point = sparse.points[pi];
        if (!isMvsSparsePointVisibleInView(views[refIdx], point))
        {
            continue;
        }

        int sourceVisible = 0;
        for (int sourceIdx : sourceIndices)
        {
            if (sourceIdx < 0 || sourceIdx >= static_cast<int>(views.size()) || sourceIdx == refIdx)
            {
                continue;
            }
            if (isMvsSparsePointVisibleInView(views[sourceIdx], point))
            {
                ++sourceVisible;
            }
        }

        if (sourceVisible >= minSourceViews)
        {
            indices.push_back(pi);
        }
    }
    return indices;
}

inline std::vector<MvsSourceViewScore> scoreMvsSourceViews(const std::vector<CameraView> &views,
                                                           const SparseCloud &sparse,
                                                           int refIdx)
{
    std::vector<MvsSourceViewScore> scores;
    if (refIdx < 0 || refIdx >= static_cast<int>(views.size()))
    {
        return scores;
    }

    scores.reserve(views.size() > 0 ? views.size() - 1 : 0);
    for (int viewIdx = 0; viewIdx < static_cast<int>(views.size()); ++viewIdx)
    {
        if (viewIdx == refIdx)
        {
            continue;
        }

        std::vector<float> angles;
        int common = 0;
        for (const auto &point : sparse.points)
        {
            if (!isMvsSparsePointVisibleInView(views[refIdx], point)
                || !isMvsSparsePointVisibleInView(views[viewIdx], point))
            {
                continue;
            }
            ++common;
            angles.push_back(mvsTriangulationAngleDeg(views[refIdx], views[viewIdx], point));
        }

        float medianAngle = 0.f;
        if (!angles.empty())
        {
            const auto mid = angles.begin() + static_cast<long>(angles.size() / 2);
            std::nth_element(angles.begin(), mid, angles.end());
            medianAngle = *mid;
        }

        const float angleWeight =
            medianAngle < 0.2f ? 0.25f :
            medianAngle > 35.0f ? 0.50f :
            1.0f;
        const float proximityPenalty = 0.001f * static_cast<float>(std::abs(viewIdx - refIdx));

        MvsSourceViewScore score;
        score.viewIndex = viewIdx;
        score.commonVisiblePoints = common;
        score.medianTriangulationAngleDeg = medianAngle;
        score.score = static_cast<float>(common) * angleWeight - proximityPenalty;
        scores.push_back(score);
    }

    std::sort(scores.begin(), scores.end(), [](const MvsSourceViewScore &a, const MvsSourceViewScore &b)
    {
        if (a.score != b.score)
        {
            return a.score > b.score;
        }
        if (a.commonVisiblePoints != b.commonVisiblePoints)
        {
            return a.commonVisiblePoints > b.commonVisiblePoints;
        }
        return a.viewIndex < b.viewIndex;
    });
    return scores;
}

inline std::vector<int> nearestMvsSourceViewIndices(int viewCount, int refIdx, int maxSources)
{
    std::vector<int> selected;
    if (viewCount <= 1 || refIdx < 0 || refIdx >= viewCount || maxSources <= 0)
    {
        return selected;
    }

    selected.reserve(static_cast<size_t>(maxSources));
    for (int delta = 1; delta <= viewCount && static_cast<int>(selected.size()) < maxSources; ++delta)
    {
        for (int sign : {-1, 1})
        {
            const int candidate = refIdx + sign * delta;
            if (candidate < 0 || candidate >= viewCount)
            {
                continue;
            }
            selected.push_back(candidate);
            if (static_cast<int>(selected.size()) >= maxSources)
            {
                break;
            }
        }
    }
    return selected;
}

inline std::vector<int> selectMvsSourceViewIndices(const std::vector<CameraView> &views,
                                                   const SparseCloud &sparse,
                                                   int refIdx,
                                                   int maxSources)
{
    if (maxSources <= 0)
    {
        return {};
    }

    const std::vector<MvsSourceViewScore> scores = scoreMvsSourceViews(views, sparse, refIdx);
    std::vector<int> selected;
    selected.reserve(static_cast<size_t>(maxSources));
    for (const auto &score : scores)
    {
        if (score.commonVisiblePoints <= 0 || score.score <= 0.f)
        {
            continue;
        }
        selected.push_back(score.viewIndex);
        if (static_cast<int>(selected.size()) >= maxSources)
        {
            break;
        }
    }

    if (selected.empty())
    {
        return nearestMvsSourceViewIndices(static_cast<int>(views.size()), refIdx, maxSources);
    }

    return selected;
}

} // namespace mvs
} // namespace xjw
