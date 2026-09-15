#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    // =============================================================================
    // 辅助：旋转矩阵行列式
    // =============================================================================
    double pipeline_detail::det3(const double* R)
    {
        return R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) +
               R[2] * (R[3] * R[7] - R[4] * R[6]);
    }

    void MvsPipelineService::clearFrameCaches()
    {
        _frameCaches.clear();
        _visibilityBits.clear();
        _visibilityAdjacency.clear();
        _visibilityWordCount = 0;
        _frameCachesReady = false;
    }

    bool MvsPipelineService::isSparsePointVisibleInFrame(int viewIdx, size_t pointIndex) const
    {
        if (!_frameCachesReady || viewIdx < 0 || viewIdx >= static_cast<int>(_frameCaches.size()) ||
            _visibilityWordCount == 0 || pointIndex >= _sparse.points.size())
        {
            return false;
        }

        const size_t word = pointIndex / 64;
        const size_t bit = pointIndex % 64;
        const size_t offset = static_cast<size_t>(viewIdx) * _visibilityWordCount + word;
        if (offset >= _visibilityBits.size())
        {
            return false;
        }
        return (_visibilityBits[offset] & (uint64_t{1} << bit)) != 0;
    }

    std::vector<int> MvsPipelineService::sourceViewIndicesForFrame(int refIdx, int maxSources) const
    {
        if (_frameCachesReady && refIdx >= 0 && refIdx < static_cast<int>(_frameCaches.size()) && maxSources > 0)
        {
            const auto& cached = _frameCaches[static_cast<size_t>(refIdx)].sourceViewIndices;
            const int count = std::min(maxSources, static_cast<int>(cached.size()));
            return std::vector<int>(cached.begin(), cached.begin() + count);
        }

        return selectMvsSourceViewIndices(_views, _sparse, refIdx, maxSources);
    }

    std::vector<size_t> MvsPipelineService::visibleSparsePointIndicesForFrame(int refIdx,
                                                                              const std::vector<int>& sourceIndices,
                                                                              int minSourceViews) const
    {
        if (!_frameCachesReady || refIdx < 0 || refIdx >= static_cast<int>(_frameCaches.size()))
        {
            return collectMvsVisibleSparsePointIndices(_views, _sparse, refIdx, sourceIndices, minSourceViews);
        }

        const auto& cache = _frameCaches[static_cast<size_t>(refIdx)];
        const auto& refVisible = cache.visiblePointIndices;
        if (sourceIndices.empty() || minSourceViews <= 0)
        {
            return refVisible;
        }

        auto sourceIndicesMatchCachedPrefix = [&cache, &sourceIndices]()
        {
            if (sourceIndices.size() != cache.sourceViewIndices.size())
            {
                return false;
            }
            return std::equal(sourceIndices.begin(), sourceIndices.end(), cache.sourceViewIndices.begin());
        };
        if (minSourceViews <= 1 && sourceIndicesMatchCachedPrefix())
        {
            return cache.sourceSharedPointIndices;
        }

        std::vector<size_t> filtered;
        filtered.reserve(refVisible.size());
        for (size_t pointIndex : refVisible)
        {
            int sourceVisible = 0;
            for (int sourceIdx : sourceIndices)
            {
                if (sourceIdx < 0 || sourceIdx >= static_cast<int>(_views.size()) || sourceIdx == refIdx)
                {
                    continue;
                }
                if (isSparsePointVisibleInFrame(sourceIdx, pointIndex))
                {
                    ++sourceVisible;
                    if (sourceVisible >= minSourceViews)
                    {
                        filtered.push_back(pointIndex);
                        break;
                    }
                }
            }
        }
        return filtered;
    }
} // namespace xjw::mvs
