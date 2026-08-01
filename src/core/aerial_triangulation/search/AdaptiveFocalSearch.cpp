/**
 * @file AdaptiveFocalSearch.cpp
 * @brief 旧版轻量焦距候选择优器的实现。
 *
 * 正式空三管线使用信息更完整的 SfmSearchPolicy；本类保留给窄接口调用和回归测试。
 * 它只比较已有摘要，不运行 SfM，也不修改相机。
 */

#include "search/AdaptiveFocalSearch.h"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace xjw::aerial_triangulation
{

int AdaptiveFocalSearch::selectBestCandidate(const QVector<AdaptiveFocalCandidate> &candidates,
                                             int totalImages)
{
    int bestIndex = -1;
    const int safeTotalImages = std::max(1, totalImages);
    // tuple 按字典序比较：成功、注册率、点数、低 RMS、接近默认焦距依次优先。
    auto score = [safeTotalImages](const AdaptiveFocalCandidate &candidate)
    {
        const double registrationRatio = static_cast<double>(std::max(0, candidate.registeredImages)) /
            static_cast<double>(safeTotalImages);
        const double rms = std::isfinite(candidate.meanReprojectionError)
            ? std::max(0.0, candidate.meanReprojectionError)
            : 1.0e9;
        return std::tuple<bool, double, int, double, double>{
            candidate.success,
            registrationRatio,
            std::max(0, candidate.points3D),
            -rms,
            -std::abs(candidate.focalScale - 1.0),
        };
    };

    for (int index = 0; index < candidates.size(); ++index)
    {
        if (bestIndex < 0 || score(candidates.at(bestIndex)) < score(candidates.at(index)))
        {
            bestIndex = index;
        }
    }
    return bestIndex;
}

} // namespace xjw::aerial_triangulation
