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
