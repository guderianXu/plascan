#pragma once

#include "LaserConstraintTypes.h"

#include <plapoint/search/spatial_kdtree.h>

#include <array>
#include <string>
#include <vector>

namespace xjw
{
namespace lidar
{

class LaserConstraintMap
{
public:
    bool loadPly(const std::string &path,
                 const LaserConstraintMapOptions &options,
                 std::string *errorMessage = nullptr);

    bool build(std::vector<LaserPlaneSample> samples,
               const LaserConstraintMapOptions &options,
               std::string *errorMessage = nullptr);

    std::size_t size() const
    {
        return m_samples.size();
    }

    bool empty() const
    {
        return m_samples.empty();
    }

    const std::vector<LaserPlaneSample> &samples() const
    {
        return m_samples;
    }

    bool nearestPlane(const std::array<double, 3> &query,
                      LaserPlaneSample *sample,
                      double *distanceMeters = nullptr) const;

private:
    void rebuildIndex();

    std::vector<LaserPlaneSample> m_samples;
    plapoint::search::SpatialKdTree<3, double> m_index;
};

} // namespace lidar
} // namespace xjw
