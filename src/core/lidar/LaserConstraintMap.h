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
        return _samples.size();
    }

    bool empty() const
    {
        return _samples.empty();
    }

    const std::vector<LaserPlaneSample> &samples() const
    {
        return _samples;
    }

    bool nearestPlane(const std::array<double, 3> &query,
                      LaserPlaneSample *sample,
                      double *distanceMeters = nullptr) const;

private:
    void rebuildIndex();

    std::vector<LaserPlaneSample> _samples;
    plapoint::search::SpatialKdTree<3, double> _index;
};

} // namespace lidar
} // namespace xjw
