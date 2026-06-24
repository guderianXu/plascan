// SparseCloudValidator.cpp
#include "SparseCloudValidator.h"

#include <plapoint/io/ply_io.h>
#include <plapoint/io/xyz_io.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>

namespace xjw
{
namespace mvs
{

namespace
{

using SparsePlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

bool endsWithIgnoreCase(const std::string &text, const std::string &suffix)
{
    if (text.size() < suffix.size())
    {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), text.rbegin(),
                      [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a)) ==
                                 std::tolower(static_cast<unsigned char>(b));
                      });
}

} // namespace

bool SparseCloudValidator::validate(const std::string &cloudPath,
                                     SparseCloudStats  *stats,
                                     void              * /*unused*/,
                                     std::string       *errorMsg) const
{
    std::shared_ptr<SparsePlaCloud> cloud;
    try
    {
        if (endsWithIgnoreCase(cloudPath, ".ply"))
        {
            cloud = plapoint::io::readPly<float>(cloudPath);
        }
        else
        {
            cloud = plapoint::io::readXyz<float>(cloudPath);
        }
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = "点云读取失败: " + std::string(e.what());
        return false;
    }

    std::array<float, 3> minPt = {1e18f, 1e18f, 1e18f};
    std::array<float, 3> maxPt = {-1e18f, -1e18f, -1e18f};
    const int count = cloud ? static_cast<int>(cloud->size()) : 0;
    bool hasFinitePoint = false;
    if (cloud)
    {
        const auto &points = cloud->points();
        for (std::size_t i = 0; i < cloud->size(); ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            const float values[3] = {
                points.getValue(row, 0),
                points.getValue(row, 1),
                points.getValue(row, 2)
            };
            if (!std::isfinite(values[0]) || !std::isfinite(values[1]) || !std::isfinite(values[2]))
            {
                continue;
            }
            hasFinitePoint = true;
            for (int k = 0; k < 3; ++k)
            {
                minPt[k] = std::min(minPt[k], values[k]);
                maxPt[k] = std::max(maxPt[k], values[k]);
            }
        }
    }

    if (stats)
    {
        stats->pointCount = count;
        stats->minPt = minPt;
        stats->maxPt = maxPt;
    }

    if (!hasFinitePoint)
    {
        if (errorMsg)
        {
            *errorMsg = "点云中没有有限坐标";
        }
        return false;
    }

    if (count < _options.minPoints)
    {
        if (errorMsg)
        {
            *errorMsg = "点数量不足: " + std::to_string(count) +
                        " < " + std::to_string(_options.minPoints);
        }
        return false;
    }
    return true;
}

} // namespace mvs
} // namespace xjw
