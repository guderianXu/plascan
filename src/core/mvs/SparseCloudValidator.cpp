// SparseCloudValidator.cpp
#include "SparseCloudValidator.h"
#include <fstream>
#include <sstream>
#include <cstdio>

namespace xjw
{
namespace mvs
{

bool SparseCloudValidator::validate(const std::string &cloudPath,
                                     SparseCloudStats  *stats,
                                     void              * /*unused*/,
                                     std::string       *errorMsg) const
{
    std::ifstream ifs(cloudPath);
    if (!ifs)
    {
        if (errorMsg)
        {
            *errorMsg = "无法打开点云文件: " + cloudPath;
        }
        return false;
    }

    std::array<float, 3> minPt = {1e18f, 1e18f, 1e18f};
    std::array<float, 3> maxPt = {-1e18f, -1e18f, -1e18f};
    int count = 0;
    std::string line;

    // 跳过 PLY 头部
    bool isPly = cloudPath.size() >= 4 &&
                 cloudPath.substr(cloudPath.size() - 4) == ".ply";
    if (isPly)
    {
        while (std::getline(ifs, line))
        {
            if (line == "end_header")
            {
                break;
            }
        }
    }

    while (std::getline(ifs, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream ss(line);
        float x, y, z;
        if (!(ss >> x >> y >> z))
        {
            continue;
        }
        for (int k = 0; k < 3; ++k)
        {
            float v = (k == 0 ? x : k == 1 ? y : z);
            if (v < minPt[k])
            {
                minPt[k] = v;
            }
            if (v > maxPt[k])
            {
                maxPt[k] = v;
            }
        }
        ++count;
    }

    if (stats)
    {
        stats->pointCount = count;
        stats->minPt = minPt;
        stats->maxPt = maxPt;
    }

    if (count < m_opts.minPoints)
    {
        if (errorMsg)
        {
            *errorMsg = "点数量不足: " + std::to_string(count) +
                        " < " + std::to_string(m_opts.minPoints);
        }
        return false;
    }
    return true;
}

} // namespace mvs
} // namespace xjw
