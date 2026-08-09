#pragma once

#include <string>
#include <vector>

namespace xjw
{
namespace lidar
{

enum class IsisControlPointType
{
    Unknown,
    Free,
    Constrained,
    Fixed
};

/**
 * @brief ISIS control-measure coordinate kept in the native ISIS convention.
 *
 * ISIS control networks place the upper-left pixel center at (1.0, 1.0).
 * Consumers using CSM must subtract 0.5 from sample and line; consumers using
 * OpenCV must subtract 1.0. Keeping the file convention here prevents a parser
 * from silently changing serialized control-network values.
 */
struct IsisControlMeasure
{
    std::string serialNumber;
    double samplePixels = 0.0;
    double linePixels = 0.0;
    bool ignored = false;
};

struct IsisControlPoint
{
    std::string id;
    IsisControlPointType type = IsisControlPointType::Unknown;
    std::vector<IsisControlMeasure> measures;
    bool ignored = false;
};

struct IsisControlNetwork
{
    std::string networkId;
    std::string targetName;
    std::vector<IsisControlPoint> points;

    bool validate(std::string *errorMessage = nullptr) const;
    int usableMeasureCount() const;
};

/**
 * @brief Parse the textual ISIS PVL control-network subset used by jigsaw.
 *
 * The parser deliberately accepts unknown metadata while requiring each
 * ControlPoint and ControlMeasure to contain the geometry needed by PlaScan.
 */
bool parseIsisControlNetworkPvl(const std::string &pvl,
                                IsisControlNetwork *network,
                                std::string *errorMessage = nullptr);

bool loadIsisControlNetworkPvlFile(const std::string &path,
                                   IsisControlNetwork *network,
                                   std::string *errorMessage = nullptr);

} // namespace lidar
} // namespace xjw
