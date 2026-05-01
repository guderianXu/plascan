#pragma once

#include "DisparityTriangulator.h"
#include <string>

namespace xjw
{
namespace mvs
{

class PointCloudTifIO
{
public:
    static bool writeTif(const std::string &path,
                         const TriangulationResult &triResult,
                         std::string *errorMsg = nullptr);

    static bool readTif(const std::string &path,
                        TriangulationResult &triResult,
                        std::string *errorMsg = nullptr);
};

} // namespace mvs
} // namespace xjw
