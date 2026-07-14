#pragma once

#include "MarkerDetector.h"

#include <memory>
#include <optional>

namespace xjw::control_points
{

class MarkerDetectorFactory final
{
public:
    static std::unique_ptr<MarkerDetector> create(MarkerTargetFamily family);
    static std::optional<MarkerTargetFamily> parseFamily(const QString &name);
};

} // namespace xjw::control_points
