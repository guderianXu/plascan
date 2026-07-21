#pragma once

namespace xjw::control_points
{

enum class MarkerRole
{
    TieMarker,
    ControlPoint,
    CheckPoint
};

enum class ProjectionState
{
    ManualPinned,
    AutoDetected,
    Predicted,
    Blocked,
    Disabled
};

enum class ScaleBarRole
{
    Control,
    Check
};

} // namespace xjw::control_points
