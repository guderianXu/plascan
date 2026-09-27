#pragma once

#include "placoordinate/types/CoordinateIds.h"

#include <array>
#include <cstddef>
#include <optional>

namespace placoordinate
{

    using RotationMatrix3d = std::array<double, 9>;

    enum class CoordinateFrameKind
    {
        Ecef,
        Geodetic,
        BodyFixed,
        LocalEnu,
        LocalCartesian,
    };

    enum class LinearUnit
    {
        Metre,
        Kilometre,
        ProjectUnit,
    };

    enum class AngleUnit
    {
        Degree,
        Radian,
    };

    struct RigidTransform
    {
        RotationMatrix3d rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> translation{{0.0, 0.0, 0.0}};

        static RigidTransform identity() noexcept;
        bool isIdentity(double tolerance = 1.0e-12) const noexcept;
        bool isValid(double tolerance = 1.0e-8) const noexcept;
    };

    struct CoordinateFrame
    {
        CoordinateFrameId id;
        CoordinateFrameKind kind = CoordinateFrameKind::LocalCartesian;
        LinearUnit linearUnit = LinearUnit::Metre;
        AngleUnit angleUnit = AngleUnit::Radian;
        std::optional<CoordinateFrameId> parent;
        RigidTransform toParent = RigidTransform::identity();

        static CoordinateFrame create(CoordinateFrameId frameId,
                                      CoordinateFrameKind frameKind,
                                      LinearUnit unit,
                                      AngleUnit angles,
                                      std::optional<CoordinateFrameId> parentFrame,
                                      RigidTransform transform);
    };

} // namespace placoordinate
