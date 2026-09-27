#include "placoordinate/types/CoordinateFrames.h"

#include "placoordinate/types/CoordinateErrors.h"

#include <cmath>
#include <utility>

namespace placoordinate
{

    RigidTransform RigidTransform::identity() noexcept
    {
        return {};
    }

    bool RigidTransform::isIdentity(double tolerance) const noexcept
    {
        const RigidTransform expected = identity();
        for (std::size_t index = 0; index < rotation.size(); ++index)
        {
            if (std::abs(rotation[index] - expected.rotation[index]) > tolerance)
            {
                return false;
            }
        }
        for (double value : translation)
        {
            if (std::abs(value) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    bool RigidTransform::isValid(double tolerance) const noexcept
    {
        for (double value : rotation)
        {
            if (!std::isfinite(value))
            {
                return false;
            }
        }
        for (double value : translation)
        {
            if (!std::isfinite(value))
            {
                return false;
            }
        }

        const auto dotRow = [this](int first, int second)
        {
            return rotation[static_cast<std::size_t>(first)] * rotation[static_cast<std::size_t>(second)] +
                   rotation[static_cast<std::size_t>(first + 1)] * rotation[static_cast<std::size_t>(second + 1)] +
                   rotation[static_cast<std::size_t>(first + 2)] * rotation[static_cast<std::size_t>(second + 2)];
        };
        const double determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
                                   rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
                                   rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
        return std::abs(dotRow(0, 0) - 1.0) <= tolerance && std::abs(dotRow(3, 3) - 1.0) <= tolerance &&
               std::abs(dotRow(6, 6) - 1.0) <= tolerance && std::abs(dotRow(0, 3)) <= tolerance &&
               std::abs(dotRow(0, 6)) <= tolerance && std::abs(dotRow(3, 6)) <= tolerance &&
               std::abs(determinant - 1.0) <= tolerance;
    }

    CoordinateFrame CoordinateFrame::create(CoordinateFrameId frameId,
                                            CoordinateFrameKind frameKind,
                                            LinearUnit unit,
                                            AngleUnit angles,
                                            std::optional<CoordinateFrameId> parentFrame,
                                            RigidTransform transform)
    {
        if (!transform.isValid())
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidFrame,
                                            "coordinate frame transform must be a finite proper rotation");
        }
        if (!parentFrame && !transform.isIdentity())
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidFrame,
                                            "root coordinate frame must use the identity transform");
        }
        return CoordinateFrame{std::move(frameId), frameKind, unit, angles, std::move(parentFrame), transform};
    }

} // namespace placoordinate
