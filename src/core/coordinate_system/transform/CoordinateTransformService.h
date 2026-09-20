#pragma once

#include "coordinate_system/types/CoordinateFrames.h"

#include <array>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace xjw::coordinate_system
{

    class CoordinateTransformError : public std::runtime_error
    {
    public:
        explicit CoordinateTransformError(const std::string& message) : std::runtime_error(message)
        {
        }
    };

    class CoordinateTransformService
    {
    public:
        void registerFrame(CoordinateFrame frame);
        const CoordinateFrame& resolve(const CoordinateFrameId& id) const;
        const CoordinateFrame& frame(const CoordinateFrameId& id) const;

        std::array<double, 3> transformPoint(const CoordinateFrameId& source,
                                             const CoordinateFrameId& target,
                                             const std::array<double, 3>& point) const;

        std::array<double, 3> transformVector(const CoordinateFrameId& source,
                                              const CoordinateFrameId& target,
                                              const std::array<double, 3>& vector) const;

        RotationMatrix3d transformRotation(const CoordinateFrameId& source,
                                           const CoordinateFrameId& target,
                                           const RotationMatrix3d& rotation) const;

        std::string transformChainHash(const CoordinateFrameId& source, const CoordinateFrameId& target) const;

    private:
        std::vector<CoordinateFrameId> chainToRoot(const CoordinateFrameId& id) const;
        std::array<double, 3> toRoot(const CoordinateFrameId& id, const std::array<double, 3>& point) const;
        std::array<double, 3> vectorToRoot(const CoordinateFrameId& id, const std::array<double, 3>& vector) const;
        RotationMatrix3d rotationToRoot(const CoordinateFrameId& id) const;
        static std::array<double, 3> applyRotation(const RotationMatrix3d& rotation,
                                                   const std::array<double, 3>& vector);
        static RotationMatrix3d multiplyRotation(const RotationMatrix3d& first, const RotationMatrix3d& second);
        static RotationMatrix3d transposeRotation(const RotationMatrix3d& rotation);
        static double unitScale(LinearUnit from, LinearUnit to);

        std::unordered_map<std::string, CoordinateFrame> _frames;
    };

} // namespace xjw::coordinate_system
