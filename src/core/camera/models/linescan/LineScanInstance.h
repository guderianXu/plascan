#pragma once

#include "LineScanDefinition.h"
#include "LineScanTrajectory.h"

#include "camera/core/model/CameraInstance.h"

#include <memory>
#include <optional>
#include <vector>

namespace xjw::camera_models::linescan
{

    struct LineRateSegment
    {
        double startLine = 0.5;
        double startTimeSeconds = 0.0;
        double secondsPerLine = 0.0;
    };

    struct LineTiming
    {
        double lineZero = 0.5;
        double startTimeSeconds = 0.0;
        double secondsPerLine = 0.0;
        xjw::coordinate_system::TimeScale timeScale = xjw::coordinate_system::TimeScale::Tdb;
        std::vector<LineRateSegment> segments;
    };

    class LineScanInstance final : public camera_core::CameraInstance
    {
    public:
        static LineScanInstance create(camera_core::CameraInstanceId instanceId,
                                       camera_core::ImageId imageId,
                                       std::shared_ptr<const LineScanDefinition> definition,
                                       camera_core::ImageSize imageSize,
                                       LineScanTrajectory trajectory,
                                       LineTiming timing,
                                       std::optional<xjw::coordinate_system::TimeReference> captureTime = std::nullopt);
        static std::unique_ptr<LineScanInstance>
        createUnique(camera_core::CameraInstanceId instanceId,
                     camera_core::ImageId imageId,
                     std::shared_ptr<const LineScanDefinition> definition,
                     camera_core::ImageSize imageSize,
                     LineScanTrajectory trajectory,
                     LineTiming timing,
                     std::optional<xjw::coordinate_system::TimeReference> captureTime = std::nullopt);

        const LineScanDefinition& lineScanDefinition() const noexcept;
        const LineScanTrajectory& trajectory() const noexcept;
        const LineTiming& lineTiming() const noexcept;

        bool timeForLine(double line, double* seconds) const;
        bool lineForTime(double seconds, double* line) const;

    private:
        LineScanInstance(camera_core::CameraInstanceId instanceId,
                         camera_core::ImageId imageId,
                         std::shared_ptr<const LineScanDefinition> definition,
                         camera_core::ImageSize imageSize,
                         LineScanTrajectory trajectory,
                         LineTiming timing,
                         std::optional<xjw::coordinate_system::TimeReference> captureTime);

        std::shared_ptr<const LineScanDefinition> _lineScanDefinition;
        LineScanTrajectory _trajectory;
        LineTiming _timing;
    };

} // namespace xjw::camera_models::linescan
