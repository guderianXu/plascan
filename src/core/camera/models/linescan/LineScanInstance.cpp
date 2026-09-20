#include "LineScanInstance.h"

#include "camera/core/types/CameraErrors.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace xjw::camera_models::linescan
{

    LineScanInstance LineScanInstance::create(camera_core::CameraInstanceId instanceId,
                                              camera_core::ImageId imageId,
                                              std::shared_ptr<const LineScanDefinition> definition,
                                              camera_core::ImageSize imageSize,
                                              LineScanTrajectory trajectory,
                                              LineTiming timing,
                                              std::optional<xjw::coordinate_system::TimeReference> captureTime)
    {
        if (!definition)
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidFrame,
                                                     "line-scan instance requires a definition");
        }
        bool timingValid = std::isfinite(timing.lineZero) && std::isfinite(timing.startTimeSeconds);
        if (timing.segments.empty())
        {
            timingValid = timingValid && std::isfinite(timing.secondsPerLine) && timing.secondsPerLine > 0.0;
        }
        for (std::size_t index = 0; index < timing.segments.size(); ++index)
        {
            const LineRateSegment& segment = timing.segments[index];
            timingValid = timingValid && std::isfinite(segment.startLine) &&
                          std::isfinite(segment.startTimeSeconds) && std::isfinite(segment.secondsPerLine) &&
                          segment.secondsPerLine > 0.0 &&
                          (index == 0 || (segment.startLine > timing.segments[index - 1].startLine &&
                                          segment.startTimeSeconds > timing.segments[index - 1].startTimeSeconds));
        }
        if (!timingValid)
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidTime,
                                                     "line-scan timing must be finite with a positive line period");
        }
        if (trajectory.timeScale() != timing.timeScale)
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidTime,
                                                     "line-scan timing and trajectory must use the same time scale");
        }
        return LineScanInstance(std::move(instanceId),
                                std::move(imageId),
                                std::move(definition),
                                imageSize,
                                std::move(trajectory),
                                timing,
                                captureTime);
    }

    std::unique_ptr<LineScanInstance>
    LineScanInstance::createUnique(camera_core::CameraInstanceId instanceId,
                                   camera_core::ImageId imageId,
                                   std::shared_ptr<const LineScanDefinition> definition,
                                   camera_core::ImageSize imageSize,
                                   LineScanTrajectory trajectory,
                                   LineTiming timing,
                                   std::optional<xjw::coordinate_system::TimeReference> captureTime)
    {
        return std::unique_ptr<LineScanInstance>(new LineScanInstance(create(std::move(instanceId),
                                                                             std::move(imageId),
                                                                             std::move(definition),
                                                                             imageSize,
                                                                             std::move(trajectory),
                                                                             timing,
                                                                             captureTime)));
    }

    const LineScanDefinition& LineScanInstance::lineScanDefinition() const noexcept
    {
        return *_lineScanDefinition;
    }

    const LineScanTrajectory& LineScanInstance::trajectory() const noexcept
    {
        return _trajectory;
    }

    const LineTiming& LineScanInstance::lineTiming() const noexcept
    {
        return _timing;
    }

    bool LineScanInstance::timeForLine(double line, double* seconds) const
    {
        if (!seconds || !std::isfinite(line))
        {
            return false;
        }
        const double minimumLine = _lineScanDefinition->pixelConvention() == PixelConvention::PixelCenter ? 0.5 : 0.0;
        const double maximumLine = minimumLine + static_cast<double>(imageSize().lines - 1);
        if (line < minimumLine || line > maximumLine)
        {
            return false;
        }
        if (_timing.segments.empty())
        {
            *seconds = _timing.startTimeSeconds + (line - _timing.lineZero) * _timing.secondsPerLine;
        }
        else
        {
            auto segment = std::upper_bound(
                _timing.segments.begin(),
                _timing.segments.end(),
                line,
                [](double candidate, const LineRateSegment& value) { return candidate < value.startLine; });
            if (segment != _timing.segments.begin())
            {
                --segment;
            }
            *seconds = segment->startTimeSeconds + (line - segment->startLine) * segment->secondsPerLine;
        }
        return std::isfinite(*seconds);
    }

    bool LineScanInstance::lineForTime(double seconds, double* line) const
    {
        if (!line || !std::isfinite(seconds))
        {
            return false;
        }
        if (_timing.segments.empty())
        {
            *line = _timing.lineZero + (seconds - _timing.startTimeSeconds) / _timing.secondsPerLine;
        }
        else
        {
            auto segment = _timing.segments.begin();
            for (auto candidate = _timing.segments.begin(); candidate != _timing.segments.end(); ++candidate)
            {
                if (seconds >= candidate->startTimeSeconds)
                {
                    segment = candidate;
                }
            }
            *line = segment->startLine + (seconds - segment->startTimeSeconds) / segment->secondsPerLine;
        }
        const double minimumLine = _lineScanDefinition->pixelConvention() == PixelConvention::PixelCenter ? 0.5 : 0.0;
        const double maximumLine = minimumLine + static_cast<double>(imageSize().lines - 1);
        return *line >= minimumLine && *line <= maximumLine;
    }

    LineScanInstance::LineScanInstance(camera_core::CameraInstanceId instanceId,
                                       camera_core::ImageId imageId,
                                       std::shared_ptr<const LineScanDefinition> definition,
                                       camera_core::ImageSize imageSize,
                                       LineScanTrajectory trajectory,
                                       LineTiming timing,
                                       std::optional<xjw::coordinate_system::TimeReference> captureTime)
        : camera_core::CameraInstance(std::move(instanceId),
                                      std::move(imageId),
                                      definition,
                                      imageSize,
                                      captureTime,
                                      camera_core::CapabilitySet{camera_core::CapabilityKind::Projection,
                                                                 camera_core::CapabilityKind::Ray,
                                                                 camera_core::CapabilityKind::Trajectory,
                                                                 camera_core::CapabilityKind::Optimization}),
          _lineScanDefinition(std::move(definition)), _trajectory(std::move(trajectory)), _timing(timing)
    {
    }

} // namespace xjw::camera_models::linescan
