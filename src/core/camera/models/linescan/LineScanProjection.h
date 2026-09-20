#pragma once

#include "LineScanInstance.h"
#include "LineScanOptimization.h"

#include <array>

namespace xjw::camera_models::linescan
{

    struct LineScanRay
    {
        std::array<double, 3> origin{{0.0, 0.0, 0.0}};
        std::array<double, 3> direction{{0.0, 0.0, 1.0}};
        double timeSeconds = 0.0;
    };

    struct LineScanProjectionResult
    {
        double sample = 0.0;
        double line = 0.0;
        double lineResidualPixels = 0.0;
        double positiveDepth = 0.0;
        double timeSeconds = 0.0;
        double undistortedFocalXMillimeters = 0.0;
        double undistortedFocalYMillimeters = 0.0;
    };

    struct LineScanProjectionOptions
    {
        double desiredLinePrecisionPixels = 1.0e-6;
        int maximumIterations = 20;
        bool requireInsideImage = false;
    };

    class LineScanProjection
    {
    public:
        static bool ray(const LineScanInstance& instance, double sample, double line, LineScanRay* ray);
        static bool ray(const LineScanInstance& instance,
                        double sample,
                        double line,
                        const LineScanTrajectoryBias& bias,
                        LineScanRay* ray);

        static bool projectAtLine(const LineScanInstance& instance,
                                  const std::array<double, 3>& world,
                                  double line,
                                  LineScanProjectionResult* result);
        static bool projectAtLine(const LineScanInstance& instance,
                                  const std::array<double, 3>& world,
                                  double line,
                                  const LineScanTrajectoryBias& bias,
                                  LineScanProjectionResult* result);

        static bool project(const LineScanInstance& instance,
                            const std::array<double, 3>& world,
                            LineScanProjectionResult* result,
                            int maximumIterations = 20);
        static bool project(const LineScanInstance& instance,
                            const std::array<double, 3>& world,
                            const LineScanProjectionOptions& options,
                            const LineScanTrajectoryBias& bias,
                            LineScanProjectionResult* result);
    };

} // namespace xjw::camera_models::linescan
