#pragma once

#include <optional>

namespace placamera
{

    /**
     * Exact decomposition of an absolute principal point.
     *
     * imageCenter and the calibrated offsets are retained independently so an
     * inverse projection never has to recover the centre by subtracting two
     * rounded floating-point values. cx/cy in MetashapeCalibration remain the
     * active absolute principal point.
     */
    struct PrincipalPointDecomposition
    {
        double imageCenterX = 0.0;
        double imageCenterY = 0.0;
        double cxOffset = 0.0;
        double cyOffset = 0.0;

        bool operator==(const PrincipalPointDecomposition&) const = default;
    };

    /**
     * Lossless Metashape-compatible pixel calibration.
     *
     * Pixels are formed as `(f + b1) * x + b2 * y + cx` and
     * `f * y + cy`. Radial distortion uses k1..k4. p1/p2 form the
     * Metashape tangential field and p3/p4 scale the complete field by
     * `1 + p3*r^2 + p4*r^4`.
     */
    struct MetashapeCalibration
    {
        double f = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        double b1 = 0.0;
        double b2 = 0.0;
        double k1 = 0.0;
        double k2 = 0.0;
        double k3 = 0.0;
        double k4 = 0.0;
        double p1 = 0.0;
        double p2 = 0.0;
        double p3 = 0.0;
        double p4 = 0.0;
        std::optional<PrincipalPointDecomposition> principalPointDecomposition;

        bool operator==(const MetashapeCalibration&) const = default;
    };

    /** Original frame-specific spelling retained for source compatibility. */
    using FrameCalibration = MetashapeCalibration;

} // namespace placamera
