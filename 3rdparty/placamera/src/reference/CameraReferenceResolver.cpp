#include "placamera/reference/CameraReferenceResolver.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>

namespace placamera::reference
{
    namespace
    {

        constexpr const char* kNormalizationSchema = "camera-reference-normalization/v2";
        constexpr const char* kNormalizationPolicy = "explicit-frame-graph-partial-reference-v2";
        constexpr const char* kCovariancePolicy = "sim3-position-rotation-tangent-v1";

        using placoordinate::CoordinateTransformError;

        bool finiteArray(const std::array<double, 3>& values)
        {
            for (double value : values)
            {
                if (!std::isfinite(value))
                {
                    return false;
                }
            }
            return true;
        }

        placamera::RotationMatrix transposeRotation(const placamera::RotationMatrix& rotation)
        {
            return {rotation[0],
                    rotation[3],
                    rotation[6],
                    rotation[1],
                    rotation[4],
                    rotation[7],
                    rotation[2],
                    rotation[5],
                    rotation[8]};
        }

        void appendHashValue(std::ostringstream& stream, double value)
        {
            stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value << ';';
        }

        std::string hashCanonicalString(const std::string& canonical)
        {
            std::uint64_t hash = 14695981039346656037ULL;
            for (const char value : canonical)
            {
                hash ^= static_cast<unsigned char>(value);
                hash *= 1099511628211ULL;
            }

            std::ostringstream result;
            result << std::hex << std::setw(16) << std::setfill('0') << hash;
            return result.str();
        }

        ResolvedCameraReference unresolved(const placoordinate::CoordinateFrameId& targetFrame, std::string reason)
        {
            ResolvedCameraReference result;
            result.targetFrame = targetFrame;
            result.reason = std::move(reason);
            return result;
        }

        bool covarianceTransform(const placoordinate::CoordinateTransformService& transforms,
                                 const placoordinate::CoordinateFrameId& source,
                                 const placoordinate::CoordinateFrameId& target,
                                 double* scale,
                                 placamera::RotationMatrix* targetFromSource)
        {
            if (!scale || !targetFromSource)
            {
                return false;
            }
            const std::array<std::array<double, 3>, 3> basis{{{{1.0, 0.0, 0.0}},
                                                               {{0.0, 1.0, 0.0}},
                                                               {{0.0, 0.0, 1.0}}}};
            std::array<std::array<double, 3>, 3> transformed{};
            std::array<double, 3> norms{};
            for (std::size_t column = 0; column < basis.size(); ++column)
            {
                transformed[column] = transforms.transformVector(source, target, basis[column]);
                norms[column] = std::hypot(transformed[column][0],
                                           std::hypot(transformed[column][1], transformed[column][2]));
                if (!std::isfinite(norms[column]) || !(norms[column] > 0.0))
                {
                    return false;
                }
            }
            *scale = (norms[0] + norms[1] + norms[2]) / 3.0;
            const double scale_tolerance = 1.0e-10 * std::max(1.0, *scale);
            if (std::abs(norms[0] - *scale) > scale_tolerance ||
                std::abs(norms[1] - *scale) > scale_tolerance ||
                std::abs(norms[2] - *scale) > scale_tolerance)
            {
                return false;
            }
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    (*targetFromSource)[row * 3 + column] = transformed[column][row] / *scale;
                }
            }
            return true;
        }

    } // namespace

    ResolvedCameraReference CameraReferenceResolver::resolve(const CameraReferenceObservation& observation,
                                                             const placoordinate::CoordinateFrameId& targetFrame,
                                                             const CameraReferenceResolveOptions& options) const
    {
        if (!_transforms)
        {
            return unresolved(targetFrame, "coordinate transform service is not configured");
        }
        if (!observation.enabled)
        {
            return unresolved(targetFrame, "reference observation is disabled");
        }
        if (!observation.position && !observation.orientation)
        {
            return unresolved(targetFrame, "reference observation has neither position nor orientation");
        }
        if (observation.position && !finiteArray(*observation.position))
        {
            return unresolved(targetFrame, "reference position contains non-finite values");
        }

        if (observation.orientation)
        {
            if (options.orientationConvention.empty())
            {
                return unresolved(targetFrame, "orientation convention is missing");
            }
            if (options.orientationConvention != "camera_to_world" &&
                options.orientationConvention != "world_to_camera")
            {
                return unresolved(targetFrame, "unsupported orientation convention: " + options.orientationConvention);
            }
        }
        if (observation.leverArm && observation.leverArm->direction == LeverArmDirection::Unknown)
        {
            return unresolved(targetFrame, "lever arm direction is missing");
        }
        if (observation.leverArm && !observation.leverArm->vectorFrame)
        {
            return unresolved(targetFrame, "lever arm vector frame is missing");
        }
        if (observation.leverArm && !finiteArray(observation.leverArm->vector))
        {
            return unresolved(targetFrame, "lever arm contains non-finite values");
        }
        if (observation.leverArm && !observation.position)
        {
            return unresolved(targetFrame, "a lever arm requires a reference position");
        }
        if (observation.covariance &&
            ((observation.covariance->hasPosition() && !observation.position) ||
             (observation.covariance->hasRotation() && !observation.orientation)))
        {
            return unresolved(targetFrame, "reference covariance contains a component with no reference value");
        }

        try
        {
            std::optional<std::array<double, 3>> transformedPosition;
            if (observation.position)
            {
                transformedPosition =
                    _transforms->transformPoint(observation.frame, targetFrame, *observation.position);
            }

            std::optional<placamera::RotationMatrix> sourceOrientation;
            if (observation.orientation)
            {
                sourceOrientation = *observation.orientation;
                if (options.orientationConvention == "world_to_camera")
                {
                    sourceOrientation = transposeRotation(*sourceOrientation);
                }
            }

            std::optional<std::array<double, 3>> cameraCenter = transformedPosition;
            bool leverArmApplied = false;
            if (observation.leverArm)
            {
                const std::array<double, 3> leverArmInSourceFrame = _transforms->transformVector(
                    *observation.leverArm->vectorFrame, observation.frame, observation.leverArm->vector);
                const std::array<double, 3> leverArm =
                    _transforms->transformVector(observation.frame, targetFrame, leverArmInSourceFrame);
                const double sign = observation.leverArm->direction == LeverArmDirection::SensorToCamera ? 1.0 : -1.0;
                for (std::size_t index = 0; index < cameraCenter->size(); ++index)
                {
                    (*cameraCenter)[index] += sign * leverArm[index];
                }
                leverArmApplied = true;
            }

            std::optional<placamera::RotationMatrix> targetOrientation;
            if (sourceOrientation)
            {
                targetOrientation =
                    _transforms->transformRotation(observation.frame, targetFrame, *sourceOrientation);
            }

            ResolvedCameraReference result;
            result.status = ReferenceResolutionStatus::Resolved;
            result.position = cameraCenter;
            result.orientation = targetOrientation;
            if (cameraCenter && targetOrientation)
            {
                result.pose = placamera::Pose::create(targetFrame, *cameraCenter, *targetOrientation);
            }
            if (observation.covariance)
            {
                double scale = 1.0;
                placamera::RotationMatrix target_from_source{};
                if (!covarianceTransform(*_transforms,
                                         observation.frame,
                                         targetFrame,
                                         &scale,
                                         &target_from_source))
                {
                    return unresolved(targetFrame,
                                      "reference frame transform is not a uniform similarity for covariance propagation");
                }
                const auto propagated =
                    propagatePoseCovarianceSimilarity(*observation.covariance, scale, target_from_source);
                if (!propagated)
                {
                    return unresolved(targetFrame, propagated.message());
                }
                result.covariance = propagated.value();
            }
            result.targetFrame = targetFrame;
            result.leverArmApplied = leverArmApplied;

            const std::string poseChainHash = _transforms->transformChainHash(observation.frame, targetFrame);
            const std::string leverChainHash =
                observation.leverArm
                    ? _transforms->transformChainHash(*observation.leverArm->vectorFrame, observation.frame)
                    : std::string();

            std::ostringstream provenance;
            provenance << "schema=" << kNormalizationSchema << ";policy=" << kNormalizationPolicy
                       << ";covariance-policy=" << kCovariancePolicy << ";source-frame=" << observation.frame.value()
                       << ";target-frame=" << targetFrame.value() << ";orientation=" << options.orientationConvention
                       << ";pose-chain=" << poseChainHash
                       << ";lever-chain=" << (leverChainHash.empty() ? std::string("none") : leverChainHash);
            result.transformProvenanceHash = hashCanonicalString(provenance.str());

            std::ostringstream canonical;
            canonical << "image=" << observation.image.value() << ";source=" << observation.source.value()
                      << ";frame=" << observation.frame.value() << ";target=" << targetFrame.value()
                      << ";orientation=" << options.orientationConvention << ";lever="
                      << static_cast<int>(observation.leverArm ? observation.leverArm->direction
                                                               : LeverArmDirection::Unknown)
                      << ";vector-frame="
                      << (observation.leverArm && observation.leverArm->vectorFrame
                              ? observation.leverArm->vectorFrame->value()
                              : std::string("none"))
                      << ";chain=" << poseChainHash
                      << ";lever-chain=" << (leverChainHash.empty() ? std::string("none") : leverChainHash) << ';';
            if (observation.position)
            {
                for (double value : *observation.position)
                {
                    appendHashValue(canonical, value);
                }
            }
            if (observation.orientation)
            {
                for (double value : *observation.orientation)
                {
                    appendHashValue(canonical, value);
                }
            }
            if (observation.leverArm)
            {
                for (double value : observation.leverArm->vector)
                {
                    appendHashValue(canonical, value);
                }
            }
            if (observation.covariance)
            {
                canonical << "covariance-layout=" << static_cast<int>(observation.covariance->layout()) << ';';
                for (double value : observation.covariance->values())
                {
                    appendHashValue(canonical, value);
                }
            }
            result.transformHash = hashCanonicalString(canonical.str());
            return result;
        }
        catch (const CoordinateTransformError& error)
        {
            return unresolved(targetFrame, error.what());
        }
        catch (const std::exception& error)
        {
            return unresolved(targetFrame, std::string("reference resolution failed: ") + error.what());
        }
    }

} // namespace placamera::reference
