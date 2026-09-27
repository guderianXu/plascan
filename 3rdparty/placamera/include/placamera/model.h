#pragma once

#include "placamera/capabilities.h"
#include "placamera/optimization.h"
#include "placamera/result.h"
#include "placamera/types.h"

#include <memory>
#include <optional>
#include <string_view>

namespace placamera
{

    struct EvaluationOptions
    {
        double desiredPrecisionPixels = 1.0e-7;
        int maximumIterations = 30;
        bool requireInsideImage = false;
    };

    struct Projection
    {
        ImageCoordinate image;
        std::optional<double> positiveDepth;
        std::optional<TimeReference> acquisitionTime;
    };

    struct ImagingLocus
    {
        GroundCoordinate origin;
        Vector3 direction{{0.0, 0.0, 1.0}};
        std::optional<TimeReference> acquisitionTime;
    };

    class CameraDefinition
    {
    public:
        virtual ~CameraDefinition() = default;

        virtual const CameraDefinitionId& definitionId() const noexcept = 0;
        virtual std::string_view modelType() const noexcept = 0;
        virtual int parameterSchemaVersion() const noexcept = 0;
        virtual const FrameId& groundFrame() const noexcept = 0;
        virtual CapabilitySet capabilities() const noexcept = 0;
    };

    /** Canonical shared ownership for immutable camera models. */
    template <typename T> using CameraModelPtr = std::shared_ptr<const T>;

    class RasterModel;
    using RasterModelPtr = CameraModelPtr<RasterModel>;

    class RasterModel
    {
    public:
        virtual ~RasterModel() = default;

        virtual const CameraInstanceId& instanceId() const noexcept = 0;
        virtual const CameraDefinition& definition() const noexcept = 0;
        virtual const CameraDefinitionId& definitionId() const noexcept = 0;
        virtual const ImageId& imageId() const noexcept = 0;
        virtual std::string_view modelType() const noexcept = 0;
        virtual int parameterSchemaVersion() const noexcept = 0;
        virtual const FrameId& groundFrame() const noexcept = 0;
        virtual const ImageSize& imageSize() const noexcept = 0;
        virtual const std::optional<TimeReference>& captureTime() const noexcept = 0;
        virtual CapabilitySet capabilities() const noexcept = 0;

        virtual EvaluationResult<Projection> groundToImage(const GroundCoordinate& ground,
                                                           const EvaluationOptions& options = {}) const = 0;

        virtual EvaluationResult<ImagingLocus> imageToImagingLocus(const ImageCoordinate& image,
                                                                   const EvaluationOptions& options = {}) const = 0;

        virtual OptimizationLayout optimizationLayout() const;
        virtual Result<RasterModelPtr> withOptimizationUpdate(const OptimizationUpdate& update) const;
    };

} // namespace placamera
