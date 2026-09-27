#pragma once

#include "placamera/model.h"

#include <array>
#include <memory>
#include <optional>
#include <string_view>
#include <variant>

namespace placamera
{

    struct ReferenceEllipsoid
    {
        double semiMajorAxisMeters = 6378137.0;
        double inverseFlattening = 298.257223563;

        static ReferenceEllipsoid wgs84() noexcept;
    };

    struct GeodeticCoordinate
    {
        double longitudeDegrees = 0.0;
        double latitudeDegrees = 0.0;
        double heightMeters = 0.0;
    };

    using RpcCoefficients = std::array<double, 20>;

    struct RpcParameters
    {
        double lineOffset = 0.0;
        double sampleOffset = 0.0;
        double latitudeOffset = 0.0;
        double longitudeOffset = 0.0;
        double heightOffset = 0.0;
        double lineScale = 0.0;
        double sampleScale = 0.0;
        double latitudeScale = 0.0;
        double longitudeScale = 0.0;
        double heightScale = 0.0;
        RpcCoefficients lineNumerator{};
        RpcCoefficients lineDenominator{};
        RpcCoefficients sampleNumerator{};
        RpcCoefficients sampleDenominator{};
        std::optional<double> errorBiasMeters;
        std::optional<double> errorRandomMeters;
    };

    struct RpcImageCorrection
    {
        double sampleOffsetPixels = 0.0;
        double sampleSamplePixels = 0.0;
        double sampleLinePixels = 0.0;
        double lineOffsetPixels = 0.0;
        double lineSamplePixels = 0.0;
        double lineLinePixels = 0.0;
    };

    struct RpcGroundCorrection
    {
        double sampleOffsetPixels = 0.0;
        double lineOffsetPixels = 0.0;
        double sampleLongitudePixelsPerDegree = 0.0;
        double sampleLatitudePixelsPerDegree = 0.0;
        double sampleHeightPixelsPerMeter = 0.0;
        double lineLongitudePixelsPerDegree = 0.0;
        double lineLatitudePixelsPerDegree = 0.0;
        double lineHeightPixelsPerMeter = 0.0;
    };

    enum class RpcCorrectionDomain
    {
        NormalizedImage,
        GroundCoordinates,
    };

    /**
     * A domain-tagged RPC correction. The two affine forms use different
     * independent variables and are deliberately not implicitly convertible.
     */
    class RpcCorrection
    {
    public:
        static RpcCorrection normalizedImage(RpcImageCorrection correction = {});
        static RpcCorrection groundCoordinates(RpcGroundCorrection correction = {});

        RpcCorrectionDomain domain() const noexcept;
        const RpcImageCorrection* normalizedImageValue() const noexcept;
        const RpcGroundCorrection* groundCoordinateValue() const noexcept;

    private:
        explicit RpcCorrection(RpcImageCorrection correction);
        explicit RpcCorrection(RpcGroundCorrection correction);

        std::variant<RpcImageCorrection, RpcGroundCorrection> _value;
    };

    EvaluationResult<Vector3> geodeticToCartesian(const GeodeticCoordinate& geodetic,
                                                  const ReferenceEllipsoid& ellipsoid);

    EvaluationResult<GeodeticCoordinate> cartesianToGeodetic(const Vector3& cartesian,
                                                             const ReferenceEllipsoid& ellipsoid);

    class RpcDefinition final : public CameraDefinition
    {
    public:
        static constexpr int ParameterSchemaVersion = 1;

        static std::shared_ptr<const RpcDefinition> create(CameraDefinitionId definitionId,
                                                           FrameId groundFrame,
                                                           RpcParameters parameters,
                                                           ReferenceEllipsoid ellipsoid = ReferenceEllipsoid::wgs84());

        const CameraDefinitionId& definitionId() const noexcept override;
        std::string_view modelType() const noexcept override;
        int parameterSchemaVersion() const noexcept override;
        const FrameId& groundFrame() const noexcept override;
        const RpcParameters& parameters() const noexcept;
        const ReferenceEllipsoid& ellipsoid() const noexcept;
        CapabilitySet capabilities() const noexcept override;

    private:
        RpcDefinition(CameraDefinitionId definitionId,
                      FrameId groundFrame,
                      RpcParameters parameters,
                      ReferenceEllipsoid ellipsoid);

        static void validate(const RpcParameters& parameters, const ReferenceEllipsoid& ellipsoid);

        CameraDefinitionId _definitionId;
        FrameId _groundFrame;
        RpcParameters _parameters;
        ReferenceEllipsoid _ellipsoid;
    };

    class RpcModel final : public RasterModel
    {
    public:
        static constexpr int InstanceSchemaVersion = 2;

        static RpcModel create(CameraInstanceId instanceId,
                               ImageId imageId,
                               std::shared_ptr<const RpcDefinition> definition,
                               ImageSize imageSize,
                               RpcImageCorrection correction = {},
                               std::optional<TimeReference> captureTime = std::nullopt);

        static RpcModel createWithCorrection(CameraInstanceId instanceId,
                                             ImageId imageId,
                                             std::shared_ptr<const RpcDefinition> definition,
                                             ImageSize imageSize,
                                             RpcCorrection correction,
                                             std::optional<TimeReference> captureTime = std::nullopt);

        const CameraInstanceId& instanceId() const noexcept override;
        const CameraDefinition& definition() const noexcept override;
        const CameraDefinitionId& definitionId() const noexcept override;
        const ImageId& imageId() const noexcept override;
        std::string_view modelType() const noexcept override;
        int parameterSchemaVersion() const noexcept override;
        const FrameId& groundFrame() const noexcept override;
        const ImageSize& imageSize() const noexcept override;
        const std::optional<TimeReference>& captureTime() const noexcept override;
        CapabilitySet capabilities() const noexcept override;

        const RpcDefinition& rpcDefinition() const noexcept;
        RpcCorrectionDomain correctionDomain() const noexcept;
        const RpcCorrection& correction() const noexcept;
        const RpcImageCorrection* normalizedImageCorrection() const noexcept;
        const RpcGroundCorrection* groundCorrection() const noexcept;

        /** Legacy accessor retained for source compatibility; rejects a ground-domain model. */
        const RpcImageCorrection& imageCorrection() const;

        EvaluationResult<Projection> groundToImage(const GroundCoordinate& ground,
                                                   const EvaluationOptions& options = {}) const override;

        EvaluationResult<Projection> groundToImageGeodetic(const GeodeticCoordinate& ground,
                                                           const EvaluationOptions& options = {}) const;

        EvaluationResult<GeodeticCoordinate> imageToGroundAtHeight(const ImageCoordinate& image,
                                                                   double ellipsoidalHeightMeters,
                                                                   const EvaluationOptions& options = {}) const;

        EvaluationResult<ImagingLocus> imageToImagingLocus(const ImageCoordinate& image,
                                                           const EvaluationOptions& options = {}) const override;

        OptimizationLayout optimizationLayout() const override;
        Result<RasterModelPtr> withOptimizationUpdate(const OptimizationUpdate& update) const override;

        RpcModel withImageCorrection(CameraInstanceId instanceId, RpcImageCorrection correction) const;
        RpcModel withGroundCorrection(CameraInstanceId instanceId, RpcGroundCorrection correction) const;
        RpcModel withCorrection(CameraInstanceId instanceId, RpcCorrection correction) const;

    private:
        RpcModel(CameraInstanceId instanceId,
                 ImageId imageId,
                 std::shared_ptr<const RpcDefinition> definition,
                 ImageSize imageSize,
                 RpcCorrection correction,
                 std::optional<TimeReference> captureTime);

        CameraInstanceId _instanceId;
        ImageId _imageId;
        std::shared_ptr<const RpcDefinition> _definition;
        ImageSize _imageSize;
        RpcCorrection _correction;
        std::optional<TimeReference> _captureTime;
    };

} // namespace placamera
