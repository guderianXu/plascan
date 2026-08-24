#pragma once

#include "CameraModel.h"

#include <array>
#include <optional>
#include <string>

namespace xjw
{

    /**
     * @brief RPC00B rational-polynomial sensor model.
     *
     * RPC coefficients operate on longitude/latitude in degrees and WGS84
     * ellipsoidal height in metres. The common CameraModel interface receives
     * WGS84 ECEF coordinates in metres so its groundMeters contract remains true.
     */
    class RpcCameraModel final : public CameraModel
    {
    public:
        using Coefficients = std::array<double, 20>;
        using GeodeticCoordinate = std::array<double, 3>; ///< longitude deg, latitude deg, ellipsoidal height m
        using EcefCoordinate = std::array<double, 3>;

        struct Parameters
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
            Coefficients lineNumerator{};
            Coefficients lineDenominator{};
            Coefficients sampleNumerator{};
            Coefficients sampleDenominator{};
            std::optional<double> errorBiasMeters;
            std::optional<double> errorRandomMeters;
        };

        struct InverseOptions
        {
            double pixelTolerance = 1.0e-7;
            int maximumIterations = 30;
        };

        /**
         * @brief Additive affine correction in normalized RPC image coordinates.
         *
         * Each output is expressed in pixels. normalizedSample and
         * normalizedLine are the uncorrected RPC image coordinates relative to
         * SAMP_OFF/LINE_OFF and divided by SAMP_SCALE/LINE_SCALE.
         */
        struct ImageCorrection
        {
            double sampleOffsetPixels = 0.0;
            double sampleSamplePixels = 0.0;
            double sampleLinePixels = 0.0;
            double lineOffsetPixels = 0.0;
            double lineSamplePixels = 0.0;
            double lineLinePixels = 0.0;
        };

        RpcCameraModel() = default;

        CameraModelType modelType() const noexcept override;
        bool isValid() const noexcept override
        {
            return _isValid;
        }
        std::optional<CameraImageSize> imageSize() const noexcept override
        {
            return _imageSize;
        }
        std::string_view worldFrameName() const noexcept override;
        bool rayForPixel(const CameraImageCoordinate& pixel, CameraImagingRay* ray) const override;
        bool groundToImage(const EcefCoordinate& groundMeters, CameraGroundProjection* projection) const override;

        bool setParameters(const Parameters& parameters, std::string* errorMessage = nullptr);
        const Parameters& parameters() const noexcept
        {
            return _parameters;
        }
        void setImageSize(std::optional<CameraImageSize> imageSize);
        bool setImageCorrection(const ImageCorrection& correction, std::string* errorMessage = nullptr);
        const ImageCorrection& imageCorrection() const noexcept
        {
            return _imageCorrection;
        }

        bool groundToImageGeodetic(const GeodeticCoordinate& ground, CameraImageCoordinate* image) const;
        bool groundToImageGeodeticUncorrected(const GeodeticCoordinate& ground, CameraImageCoordinate* image) const;
        bool imageToGroundAtHeight(const CameraImageCoordinate& image,
                                   double ellipsoidalHeightMeters,
                                   GeodeticCoordinate* ground) const;
        bool imageToGroundAtHeight(const CameraImageCoordinate& image,
                                   double ellipsoidalHeightMeters,
                                   GeodeticCoordinate* ground,
                                   const InverseOptions& options) const;
        bool imageToEcefAtHeight(const CameraImageCoordinate& image,
                                 double ellipsoidalHeightMeters,
                                 EcefCoordinate* ground) const;

        static bool geodeticToEcef(const GeodeticCoordinate& geodetic, EcefCoordinate* ecef);
        static bool ecefToGeodetic(const EcefCoordinate& ecef, GeodeticCoordinate* geodetic);

    private:
        bool evaluateNormalizedUncorrected(double normalizedLongitude,
                                           double normalizedLatitude,
                                           double normalizedHeight,
                                           CameraImageCoordinate* image) const;
        bool applyImageCorrection(const CameraImageCoordinate& uncorrected, CameraImageCoordinate* corrected) const;
        bool evaluateNormalized(double normalizedLongitude,
                                double normalizedLatitude,
                                double normalizedHeight,
                                CameraImageCoordinate* image) const;

        Parameters _parameters;
        ImageCorrection _imageCorrection;
        std::optional<CameraImageSize> _imageSize;
        bool _isValid = false;
    };

} // namespace xjw
