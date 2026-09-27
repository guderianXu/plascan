#pragma once

#include "placamera/frame_camera.h"

#include <array>
#include <istream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace placamera
{

    struct CameraImportCalibrationTerm
    {
        std::string name;
        double value = 0.0;
    };

    /** Source calibration details outside the Brown-Conrady target model. */
    struct CameraImportCompatibility
    {
        std::vector<CameraImportCalibrationTerm> sourceOnlyTerms;
        std::optional<std::string> unsupportedReason;

        bool isExactlyRepresentable(double tolerance = 1.0e-12) const noexcept;
    };

    /** Typed pixel calibration retained by dataset importers before model binding. */
    struct ImportedPixelCalibration
    {
        /** Row-major 3x3 pixel calibration matrix. */
        std::array<double, 9> intrinsicMatrix{};
        BrownConradyDistortion distortion;
        FrameProjectionModel projectionModel = FrameProjectionModel::Perspective;
        /**
         * Authoritative lossless calibration when the source supplies
         * Metashape semantics. intrinsicMatrix and distortion remain a
         * normalized inspection view; model binding rejects disagreement
         * between that view and this value.
         */
        std::optional<MetashapeCalibration> metashapeCalibration;
    };

    /** Pixel calibration and camera-to-world pose from an external dataset. */
    struct ImportedCamera
    {
        std::string imageName;
        ImportedPixelCalibration calibration;
        std::array<double, 9> cameraToWorldRotation{};
        std::array<double, 3> center{};
        CameraAcquisitionState acquisition;
        CameraImportCompatibility compatibility;
    };

    using DatasetCalibrationTerm = CameraImportCalibrationTerm;
    using DatasetCalibrationCompatibility = CameraImportCompatibility;
    using DatasetPixelCalibration = ImportedPixelCalibration;
    using DatasetFrameCamera = ImportedCamera;

    /** Parse camera geometry from a Middlebury *_par.txt stream. */
    Result<std::vector<ImportedCamera>> readMiddleburyPar(std::istream& input);

    /** Parse camera geometry from an EPFL/Strecha .camera stream. The caller supplies the image identity. */
    Result<ImportedCamera> readEpflCamera(std::istream& input);

    /** Parse adjusted sensor calibration and camera transforms from a Metashape chunk document. */
    Result<std::vector<ImportedCamera>> parseMetashapeDocument(std::string_view xml);

    /** Build typed central-camera geometry when the imported calibration is exactly representable. */
    Result<CentralCameraGeometry> makeCentralCameraGeometry(const ImportedCamera& camera,
                                                            CameraDefinitionId definitionId,
                                                            FrameId worldFrame,
                                                            double tolerance = 1.0e-12);

    /** Legacy dataset-oriented spelling retained for source compatibility. */
    Result<CentralCameraGeometry> makeDatasetCentralCamera(const DatasetFrameCamera& camera,
                                                           CameraDefinitionId definitionId,
                                                           FrameId worldFrame,
                                                           double tolerance = 1.0e-12);

    /** Legacy perspective-oriented spelling retained for source compatibility. */
    Result<FramePinholeGeometry> makeDatasetFramePinhole(const DatasetFrameCamera& camera,
                                                         CameraDefinitionId definitionId,
                                                         FrameId worldFrame,
                                                         double tolerance = 1.0e-12);

} // namespace placamera
