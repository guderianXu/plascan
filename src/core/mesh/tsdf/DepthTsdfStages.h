#pragma once
#include "DepthTsdfInternals.h"

namespace xjw::mesh::tsdf_detail
{
    struct TsdfIntegrationState
    {
        QVector<float>& effective_frame_quality_weights;
        QVector<cv::Mat>& effective_depth_valid_masks;
        const int& erosion_pixels;
        std::uint64_t& boundary_recovered_depth_valid_pixel_count;
        QVector<cv::Mat>& reference_anchored_consensus_depths;
        QVector<cv::Mat>& contour_band_masks;
        std::uint64_t& cross_view_consensus_contour_band_pixel_count;
        std::vector<DepthGeometryLocalSourceEncoding>& local_source_encodings;
        std::vector<std::uint8_t>& orbital_gap_boundary_frames;
        std::vector<float>& tsdf;
        std::vector<float>& weight;
        std::vector<float>& evidenceSupportWeight;
        std::vector<float>& maximumObservationWeight;
        std::vector<float>& maximumEvidenceSupportObservationWeight;
        std::vector<std::uint16_t>& maximumGeometrySupportCount;
        std::vector<std::uint8_t>& strongAdaptiveSurfaceObservation;
        std::vector<std::uint16_t>& support;
        std::vector<DepthGeometrySourceMask>& geometrySourceMask;
        std::vector<std::uint16_t>& minimumInverseDepthSpread;
        std::vector<float>& surfaceTsdfWeightedSum;
        std::vector<float>& surfaceObservationWeight;
        std::vector<std::uint8_t>& crossViewRepairedSurfaceWeight;
        std::vector<float>& orbitalGapBoundaryObservationWeight;
        std::vector<float>& contourBandObservationWeight;
        std::vector<DepthVisibilityHistogram>& visibilityHistograms;
        std::vector<std::uint8_t>& primarySurfaceObservation;
        const float& base_truncation_voxels;
        const DepthUncertaintyBandEstimate& uncertainty_band;
        const bool& uncertainty_adaptation_available;
        const bool& orbital_gap_adaptive_truncation;
        const float& effective_uncertainty_adaptive_scale;
        const float& adaptive_maximum_truncation_voxels;
        const float& effective_truncation_voxels;
        const float& effective_surface_support_band_voxels;
        const float& truncation;
        const float& surface_support_distance;
        const float& weak_evidence_surface_band_voxels;
        const float& weak_evidence_surface_distance;
        const float& maximum_free_space_distance;
        DepthTsdfNarrowBandActivation& narrow_band_activation;
        std::vector<std::uint8_t>& primary_bridge_reach;
        std::atomic_bool& cancelled;
        std::atomic<int>& completed_z_slices;
        std::atomic<int>& last_progress_percent;
        std::mutex& progress_callback_mutex;
        const int& zSamples;
        const int& workerCount;
    };
    struct TsdfSurfaceState
    {
        QVector<float>& effective_frame_quality_weights;
        QVector<cv::Mat>& effective_depth_valid_masks;
        const int& erosion_pixels;
        const DepthGeometrySourceEncoding& geometry_source_encoding;
        std::vector<float>& tsdf;
        std::vector<float>& weight;
        std::vector<std::uint8_t>& strongAdaptiveSurfaceObservation;
        std::vector<std::uint16_t>& support;
        std::vector<DepthGeometrySourceMask>& geometrySourceMask;
        std::vector<std::uint16_t>& minimumInverseDepthSpread;
        std::vector<float>& surfaceObservationWeight;
        const float& maximum_voxel_size;
        const float& truncation;
        std::atomic_bool& cancelled;
        const int& workerCount;
        std::vector<std::uint8_t>& supported;
        std::vector<std::uint8_t>& adaptiveTgvExtractionSupport;
        std::vector<float>& visual_hull_completion_tsdf;
        std::vector<std::uint8_t>& visual_hull_completion_support;
        std::array<float, 3>& native_carrier_bounds_min;
        std::array<float, 3>& native_carrier_bounds_max;
        std::array<int, 3>& native_carrier_dimensions;
        std::array<int, 3>& native_carrier_cells;
        std::vector<std::uint8_t>& native_carrier_occupied;
        std::vector<float>& native_carrier_field;
    };
    bool integrateDepthFrames(const QVector<DepthTsdfFrame>&,
                              const DepthTsdfOptions&,
                              DepthTsdfResult&,
                              TsdfIntegrationState);
    bool extractAndPostprocessSurface(const QVector<DepthTsdfFrame>&,
                                      const DepthTsdfOptions&,
                                      DepthTsdfResult&,
                                      TsdfSurfaceState);
} // namespace xjw::mesh::tsdf_detail

namespace xjw::mesh::tsdf_detail
{
    struct TsdfSupportState
    {
        std::vector<std::uint8_t>& supported;
        QVector<float>& effective_frame_quality_weights;
        QVector<cv::Mat>& effective_depth_valid_masks;
        const int& erosion_pixels;
        QVector<DepthTsdfFrame>& retained_frames;
        std::vector<float>& tsdf;
        std::vector<float>& weight;
        std::vector<float>& evidenceSupportWeight;
        std::vector<float>& maximumObservationWeight;
        std::vector<float>& maximumEvidenceSupportObservationWeight;
        std::vector<std::uint16_t>& maximumGeometrySupportCount;
        std::vector<std::uint8_t>& strongAdaptiveSurfaceObservation;
        std::vector<std::uint16_t>& support;
        std::vector<DepthGeometrySourceMask>& geometrySourceMask;
        std::vector<std::uint16_t>& minimumInverseDepthSpread;
        std::vector<float>& surfaceTsdfWeightedSum;
        std::vector<float>& surfaceObservationWeight;
        std::vector<std::uint8_t>& crossViewRepairedSurfaceWeight;
        std::vector<float>& orbitalGapBoundaryObservationWeight;
        std::vector<float>& contourBandObservationWeight;
        std::vector<DepthVisibilityHistogram>& visibilityHistograms;
        const float& truncation;
        std::atomic_bool& cancelled;
        const int& workerCount;
        std::vector<std::uint8_t>& adaptiveTgvExtractionSupport;
        std::vector<float>& visual_hull_completion_tsdf;
        std::vector<std::uint8_t>& visual_hull_completion_support;
        std::array<float, 3>& native_carrier_bounds_min;
        std::array<float, 3>& native_carrier_bounds_max;
        std::array<int, 3>& native_carrier_dimensions;
        std::array<int, 3>& native_carrier_cells;
        std::vector<std::uint8_t>& native_carrier_occupied;
        std::vector<float>& native_carrier_field;
    };
    bool
    recoverVolumeSupport(const QVector<DepthTsdfFrame>&, const DepthTsdfOptions&, DepthTsdfResult&, TsdfSupportState);
} // namespace xjw::mesh::tsdf_detail

namespace xjw::mesh::tsdf_detail
{
    struct TsdfPostprocessContext
    {
        TsdfSurfaceState surface;
        std::chrono::steady_clock::time_point started;
        std::function<void(const QString&, const TriMesh&)> capture;
        std::function<void(const QString&, int)> progress;
        std::function<bool()> cancelled;
        float minimumDegenerateFaceArea = 0.0f;
    };
    std::pair<MeshFaceOrientationStatistics, bool> repairFaceOrientationSafely(TriMesh*);
    bool extractTsdfIsoSurface(const QVector<DepthTsdfFrame>&,
                               const DepthTsdfOptions&,
                               DepthTsdfResult&,
                               TsdfPostprocessContext&);
    bool
    cleanTsdfMesh(const QVector<DepthTsdfFrame>&, const DepthTsdfOptions&, DepthTsdfResult&, TsdfPostprocessContext&);
    bool simplifyTsdfMesh(const QVector<DepthTsdfFrame>&,
                          const DepthTsdfOptions&,
                          DepthTsdfResult&,
                          TsdfPostprocessContext&);
    bool finalizeTsdfMesh(const QVector<DepthTsdfFrame>&,
                          const DepthTsdfOptions&,
                          DepthTsdfResult&,
                          TsdfPostprocessContext&);
} // namespace xjw::mesh::tsdf_detail

namespace xjw::mesh::tsdf_detail
{
    DepthTsdfResult buildTsdfVolume(const QVector<DepthTsdfFrame>&, const DepthTsdfOptions&);
}
