#include "DepthTsdfStages.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;
    bool tsdf_detail::extractAndPostprocessSurface(const QVector<DepthTsdfFrame>& frames,
                                                   const DepthTsdfOptions& options,
                                                   DepthTsdfResult& result,
                                                   TsdfSurfaceState state)
    {
        auto& effective_frame_quality_weights = state.effective_frame_quality_weights;
        auto& effective_depth_valid_masks = state.effective_depth_valid_masks;
        auto& erosion_pixels = state.erosion_pixels;
        auto& geometry_source_encoding = state.geometry_source_encoding;
        auto& tsdf = state.tsdf;
        auto& weight = state.weight;
        auto& strongAdaptiveSurfaceObservation = state.strongAdaptiveSurfaceObservation;
        auto& support = state.support;
        auto& geometrySourceMask = state.geometrySourceMask;
        auto& minimumInverseDepthSpread = state.minimumInverseDepthSpread;
        auto& surfaceObservationWeight = state.surfaceObservationWeight;
        auto& maximum_voxel_size = state.maximum_voxel_size;
        auto& truncation = state.truncation;
        auto& cancelled = state.cancelled;
        auto& workerCount = state.workerCount;
        auto& supported = state.supported;
        auto& adaptiveTgvExtractionSupport = state.adaptiveTgvExtractionSupport;
        auto& visual_hull_completion_tsdf = state.visual_hull_completion_tsdf;
        auto& visual_hull_completion_support = state.visual_hull_completion_support;
        auto& native_carrier_bounds_min = state.native_carrier_bounds_min;
        auto& native_carrier_bounds_max = state.native_carrier_bounds_max;
        auto& native_carrier_dimensions = state.native_carrier_dimensions;
        auto& native_carrier_cells = state.native_carrier_cells;
        auto& native_carrier_occupied = state.native_carrier_occupied;
        auto& native_carrier_field = state.native_carrier_field;
        const auto capture_stage = [&options](const QString& stage, const TriMesh& mesh)
        {
            if (options.stageSnapshot && !mesh.empty())
            {
                options.stageSnapshot(stage, mesh);
            }
        };

        using PostprocessClock = std::chrono::steady_clock;
        const auto elapsedMilliseconds = [](const PostprocessClock::time_point& start)
        { return std::chrono::duration_cast<std::chrono::milliseconds>(PostprocessClock::now() - start).count(); };
        const auto reportProgress = [&options](const QString& stage, int percent)
        {
            if (options.execution.progress)
            {
                options.execution.reportProgress((stage).toUtf8().toStdString(), (percent) / 100.0);
            }
        };
        const auto postprocessCancelled = [&options, &result]()
        {
            if (!options.execution.isCancelled())
            {
                return false;
            }
            result.errorMessage = QStringLiteral("TSDF 后处理已取消");
            return true;
        };
        const PostprocessClock::time_point postprocess_start = PostprocessClock::now();

        TsdfPostprocessContext context{state, postprocess_start, capture_stage, reportProgress, postprocessCancelled};
        return extractTsdfIsoSurface(frames, options, result, context) &&
               cleanTsdfMesh(frames, options, result, context) && simplifyTsdfMesh(frames, options, result, context) &&
               finalizeTsdfMesh(frames, options, result, context);
    }
} // namespace xjw::mesh
