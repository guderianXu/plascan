#include "cli_common.h"
#include "CliJsonIO.h"
#include "CliPathUtils.h"

#include "DenseCloudRefinementService.h"

#include <QJsonArray>
#include <QJsonObject>

#include <algorithm>
#include <iostream>
#include <string>

namespace
{

QJsonObject passReportToJson(const xjw::mvs::TerrainHeightSpikeFilterReport &report)
{
    return QJsonObject{
        {QStringLiteral("input_points"), static_cast<double>(report.inputPoints)},
        {QStringLiteral("output_points"), static_cast<double>(report.outputPoints)},
        {QStringLiteral("removed_points"), static_cast<double>(report.removedPoints)},
        {QStringLiteral("local_plane_removed_points"),
         static_cast<double>(report.localPlaneRemovedPoints)},
        {QStringLiteral("median_cell_z_range_before"), report.medianCellZRangeBefore},
        {QStringLiteral("p95_cell_z_range_before"), report.p95CellZRangeBefore},
        {QStringLiteral("median_cell_z_range_after"), report.medianCellZRangeAfter},
        {QStringLiteral("p95_cell_z_range_after"), report.p95CellZRangeAfter}
    };
}

QJsonObject resultToJson(const xjw::mvs::DenseCloudRefinementRequest &request,
                         const xjw::mvs::DenseCloudRefinementResult &result)
{
    QJsonObject localPlane{
        {QStringLiteral("enabled"), request.filterOptions.localPlaneFilterEnabled},
        {QStringLiteral("min_points"), request.filterOptions.localPlaneMinPoints},
        {QStringLiteral("min_residual_threshold"),
         request.filterOptions.localPlaneMinResidualThreshold},
        {QStringLiteral("mad_multiplier"), request.filterOptions.localPlaneMadMultiplier},
        {QStringLiteral("removed_points"),
         static_cast<double>(result.report.localPlaneRemovedPoints)}
    };
    QJsonObject filter = passReportToJson(result.report);
    filter[QStringLiteral("enabled")] = request.filterOptions.enabled;
    filter[QStringLiteral("grid_cells")] = request.filterOptions.gridResolution;
    filter[QStringLiteral("min_cell_points")] = request.filterOptions.minCellPoints;
    filter[QStringLiteral("min_height_threshold")] = request.filterOptions.minHeightThreshold;
    filter[QStringLiteral("mad_multiplier")] = request.filterOptions.madMultiplier;
    filter[QStringLiteral("local_plane_filter")] = localPlane;

    QJsonArray passReports;
    for (std::size_t index = 0; index < result.passReports.size(); ++index)
    {
        QJsonObject pass = passReportToJson(result.passReports[index]);
        pass[QStringLiteral("pass_index")] = static_cast<int>(index + 1);
        passReports.append(pass);
    }

    return QJsonObject{
        {QStringLiteral("status"), QStringLiteral("ok")},
        {QStringLiteral("input"), xjw::cli::fromStdString(request.inputPath)},
        {QStringLiteral("output"), xjw::cli::fromStdString(request.outputPath)},
        {QStringLiteral("mode"), xjw::cli::fromStdString(result.mode)},
        {QStringLiteral("streaming_chunk_mb"), request.streamingChunkMb},
        {QStringLiteral("terrain_filter_passes"), request.filterPasses},
        {QStringLiteral("input_points"), static_cast<double>(result.report.inputPoints)},
        {QStringLiteral("output_points"), static_cast<double>(result.report.outputPoints)},
        {QStringLiteral("terrain_spike_filter"), filter},
        {QStringLiteral("pass_reports"), passReports}
    };
}

} // namespace

int main(int argc, char **argv)
{
    CLI::App app{"Refine dense point clouds and write a quality report"};

    std::string inputPath;
    std::string outputPath;
    std::string reportPath;
    bool disableTerrainSpikeFilter = false;
    int terrainGridCells = 260;
    int terrainMinCellPoints = 32;
    float terrainMinHeightThreshold = 0.25f;
    float terrainMadMultiplier = 3.0f;
    bool terrainLocalPlaneFilter = true;
    int terrainLocalPlaneMinPoints = 12;
    float terrainLocalPlaneMinResidualThreshold = 0.12f;
    float terrainLocalPlaneMadMultiplier = 4.0f;
    int terrainFilterPasses = 2;
    int streamingChunkMb = 128;

    app.add_option("--input", inputPath, "Input dense cloud PLY path")->required()->check(CLI::ExistingFile);
    app.add_option("--output", outputPath, "Output refined dense cloud PLY path")->required();
    app.add_option("--report-json", reportPath, "Output JSON quality report path")->required();
    app.add_flag("--disable-terrain-spike-filter", disableTerrainSpikeFilter,
                 "Copy finite points without terrain height spike filtering");
    app.add_option("--terrain-grid-cells", terrainGridCells,
                   "XY grid resolution used by the terrain spike filter")->check(CLI::Range(1, 1024));
    app.add_option("--terrain-min-cell-points", terrainMinCellPoints,
                   "Minimum point count for robust local terrain statistics")
        ->check(CLI::Range(1, 1000000));
    app.add_option("--terrain-min-height-threshold", terrainMinHeightThreshold,
                   "Minimum absolute height threshold in local terrain cells")
        ->check(CLI::PositiveNumber);
    app.add_option("--terrain-mad-multiplier", terrainMadMultiplier,
                   "MAD multiplier used by the robust local terrain height gate")
        ->check(CLI::PositiveNumber);
    app.add_flag("--terrain-local-plane-filter,!--disable-terrain-local-plane-filter",
                 terrainLocalPlaneFilter, "Enable the second-pass local plane residual filter");
    app.add_option("--terrain-local-plane-min-points", terrainLocalPlaneMinPoints,
                   "Minimum neighborhood point count for local plane residual filtering")
        ->check(CLI::Range(3, 1000000));
    app.add_option("--terrain-local-plane-min-residual-threshold",
                   terrainLocalPlaneMinResidualThreshold,
                   "Minimum absolute residual threshold for local plane filtering")
        ->check(CLI::PositiveNumber);
    app.add_option("--terrain-local-plane-mad-multiplier", terrainLocalPlaneMadMultiplier,
                   "MAD multiplier used by the local plane residual gate")
        ->check(CLI::PositiveNumber);
    app.add_option("--terrain-filter-passes", terrainFilterPasses,
                   "Number of terrain cleanup passes")
        ->check(CLI::Range(1, 8));
    app.add_option("--streaming-chunk-mb", streamingChunkMb,
                   "Chunk size for binary PLY streaming mode")
        ->check(CLI::Range(1, 2048));

    CLI11_PARSE(app, argc, argv);

    xjw::mvs::DenseCloudRefinementRequest request;
    request.inputPath = inputPath;
    request.outputPath = outputPath;
    request.filterPasses = std::clamp(terrainFilterPasses, 1, 8);
    request.streamingChunkMb = std::clamp(streamingChunkMb, 1, 2048);
    request.filterOptions.enabled = !disableTerrainSpikeFilter;
    request.filterOptions.gridResolution = std::clamp(terrainGridCells, 1, 1024);
    request.filterOptions.minCellPoints = std::max(1, terrainMinCellPoints);
    request.filterOptions.minHeightThreshold = terrainMinHeightThreshold;
    request.filterOptions.madMultiplier = terrainMadMultiplier;
    request.filterOptions.localPlaneFilterEnabled = terrainLocalPlaneFilter;
    request.filterOptions.localPlaneMinPoints = std::max(3, terrainLocalPlaneMinPoints);
    request.filterOptions.localPlaneMinResidualThreshold = terrainLocalPlaneMinResidualThreshold;
    request.filterOptions.localPlaneMadMultiplier = terrainLocalPlaneMadMultiplier;

    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;
    if (!xjw::mvs::refineDenseCloud(request, &result, &error))
    {
        cli::fatal(error, cli::EXIT_ALGO_ERR);
    }

    QString jsonError;
    if (!xjw::cli::writeJsonFile(xjw::cli::fromStdString(reportPath), resultToJson(request, result), &jsonError))
    {
        cli::fatal(jsonError.toStdString(), cli::EXIT_IO_ERR);
    }

    std::cout << "dense_cloud_refine_cli(" << result.mode << "): " << result.report.inputPoints
              << " -> " << result.report.outputPoints << " points, removed "
              << result.report.removedPoints << " points\n";
    return cli::EXIT_OK;
}
