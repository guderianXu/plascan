#include "ProjectReferenceTerrainBa.h"

#include "DemDomIO.h"
#include "ReferenceTerrainPrior.h"

#include <QFileInfo>

#include <cmath>

namespace xjw::gui::project {
namespace {

ReferenceTerrainGrid referenceGridFromDem(const DemGridData &dem)
{
    ReferenceTerrainGrid grid;
    grid.width = dem.width;
    grid.height = dem.height;
    grid.originX = dem.minX;
    grid.originY = dem.minY;
    grid.pixelSizeX = dem.stepX;
    grid.pixelSizeY = dem.stepY;
    grid.nodata = -9999.0;
    grid.heights.resize(static_cast<std::size_t>(std::max(0, dem.width) * std::max(0, dem.height)),
                        grid.nodata);

    if (!dem.isValid() || dem.elevation.type() != CV_32FC1)
    {
        return grid;
    }

    for (int row = 0; row < dem.height; ++row)
    {
        for (int col = 0; col < dem.width; ++col)
        {
            const bool valid = dem.validMask.empty() || dem.validMask.at<uchar>(row, col) != 0;
            const float z = dem.elevation.at<float>(row, col);
            const std::size_t index = static_cast<std::size_t>(row * dem.width + col);
            grid.heights[index] = valid && std::isfinite(static_cast<double>(z))
                ? static_cast<double>(z)
                : grid.nodata;
        }
    }
    return grid;
}

QJsonObject summaryFromStats(const QString &path,
                             const ReferenceTerrainPriorOptions &options,
                             const ReferenceTerrainPriorStats &stats)
{
    QJsonObject summary;
    summary[QStringLiteral("enabled")] = true;
    summary[QStringLiteral("source_type")] = QStringLiteral("DEM");
    summary[QStringLiteral("path")] = QFileInfo(path).absoluteFilePath();
    summary[QStringLiteral("sigma_m")] = options.sigmaMeters;
    summary[QStringLiteral("max_association_distance_m")] = options.maxAssociationDistanceMeters;
    summary[QStringLiteral("huber_delta_m")] = options.huberDeltaMeters;
    summary[QStringLiteral("input_tracks")] = stats.inputTrackCount;
    summary[QStringLiteral("associated_tracks")] = stats.associatedTrackCount;
    summary[QStringLiteral("rejected_no_height")] = stats.rejectedNoHeightCount;
    summary[QStringLiteral("rejected_by_distance")] = stats.rejectedByDistanceCount;
    summary[QStringLiteral("rms_before_m")] = stats.rmsBeforeMeters;
    summary[QStringLiteral("median_abs_before_m")] = stats.medianAbsBeforeMeters;
    return summary;
}

} // namespace

ReferenceTerrainBaApplyResult applyReferenceTerrainPriorToBundleAdjust(
    std::vector<xjw::BATrack> *tracks,
    xjw::gui::BaServiceOptions *options)
{
    ReferenceTerrainBaApplyResult result;
    if (!options || !options->enableReferenceTerrainPrior)
    {
        result.success = true;
        result.summary[QStringLiteral("enabled")] = false;
        return result;
    }

    if (!tracks)
    {
        result.errorMessage = QStringLiteral("BA tracks 未初始化，无法应用参考地形约束");
        return result;
    }

    const QString demPath = options->referenceTerrainDemPath.trimmed();
    if (demPath.isEmpty())
    {
        result.errorMessage = QStringLiteral("参考 DEM 路径未指定");
        return result;
    }
    if (!QFileInfo::exists(demPath))
    {
        result.errorMessage = QStringLiteral("参考 DEM 不存在: %1").arg(demPath);
        return result;
    }

    DemGridData dem;
    QString error;
    if (!DemDomIO::readDemRaster(demPath, &dem, &error))
    {
        result.errorMessage = QStringLiteral("读取参考 DEM 失败: %1").arg(error);
        return result;
    }
    if (!dem.isValid())
    {
        result.errorMessage = QStringLiteral("参考 DEM 无有效高程栅格: %1").arg(demPath);
        return result;
    }

    ReferenceTerrainPriorOptions priorOptions;
    priorOptions.enabled = true;
    priorOptions.sigmaMeters = options->referenceTerrainSigmaMeters;
    priorOptions.maxAssociationDistanceMeters = options->referenceTerrainMaxAssociationDistanceMeters;
    priorOptions.huberDeltaMeters = options->referenceTerrainHuberDeltaMeters;

    const ReferenceTerrainGrid grid = referenceGridFromDem(dem);
    const ReferenceTerrainPriorStats stats =
        ReferenceTerrainPrior::attachHeightPlaneConstraints(grid, tracks, priorOptions);

    BAOptions priorBaOptions = ReferenceTerrainPrior::makeBundleAdjustOptions(priorOptions);
    if (priorBaOptions.enableLaserPlaneConstraints && stats.associatedTrackCount > 0)
    {
        options->baOpt.enableLaserPlaneConstraints = true;
        options->baOpt.laserPlaneWeight = priorBaOptions.laserPlaneWeight;
        options->baOpt.laserHuberDeltaMeters = priorBaOptions.laserHuberDeltaMeters;
        options->baOpt.refineCameraPose = true;
    }

    result.success = true;
    result.summary = summaryFromStats(demPath, priorOptions, stats);
    result.summary[QStringLiteral("ba_constraints_enabled")] =
        options->baOpt.enableLaserPlaneConstraints && stats.associatedTrackCount > 0;
    options->referenceTerrainPriorSummary = result.summary;
    return result;
}

} // namespace xjw::gui::project
