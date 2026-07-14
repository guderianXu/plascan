#include "MarkerQualityReport.h"

#include <algorithm>
#include <cmath>

namespace xjw::control_points
{

namespace
{

template <typename Range, typename Value, typename Inlier>
ResidualStatistics statistics(const Range &range, Value value, Inlier inlier)
{
    ResidualStatistics result;
    double sum = 0.0;
    double sum_squared = 0.0;
    for (const auto &entry : range)
    {
        const double residual = std::abs(value(entry));
        if (!std::isfinite(residual)) continue;
        ++result.totalCount;
        if (inlier(entry)) ++result.inlierCount;
        sum += residual;
        sum_squared += residual * residual;
        result.maximum = std::max(result.maximum, residual);
    }
    if (result.totalCount > 0)
    {
        result.mean = sum / static_cast<double>(result.totalCount);
        result.rms = std::sqrt(sum_squared / static_cast<double>(result.totalCount));
    }
    return result;
}

} // namespace

MarkerQualityReport buildMarkerQualityReport(
    const ControlNetworkResult &network,
    const QVector<ScaleBarResidual> &scaleBarResiduals)
{
    MarkerQualityReport report;
    report.valid = network.ok;
    report.controls = statistics(network.controlResiduals,
                                 [](const MarkerResidual &residual) { return residual.total; },
                                 [](const MarkerResidual &residual) { return residual.inlier; });
    report.checkPoints = statistics(network.checkPointResiduals,
                                    [](const MarkerResidual &residual) { return residual.total; },
                                    [](const MarkerResidual &) { return false; });

    QVector<ScaleBarResidual> controls;
    QVector<ScaleBarResidual> checks;
    for (const ScaleBarResidual &residual : scaleBarResiduals)
    {
        (residual.role == ScaleBarRole::Control ? controls : checks).push_back(residual);
    }
    const auto scale_value = [](const ScaleBarResidual &residual) { return residual.residual; };
    const auto scale_inlier = [](const ScaleBarResidual &) { return true; };
    report.controlScaleBars = statistics(controls, scale_value, scale_inlier);
    report.checkScaleBars = statistics(checks, scale_value, scale_inlier);
    return report;
}

} // namespace xjw::control_points
