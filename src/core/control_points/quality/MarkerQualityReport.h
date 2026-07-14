#pragma once

#include "registration/ControlNetworkSolver.h"

#include <QVector>

namespace xjw::control_points
{

struct ScaleBarResidual
{
    ScaleBarId scaleBarId;
    ScaleBarRole role = ScaleBarRole::Control;
    double measuredDistance = 0.0;
    double estimatedDistance = 0.0;
    double residual = 0.0;
};

struct ResidualStatistics
{
    int totalCount = 0;
    int inlierCount = 0;
    double mean = 0.0;
    double rms = 0.0;
    double maximum = 0.0;
};

struct MarkerQualityReport
{
    bool valid = false;
    ResidualStatistics controls;
    ResidualStatistics checkPoints;
    ResidualStatistics controlScaleBars;
    ResidualStatistics checkScaleBars;
};

MarkerQualityReport buildMarkerQualityReport(
    const ControlNetworkResult &network,
    const QVector<ScaleBarResidual> &scaleBarResiduals);

} // namespace xjw::control_points
