#include "SurveyControlReport.h"

#include <QJsonArray>
#include <QJsonValue>
#include <QStringList>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::qc
{

namespace
{

struct ResidualStats
{
    int totalCount = 0;
    int enabledCount = 0;
    int residualCount = 0;
    double sumSquares = 0.0;
    double maxResidual = 0.0;

    double rmse() const
    {
        return residualCount > 0
            ? std::sqrt(sumSquares / static_cast<double>(residualCount))
            : 0.0;
    }
};

bool jsonNumber(const QJsonObject &object, const QString &key, double *value)
{
    if (!value)
    {
        return false;
    }

    const QJsonValue jsonValue = object.value(key);
    if (!jsonValue.isDouble())
    {
        return false;
    }

    const double number = jsonValue.toDouble();
    if (!std::isfinite(number))
    {
        return false;
    }

    *value = number;
    return true;
}

bool recordEnabled(const QJsonObject &record)
{
    return !record.contains(QStringLiteral("enabled")) || record.value(QStringLiteral("enabled")).toBool(true);
}

double magnitudeFromComponents(const QJsonObject &residual)
{
    double sumSquares = 0.0;
    int componentCount = 0;

    const QStringList keys{
        QStringLiteral("x_m"),
        QStringLiteral("y_m"),
        QStringLiteral("z_m"),
        QStringLiteral("horizontal_m"),
        QStringLiteral("vertical_m")
    };

    for (const QString &key : keys)
    {
        double component = 0.0;
        if (jsonNumber(residual, key, &component))
        {
            sumSquares += component * component;
            ++componentCount;
        }
    }

    return componentCount > 0 ? std::sqrt(sumSquares) : std::numeric_limits<double>::quiet_NaN();
}

double residualMagnitudeFromRecord(const QJsonObject &record, bool scaleBar)
{
    double value = 0.0;
    if (scaleBar && jsonNumber(record, QStringLiteral("residual_m"), &value))
    {
        return std::abs(value);
    }

    const QJsonObject residual = record.value(QStringLiteral("residual")).toObject();
    if (jsonNumber(residual, QStringLiteral("total_m"), &value))
    {
        return std::abs(value);
    }

    const double componentMagnitude = magnitudeFromComponents(residual);
    if (std::isfinite(componentMagnitude))
    {
        return componentMagnitude;
    }

    if (scaleBar)
    {
        double measured = 0.0;
        double estimated = 0.0;
        if (jsonNumber(record, QStringLiteral("measured_m"), &measured) &&
            jsonNumber(record, QStringLiteral("estimated_m"), &estimated))
        {
            return std::abs(estimated - measured);
        }
    }

    return std::numeric_limits<double>::quiet_NaN();
}

ResidualStats summarizeResidualRecords(const QJsonArray &records, bool scaleBar)
{
    ResidualStats stats;
    stats.totalCount = records.size();

    for (const QJsonValue &value : records)
    {
        const QJsonObject record = value.toObject();
        if (!recordEnabled(record))
        {
            continue;
        }

        ++stats.enabledCount;
        const double residual = residualMagnitudeFromRecord(record, scaleBar);
        if (!std::isfinite(residual))
        {
            continue;
        }

        ++stats.residualCount;
        stats.sumSquares += residual * residual;
        stats.maxResidual = std::max(stats.maxResidual, residual);
    }

    return stats;
}

double thresholdValue(const QJsonObject &thresholds,
                      const QString &preferredKey,
                      const QString &fallbackKey,
                      double defaultValue)
{
    double value = 0.0;
    if (jsonNumber(thresholds, preferredKey, &value) || jsonNumber(thresholds, fallbackKey, &value))
    {
        return value;
    }
    return defaultValue;
}

} // namespace

QJsonObject buildSurveyControlSummary(const QJsonObject &projectMeta)
{
    const QJsonObject survey = projectMeta.value(QStringLiteral("survey_control")).toObject();
    const QJsonArray controlPoints = survey.value(QStringLiteral("control_points")).toArray();
    const QJsonArray checkPoints = survey.value(QStringLiteral("check_points")).toArray();
    const QJsonArray scaleBars = survey.value(QStringLiteral("scale_bars")).toArray();

    const ResidualStats controlStats = summarizeResidualRecords(controlPoints, false);
    const ResidualStats checkStats = summarizeResidualRecords(checkPoints, false);
    const ResidualStats scaleStats = summarizeResidualRecords(scaleBars, true);

    const QJsonObject thresholds = survey.value(QStringLiteral("quality_thresholds")).toObject();
    const double controlWarn = thresholdValue(thresholds,
                                             QStringLiteral("control_point_rmse_warn_m"),
                                             QStringLiteral("gcp_rmse_warn_m"),
                                             0.0);
    const double checkWarn = thresholdValue(thresholds,
                                           QStringLiteral("check_point_rmse_warn_m"),
                                           QStringLiteral("checkpoint_rmse_warn_m"),
                                           0.0);
    const double scaleWarn = thresholdValue(thresholds,
                                           QStringLiteral("scale_bar_rmse_warn_m"),
                                           QStringLiteral("scalebar_rmse_warn_m"),
                                           0.0);

    bool warn = false;
    warn = warn || (controlWarn > 0.0 && controlStats.residualCount > 0 && controlStats.rmse() > controlWarn);
    warn = warn || (checkWarn > 0.0 && checkStats.residualCount > 0 && checkStats.rmse() > checkWarn);
    warn = warn || (scaleWarn > 0.0 && scaleStats.residualCount > 0 && scaleStats.rmse() > scaleWarn);

    const bool hasSurveyData = controlStats.totalCount > 0 || checkStats.totalCount > 0 || scaleStats.totalCount > 0;

    QJsonObject summary;
    summary[QStringLiteral("control_point_count")] = controlStats.totalCount;
    summary[QStringLiteral("enabled_control_point_count")] = controlStats.enabledCount;
    summary[QStringLiteral("control_point_residual_count")] = controlStats.residualCount;
    summary[QStringLiteral("control_point_rmse_m")] = controlStats.rmse();
    summary[QStringLiteral("control_point_max_residual_m")] = controlStats.maxResidual;
    summary[QStringLiteral("check_point_count")] = checkStats.totalCount;
    summary[QStringLiteral("enabled_check_point_count")] = checkStats.enabledCount;
    summary[QStringLiteral("check_point_residual_count")] = checkStats.residualCount;
    summary[QStringLiteral("check_point_rmse_m")] = checkStats.rmse();
    summary[QStringLiteral("check_point_max_residual_m")] = checkStats.maxResidual;
    summary[QStringLiteral("scale_bar_count")] = scaleStats.totalCount;
    summary[QStringLiteral("enabled_scale_bar_count")] = scaleStats.enabledCount;
    summary[QStringLiteral("scale_bar_residual_count")] = scaleStats.residualCount;
    summary[QStringLiteral("scale_bar_rmse_m")] = scaleStats.rmse();
    summary[QStringLiteral("scale_bar_max_residual_m")] = scaleStats.maxResidual;
    summary[QStringLiteral("status")] = hasSurveyData
        ? (warn ? QStringLiteral("warn") : QStringLiteral("ok"))
        : QStringLiteral("missing");
    return summary;
}

} // namespace xjw::qc
