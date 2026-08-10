#include "SmallBodyGlobalProducts.h"

#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{
namespace
{

constexpr std::int64_t kHardMaximumPixelCount = 50000000;

} // namespace

bool SmallBodyGlobalOptions::validate(QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }

    if (targetName.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("目标天体名称不能为空。");
        }
        return false;
    }

    const QString normalized_frame = bodyFixedFrame.trimmed().toUpper();
    if (normalized_frame.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("体固连坐标系名称不能为空。");
        }
        return false;
    }
    if (normalized_frame.contains(QStringLiteral("J2000"))
        || normalized_frame.contains(QStringLiteral("ICRF"))
        || normalized_frame.contains(QStringLiteral("INERTIAL")))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "全球表面 DEM/DOM 必须使用体固连坐标。J2000、ICRF 等惯性坐标中的模型，"
                "需要先按观测时刻转换到目标天体的体固连坐标系。");
        }
        return false;
    }
    const QString normalized_unit = surfaceCoordinateUnit.trimmed().toLower();
    if (normalized_unit != QLatin1String("m") && normalized_unit != QLatin1String("km"))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("网格坐标单位只支持 m 或 km。");
        }
        return false;
    }
    if (!std::isfinite(angularResolutionDeg) || angularResolutionDeg <= 0.0
        || angularResolutionDeg > 90.0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("角分辨率必须在 0 到 90 度之间。");
        }
        return false;
    }
    if (!std::isfinite(referenceRadiusM) || referenceRadiusM < 0.0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("参考半径不能为负数。");
        }
        return false;
    }
    if (!std::isfinite(centralMeridianDeg)
        || centralMeridianDeg < -180.0 || centralMeridianDeg > 180.0)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("中央经线必须在 -180° 到 180° 之间。");
        }
        return false;
    }
    if (!automaticCenter
        && (!std::isfinite(bodyCenter[0]) || !std::isfinite(bodyCenter[1])
            || !std::isfinite(bodyCenter[2])))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("手动体心坐标必须为有限数值。");
        }
        return false;
    }

    const double width_value = std::ceil(360.0 / angularResolutionDeg);
    const double height_value = std::ceil(180.0 / angularResolutionDeg);
    if (!std::isfinite(width_value) || !std::isfinite(height_value)
        || width_value > std::numeric_limits<int>::max()
        || height_value > std::numeric_limits<int>::max())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("角分辨率过小，全球栅格尺寸无法表示。");
        }
        return false;
    }
    const auto width = static_cast<std::int64_t>(width_value);
    const auto height = static_cast<std::int64_t>(height_value);
    const std::int64_t effective_maximum =
        std::min(maximumPixelCount, kHardMaximumPixelCount);
    if (maximumPixelCount <= 0 || width <= 0 || height <= 0
        || width > effective_maximum / height)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "目标全球栅格超过安全像素预算：%1 × %2，当前有效上限为 %3。")
                                .arg(width)
                                .arg(height)
                                .arg(effective_maximum);
        }
        return false;
    }
    return true;
}

} // namespace xjw
