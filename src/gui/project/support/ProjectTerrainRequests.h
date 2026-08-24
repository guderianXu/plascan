#pragma once

#include "OrthoGenerationOptions.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace xjw::gui::project
{

inline constexpr std::int64_t kSmallBodyHardMaximumPixelCount = 50000000;

enum class DemGenerationMode
{
    Planar,
    ImageStereo,
    SmallBodyGlobal
};

struct ImageStereoDemRequestOptions
{
    QStringList sourceImages;
    double gridResolutionMeters = 2.0;
    int maximumFeatures = 20000;
    double maximumReprojectionErrorPixels = 1.5;
};

struct SmallBodyGlobalDemOptions
{
    QString targetName = QStringLiteral("Small Body");
    QString bodyFixedFrame = QStringLiteral("MODEL_LOCAL_BODY_FIXED");
    QString surfaceCoordinateUnit = QStringLiteral("m");
    bool automaticCenter = true;
    double bodyCenterX = 0.0;
    double bodyCenterY = 0.0;
    double bodyCenterZ = 0.0;
    double referenceRadiusM = 0.0;
    double angularResolutionDeg = 0.25;
    double centralMeridianDeg = 0.0;
    std::int64_t maximumPixelCount = 25000000;
    bool writeReportPreview = true;

    bool validate(QString *errorMessage = nullptr) const
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
        if (!std::isfinite(angularResolutionDeg)
            || angularResolutionDeg <= 0.0
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
            && (!std::isfinite(bodyCenterX)
                || !std::isfinite(bodyCenterY)
                || !std::isfinite(bodyCenterZ)))
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
            std::min(maximumPixelCount, kSmallBodyHardMaximumPixelCount);
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
};

struct DemGenerationRequest
{
    // Existing planar fields are intentionally retained for source compatibility.
    QString sourcePointCloudPath;
    QString outputDirectory;
    double resolution = 0.0;
    QString dataType = QStringLiteral("float32");

    DemGenerationMode mode = DemGenerationMode::Planar;
    ImageStereoDemRequestOptions imageStereoOptions;
    QString sourceSurfacePath;
    SmallBodyGlobalDemOptions smallBodyOptions;

    bool isSmallBodyGlobal() const
    {
        return mode == DemGenerationMode::SmallBodyGlobal;
    }

    bool isImageStereo() const
    {
        return mode == DemGenerationMode::ImageStereo;
    }

    bool validate(QString *errorMessage = nullptr) const
    {
        if (errorMessage)
        {
            errorMessage->clear();
        }

        if (isImageStereo())
        {
            QStringList unique_images;
            for (const QString &path : imageStereoOptions.sourceImages)
            {
                const QString normalized = path.trimmed();
                if (!normalized.isEmpty() && !unique_images.contains(normalized, Qt::CaseInsensitive))
                {
                    unique_images.append(normalized);
                }
            }
            if (unique_images.size() < 2)
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("请至少勾选两张具有相机信息且存在重叠区域的影像。");
                }
                return false;
            }
            if (!(imageStereoOptions.gridResolutionMeters > 0.0)
                || imageStereoOptions.maximumFeatures < 100
                || !(imageStereoOptions.maximumReprojectionErrorPixels > 0.0))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("影像立体 DEM 的分辨率、特征点数或重投影阈值无效。");
                }
                return false;
            }
            return true;
        }

        if (isSmallBodyGlobal())
        {
            const QString surface_path = sourceSurfacePath.trimmed();
            if (surface_path.isEmpty())
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("请选择用于生成全球 DEM/DOM 的 PLY/OBJ 三角网格。");
                }
                return false;
            }

            const QString lower_path = surface_path.toLower();
            if (!lower_path.endsWith(QStringLiteral(".ply"))
                && !lower_path.endsWith(QStringLiteral(".obj")))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("小天体全球产品仅支持 PLY 或 OBJ 三角网格。");
                }
                return false;
            }
            return smallBodyOptions.validate(errorMessage);
        }

        if (sourcePointCloudPath.trimmed().isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("请选择用于生成 DEM 的点云文件。");
            }
            return false;
        }
        if (resolution < 0.0)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("DEM 分辨率不能为负数。");
            }
            return false;
        }
        if (dataType.trimmed().isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("DEM 数据类型不能为空。");
            }
            return false;
        }
        return true;
    }
};

enum class OrthoGenerationMode
{
    Standard,
    Rpc
};

struct OrthoGenerationRequest
{
    OrthoGenerationMode mode = OrthoGenerationMode::Standard;
    QStringList sourceImages;
    QString demPath;
    QString pointCloudPath;
    QString outputPath;
    xjw::OrthoGenerationOptions options;

    static bool fromJson(const QJsonObject &settings,
                         OrthoGenerationRequest *request,
                         QString *errorMessage = nullptr)
    {
        if (!request)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("内部错误：正射影像请求输出为空。");
            }
            return false;
        }

        OrthoGenerationRequest parsed;
        parsed.mode = settings.value(QStringLiteral("product_mode")).toString()
                    == QLatin1String("rpc")
            ? OrthoGenerationMode::Rpc
            : OrthoGenerationMode::Standard;
        const QJsonArray images = settings.value(QStringLiteral("images")).toArray();
        parsed.sourceImages.reserve(images.size());
        for (const QJsonValue &value : images)
        {
            const QString image_path = value.toString().trimmed();
            if (!image_path.isEmpty())
            {
                parsed.sourceImages.append(image_path);
            }
        }
        parsed.demPath = settings.value(QStringLiteral("dem_path")).toString().trimmed();
        parsed.pointCloudPath =
            settings.value(QStringLiteral("point_cloud_path")).toString().trimmed();
        parsed.outputPath = settings.value(QStringLiteral("output_path")).toString().trimmed();
        if (!xjw::OrthoGenerationOptions::fromJson(
                settings, &parsed.options, errorMessage))
        {
            return false;
        }

        *request = std::move(parsed);
        if (errorMessage)
        {
            errorMessage->clear();
        }
        return true;
    }

    bool validate(QString *errorMessage = nullptr) const
    {
        if (!options.validate(errorMessage))
        {
            return false;
        }
        if (isRpc() && sourceImages.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("请至少选择一张带内嵌地理定位模型的影像。");
            }
            return false;
        }
        if (outputPath.trimmed().isEmpty() && demPath.trimmed().isEmpty() &&
            sourceImages.isEmpty())
        {
            // 空路径允许由项目上下文推导；这里仅明确该请求不是无效对象。
            if (errorMessage)
            {
                errorMessage->clear();
            }
        }
        return true;
    }

    bool isRpc() const
    {
        return mode == OrthoGenerationMode::Rpc;
    }

    QJsonObject toResolvedSettings() const
    {
        QJsonObject settings = options.toResolvedJson();
        settings[QStringLiteral("product_mode")] = isRpc()
            ? QStringLiteral("rpc") : QStringLiteral("standard");
        settings[QStringLiteral("images")] = QJsonArray::fromStringList(sourceImages);
        settings[QStringLiteral("dem_path")] = demPath;
        settings[QStringLiteral("point_cloud_path")] = pointCloudPath;
        settings[QStringLiteral("output_path")] = outputPath;
        return settings;
    }
};

} // namespace xjw::gui::project
