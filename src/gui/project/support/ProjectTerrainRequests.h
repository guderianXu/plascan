#pragma once

#include "OrthoGenerationOptions.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <utility>

namespace xjw::gui::project
{

struct DemGenerationRequest
{
    QString sourcePointCloudPath;
    QString outputDirectory;
    double resolution = 0.0;
    QString dataType = QStringLiteral("float32");

    bool validate(QString *errorMessage = nullptr) const
    {
        if (errorMessage)
        {
            errorMessage->clear();
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

struct OrthoGenerationRequest
{
    QStringList sourceImages;
    QString demPath;
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

    QJsonObject toResolvedSettings() const
    {
        QJsonObject settings = options.toResolvedJson();
        settings[QStringLiteral("images")] = QJsonArray::fromStringList(sourceImages);
        settings[QStringLiteral("dem_path")] = demPath;
        settings[QStringLiteral("output_path")] = outputPath;
        return settings;
    }
};

} // namespace xjw::gui::project
