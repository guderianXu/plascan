#pragma once

#include <QString>

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

} // namespace xjw::gui::project
