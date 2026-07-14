#include "MarkerDetectionJobBuilder.h"

#include "ProjectData.h"
#include "ProjectIO.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QSet>

namespace xjw::gui::markers
{

MarkerDetectionJobBuildResult MarkerDetectionJobBuilder::build(
    const ProjectData &projectData,
    const MarkerDetectionJobBuildOptions &options)
{
    MarkerDetectionJobBuildResult result;
    result.job.baseRevision = options.baseRevision;
    result.job.targetFamilies = options.targetFamilies;
    result.job.detectorOptions = options.detectorOptions;
    result.job.maxConcurrentImages = options.maxConcurrentImages;

    if (!projectData.hasProject())
    {
        result.errors.push_back(QStringLiteral("尚未打开项目"));
        return result;
    }
    if (options.targetFamilies.isEmpty())
    {
        result.errors.push_back(QStringLiteral("未选择标靶类型"));
        return result;
    }

    const QString project_path = projectData.currentProjectPath();
    const QJsonArray images = projectData.coreFilesMeta()
                                  .value(QStringLiteral("images"))
                                  .toArray();
    QSet<QString> image_ids;
    for (const QJsonValue &value : images)
    {
        const QJsonObject metadata = value.toObject();
        const QString image_id = metadata.value(QStringLiteral("image_uuid")).toString().trimmed();
        const QString stored_path = metadata.value(QStringLiteral("path")).toString().trimmed();
        const QString image_path = ProjectIO::resolveProjectResourcePath(project_path, stored_path);

        if (image_id.isEmpty())
        {
            result.errors.push_back(QStringLiteral("影像缺少稳定 UUID: %1").arg(stored_path));
            continue;
        }
        if (image_ids.contains(image_id))
        {
            result.errors.push_back(QStringLiteral("项目中存在重复影像 UUID: %1").arg(image_id));
            continue;
        }
        if (image_path.isEmpty() || !QFileInfo::exists(image_path))
        {
            result.errors.push_back(QStringLiteral("检测影像不存在: %1").arg(image_path));
            continue;
        }

        QString mask_path;
        const QString stored_mask = metadata.value(QStringLiteral("mask_path")).toString().trimmed();
        if (!stored_mask.isEmpty())
        {
            mask_path = ProjectIO::resolveProjectResourcePath(project_path, stored_mask);
            if (!QFileInfo::exists(mask_path))
            {
                result.errors.push_back(QStringLiteral("影像声明的检测蒙版不存在: %1").arg(mask_path));
                continue;
            }
        }
        else
        {
            mask_path = ProjectIO::findMaskForImage(project_path, image_path);
        }

        QString signature = metadata.value(QStringLiteral("image_content_signature")).toString();
        if (signature.isEmpty())
        {
            signature = metadata.value(QStringLiteral("content_signature")).toString();
        }
        result.job.images.push_back({image_id,
                                     QDir::cleanPath(image_path),
                                     QDir::cleanPath(mask_path),
                                     signature});
        image_ids.insert(image_id);
    }

    if (images.isEmpty())
    {
        result.errors.push_back(QStringLiteral("项目中没有可检测的影像"));
    }
    result.ok = !result.job.images.isEmpty();
    return result;
}

} // namespace xjw::gui::markers
