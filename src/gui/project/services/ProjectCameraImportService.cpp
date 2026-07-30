#include "ProjectCameraImportService.h"

#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"

#include <QDir>
#include <QFileInfo>

namespace xjw::gui::project {

using xjw::common::project::parseTsaiCamera;

SingleCameraImportStatus buildSingleCameraImport(const QString &imagePath,
                                                 const QString &tsaiPath,
                                                 SingleCameraImportResult *out)
{
    // 先初始化输出，保证调用方即使失败也能拿到确定状态。
    if (!out) return SingleCameraImportStatus::ParseFailed;
    out->imageAbsPath = QFileInfo(imagePath).absoluteFilePath();
    out->cameraMeta = QJsonObject();
    out->error.clear();

    // UI 传入空路径通常来自无效选中项，直接返回明确错误。
    if (out->imageAbsPath.isEmpty()) {
        out->error = QStringLiteral("请选择有效的影像");
        return SingleCameraImportStatus::EmptyImagePath;
    }

    // 统一复用 ProjectSupportUtils 的 tsai 解析逻辑。
    QString parseErr;
    if (!parseTsaiCamera(tsaiPath, &out->cameraMeta, &parseErr)) {
        out->error = parseErr;
        return SingleCameraImportStatus::ParseFailed;
    }
    out->cameraMeta[QStringLiteral("source_file")] = QFileInfo(tsaiPath).absoluteFilePath();

    return SingleCameraImportStatus::Ok;
}

BatchCameraImportStatus buildBatchCameraImport(const QString &tsaiFolder,
                                               const QStringList &projectImages,
                                               BatchCameraImportResult *out)
{
    // 初始化统计字段，避免上次调用残留。
    if (!out) return BatchCameraImportStatus::NoImportable;
    out->cameraMetaByImage.clear();
    out->ambiguousCount = 0;
    out->parseFailedCount = 0;
    out->unmatchedCount = 0;
    out->parseErrors.clear();

    // 1) 收集目录中的 tsai 文件。
    QDir tsaiDir(tsaiFolder);
    const QFileInfoList tsaiFiles = tsaiDir.entryInfoList(
        QStringList() << QStringLiteral("*.tsai") << QStringLiteral("*.TSAI"),
        QDir::Files | QDir::NoSymLinks,
        QDir::Name
    );
    if (tsaiFiles.isEmpty()) {
        return BatchCameraImportStatus::NoTsaiFiles;
    }

    if (projectImages.isEmpty()) {
        return BatchCameraImportStatus::NoProjectImages;
    }

    // 2) 构建影像 baseName 索引（同名多影像会形成歧义）。
    QMap<QString, QStringList> imageBaseMap;
    for (const QString &imagePath : projectImages) {
        const QFileInfo imageInfo(imagePath);
        const QString key = imageInfo.completeBaseName().toLower();
        if (!key.isEmpty()) imageBaseMap[key].append(imageInfo.absoluteFilePath());
    }

    // 3) 遍历 tsai 文件并尝试匹配与解析。
    for (const QFileInfo &tsaiFile : tsaiFiles) {
        const QString key = tsaiFile.completeBaseName().toLower();
        const QStringList matchedImages = imageBaseMap.value(key);
        if (matchedImages.isEmpty()) {
            ++out->unmatchedCount;
            continue;
        }
        // 同名命中多张影像时跳过，避免误绑定。
        if (matchedImages.size() > 1) {
            ++out->ambiguousCount;
            continue;
        }

        QJsonObject cameraMeta;
        QString parseErr;
        if (!parseTsaiCamera(tsaiFile.absoluteFilePath(), &cameraMeta, &parseErr)) {
            ++out->parseFailedCount;
            out->parseErrors.push_back(parseErr);
            continue;
        }
        cameraMeta[QStringLiteral("source_file")] = tsaiFile.absoluteFilePath();

        out->cameraMetaByImage.insert(matchedImages.first(), cameraMeta);
    }

    if (out->cameraMetaByImage.isEmpty()) {
        return BatchCameraImportStatus::NoImportable;
    }

    return BatchCameraImportStatus::Ok;
}

} // namespace xjw::gui::project
