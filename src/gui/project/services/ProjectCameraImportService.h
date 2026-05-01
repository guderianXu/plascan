#pragma once

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

namespace xjw::gui::project {

// 单张相机导入状态：用于在 UI 层区分路径错误与解析失败。
enum class SingleCameraImportStatus {
    Ok,
    EmptyImagePath,
    ParseFailed
};

// 单张相机导入结果：包含归一化后的影像路径、相机 JSON 以及错误信息。
struct SingleCameraImportResult {
    QString imageAbsPath;
    QJsonObject cameraMeta;
    QString error;
};

// 构建单张导入数据（不落盘）：ProjectManager 负责后续写回 ProjectData。
SingleCameraImportStatus buildSingleCameraImport(const QString &imagePath,
                                                 const QString &tsaiPath,
                                                 SingleCameraImportResult *out);

// 批量导入状态：区分“文件夹无 tsai”“项目无影像”“存在 tsai 但均不可导入”。
enum class BatchCameraImportStatus {
    Ok,
    NoTsaiFiles,
    NoProjectImages,
    NoImportable
};

// 批量导入结果：包含待写回映射和统计信息。
struct BatchCameraImportResult {
    // 最终可写回的影像->相机元数据映射。
    QMap<QString, QJsonObject> cameraMetaByImage;
    // 多个影像命中同一 baseName 时的歧义数量。
    int ambiguousCount = 0;
    // 解析失败数量及错误列表（供日志输出）。
    int parseFailedCount = 0;
    int unmatchedCount = 0;
    QStringList parseErrors;
};

// 构建批量导入数据：按 baseName 匹配 tsai 与项目影像，并解析为统一相机 JSON。
BatchCameraImportStatus buildBatchCameraImport(const QString &tsaiFolder,
                                               const QStringList &projectImages,
                                               BatchCameraImportResult *out);

} // namespace xjw::gui::project
