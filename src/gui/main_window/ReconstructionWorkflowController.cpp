#include "ReconstructionWorkflowController.h"

#include "reconstruction/GenerateModelDialog.h"
#include "ProjectManager.h"
#include "reconstruction/TextureMappingDialog.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMainWindow>

ReconstructionWorkflowController::ReconstructionWorkflowController(
    QMainWindow *mainWindow,
    QObject *parent)
    : QObject(parent)
    , _mainWindow(mainWindow)
{
}

void ReconstructionWorkflowController::setProjectManager(ProjectManager *projectManager)
{
    _projectManager = projectManager;
}

void ReconstructionWorkflowController::markProjectWorkspaceDirty()
{
    if (_projectManager)
    {
        _projectManager->markWorkspaceDirty();
    }
}

QString ReconstructionWorkflowController::projectPath() const
{
    return _projectManager ? _projectManager->currentProjectPath() : QString();
}

namespace
{

QString existingCleanPath(const QString &path)
{
    const QString cleanPath = QDir::cleanPath(path.trimmed());
    if (cleanPath.isEmpty() || !QFileInfo::exists(cleanPath))
    {
        return QString();
    }
    return cleanPath;
}

QString sparseCloudPathFromAtRecord(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("files"))
                       .toObject()
                       .value(QStringLiteral("sparse_cloud_xyz"))
                       .toString();
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("sparse_cloud_xyz")).toString();
    }
    return existingCleanPath(path);
}

QString modelPathFromRecord(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("final_model_path")).toString();
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("model_ply")).toString();
    }
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("mesh_ply")).toString();
    }
    return existingCleanPath(path);
}

void appendModelSourceCandidate(QJsonArray *candidates,
                                const QString &sourceData,
                                const QString &sourceLabel,
                                const QString &path,
                                const QString &displayPrefix,
                                bool supported,
                                const QString &note = QString())
{
    if (!candidates)
    {
        return;
    }

    const QString cleanPath = supported
        ? existingCleanPath(path)
        : QDir::cleanPath(path.trimmed());
    if (cleanPath.isEmpty())
    {
        return;
    }

    QJsonObject candidate;
    candidate[QStringLiteral("source_data")] = sourceData;
    candidate[QStringLiteral("source_label")] = sourceLabel;
    candidate[QStringLiteral("source_path")] = cleanPath;
    candidate[QStringLiteral("display")] =
        QStringLiteral("%1 - %2").arg(displayPrefix, QFileInfo(cleanPath).fileName());
    candidate[QStringLiteral("supported")] = supported;
    if (!note.isEmpty())
    {
        candidate[QStringLiteral("note")] = note;
    }
    candidates->append(candidate);
}

QJsonArray buildGenerateModelSourceCandidates(const QJsonObject &metadata)
{
    QJsonArray candidates;
    QStringList seenPaths;

    const QJsonArray depthResults =
        metadata.value(QStringLiteral("depth_map_results")).toArray();
    QStringList seenDepthDirs;
    for (int index = depthResults.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = depthResults.at(index).toObject();
        QString depthPath = record.value(QStringLiteral("mvs_output_dir")).toString();
        if (depthPath.isEmpty())
        {
            depthPath = record.value(QStringLiteral("raw_depth_path")).toString();
        }
        if (depthPath.isEmpty())
        {
            depthPath = record.value(QStringLiteral("depth_png")).toString();
        }

        const QString cleanPath = existingCleanPath(depthPath);
        const QString depthKey = QFileInfo(cleanPath).isDir()
            ? cleanPath
            : QFileInfo(cleanPath).absolutePath();
        if (cleanPath.isEmpty() || seenDepthDirs.contains(depthKey))
        {
            continue;
        }

        seenDepthDirs.push_back(depthKey);
        appendModelSourceCandidate(
            &candidates,
            QStringLiteral("depth_maps"),
            QStringLiteral("深度图"),
            cleanPath,
            QStringLiteral("深度图"),
            true,
            QStringLiteral("深度图将作为生成模型入口；若该深度图目录已有融合点云，将直接复用并生成网格。"));
    }

    const QJsonArray denseResults =
        metadata.value(QStringLiteral("dense_cloud_results")).toArray();
    for (int index = denseResults.size() - 1; index >= 0; --index)
    {
        const QString path = existingCleanPath(
            denseResults.at(index)
                .toObject()
                .value(QStringLiteral("dense_cloud_xyz"))
                .toString());
        if (path.isEmpty() || seenPaths.contains(path))
        {
            continue;
        }

        seenPaths.push_back(path);
        appendModelSourceCandidate(
            &candidates,
            QStringLiteral("point_cloud"),
            QStringLiteral("点云"),
            path,
            QStringLiteral("点云"),
            true,
            QStringLiteral("将使用当前网格重建管线从点云生成模型。"));
    }

    const QJsonArray sparseResults =
        metadata.value(QStringLiteral("aerial_triangulation_results")).toArray();
    for (int index = sparseResults.size() - 1; index >= 0; --index)
    {
        const QString path = sparseCloudPathFromAtRecord(sparseResults.at(index).toObject());
        if (path.isEmpty() || seenPaths.contains(path))
        {
            continue;
        }

        seenPaths.push_back(path);
        appendModelSourceCandidate(
            &candidates,
            QStringLiteral("tie_points"),
            QStringLiteral("连接点"),
            path,
            QStringLiteral("连接点"),
            true,
            QStringLiteral("连接点生成的是快速预览级模型，细节质量低于点云或深度图。"));
    }

    const QJsonArray modelResults =
        metadata.value(QStringLiteral("model_results")).toArray();
    for (int index = modelResults.size() - 1; index >= 0; --index)
    {
        const QString path = modelPathFromRecord(modelResults.at(index).toObject());
        if (path.isEmpty() || seenPaths.contains(path))
        {
            continue;
        }

        seenPaths.push_back(path);
        appendModelSourceCandidate(
            &candidates,
            QStringLiteral("model"),
            QStringLiteral("模型"),
            path,
            QStringLiteral("模型"),
            true,
            QStringLiteral("将从已有模型顶点重新生成模型，适合快速重建或换参数预览。"));
    }

    return candidates;
}

} // namespace

void ReconstructionWorkflowController::openGenerateModelDialog()
{
    auto *dialog = prepareDialog<GenerateModelDialog>(
        DialogSettingKeys::GenerateModel,
        _generateModelStore);
    if (!dialog)
    {
        return;
    }

    if (_projectManager)
    {
        dialog->setSourceCandidates(
            buildGenerateModelSourceCandidates(_projectManager->currentMeta()));
    }

    connect(
        dialog,
        &GenerateModelDialog::runRequested,
        this,
        [this](const QJsonObject &settings)
        {
            LOG_INFO(QStringLiteral("生成模型: %1")
                         .arg(QString::fromUtf8(
                             QJsonDocument(settings).toJson(QJsonDocument::Compact))));
            if (_projectManager)
            {
                _projectManager->startGenerateModelAsync(settings);
            }
        },
        Qt::QueuedConnection);

    dialog->exec();
}

void ReconstructionWorkflowController::openTextureMappingDialog()
{
    auto *dialog = prepareDialog<TextureMappingDialog>(
        DialogSettingKeys::TextureMapping,
        _texStore);
    if (!dialog)
    {
        return;
    }

    connect(
        dialog,
        &TextureMappingDialog::runRequested,
        this,
        [this](const QJsonObject &settings)
        {
            LOG_INFO(QStringLiteral("纹理映射: %1")
                         .arg(QString::fromUtf8(
                             QJsonDocument(settings).toJson(QJsonDocument::Compact))));
            if (_projectManager)
            {
                _projectManager->startTextureMappingAsync(settings);
            }
        });

    dialog->exec();
}
