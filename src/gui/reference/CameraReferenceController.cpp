#include "CameraReferenceController.h"

#include "CameraReferenceCsvExporter.h"
#include "MetashapeCameraReferenceImporter.h"
#include "MetashapeCameraReferenceSetBuilder.h"
#include "project/ProjectSessionModel.h"
#include "ProjectCameraReferenceRepository.h"
#include "model/CameraReferenceSet.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>

#include <optional>

namespace xjw::gui::reference
{
namespace
{

std::optional<QString> sourceContentHash(const QString &cameraPath,
                                         const QString &offsetPath,
                                         QString *error)
{
    QCryptographicHash hash(QCryptographicHash::Sha256);
    const QList<QPair<QString, QString>> sources{
        {QStringLiteral("camera"), cameraPath},
        {QStringLiteral("gnss_offset"), offsetPath}
    };
    for (const auto &[kind, path] : sources)
    {
        if (path.isEmpty())
        {
            continue;
        }
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly))
        {
            if (error)
            {
                *error = QStringLiteral("无法读取相机参考源文件 %1：%2")
                    .arg(path, file.errorString());
            }
            return std::nullopt;
        }
        const QByteArray content = file.readAll();
        const QByteArray label = kind.toUtf8() + ':'
            + QFileInfo(path).fileName().toUtf8();
        hash.addData(QByteArray::number(label.size()));
        hash.addData(QByteArrayLiteral(":"));
        hash.addData(label);
        hash.addData(QByteArray::number(content.size()));
        hash.addData(QByteArrayLiteral(":"));
        hash.addData(content);
    }
    return QString::fromLatin1(hash.result().toHex());
}

QString siblingOffsetPath(const QString &cameraPath)
{
    const QDir directory = QFileInfo(cameraPath).absoluteDir();
    const QStringList files = directory.entryList(QDir::Files | QDir::Readable);
    for (const QString &file : files)
    {
        if (file.compare(QStringLiteral("GNSS_offset.txt"), Qt::CaseInsensitive) == 0)
        {
            return directory.filePath(file);
        }
    }
    return {};
}

} // namespace

CameraReferenceController::CameraReferenceController(
    ProjectData *projectData,
    ProjectCameraReferenceRepository *repository,
    QWidget *parentWidget,
    QObject *parent)
    : QObject(parent)
    , _projectData(projectData)
    , _repository(repository)
    , _parentWidget(parentWidget)
{
}

void CameraReferenceController::importMetashapeReference()
{
    if (!_projectData || !_projectData->hasProject() || !_repository)
    {
        QMessageBox::information(_parentWidget,
                                 tr("导入相机参考"),
                                 tr("请先创建或打开项目。"));
        return;
    }
    const QString cameraPath = QFileDialog::getOpenFileName(
        _parentWidget,
        tr("导入 Metashape 相机参考"),
        initialDirectory(),
        tr("Metashape 相机参考 (Cameras*.txt *.txt);;所有文件 (*)"));
    if (cameraPath.isEmpty())
    {
        return;
    }
    const QString offsetPath = siblingOffsetPath(cameraPath);
    reference_import::MetashapeCameraReferenceImportResult imported;
    QString error;
    if (!reference_import::importMetashapeCameraReferenceTxt(
            cameraPath, offsetPath, &imported, &error))
    {
        QMessageBox::critical(_parentWidget, tr("导入相机参考"), error);
        return;
    }
    const std::optional<QString> contentHash = sourceContentHash(
        cameraPath, offsetPath, &error);
    if (!contentHash)
    {
        QMessageBox::critical(_parentWidget, tr("导入相机参考"), error);
        return;
    }

    if (!_repository->referenceSet().records().isEmpty()
        || !_repository->referenceSet().unmatchedRecords().isEmpty())
    {
        const QMessageBox::StandardButton choice = QMessageBox::question(
            _parentWidget,
            tr("替换相机参考"),
            tr("当前项目已有相机参考。是否用新文件完整替换？"),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No);
        if (choice != QMessageBox::Yes)
        {
            return;
        }
    }

    camera_reference::CameraReferenceSet referenceSet;
    try
    {
        referenceSet = buildMetashapeCameraReferenceSet(
            imported, _projectData->metadata(), cameraPath, offsetPath, *contentHash);
    }
    catch (const std::exception &exception)
    {
        QMessageBox::critical(_parentWidget,
                              tr("导入相机参考"),
                              QString::fromUtf8(exception.what()));
        return;
    }
    if (!_repository->replaceReferenceSet(referenceSet, &error))
    {
        QMessageBox::critical(_parentWidget, tr("导入相机参考"), error);
        return;
    }

    QString message = tr("已导入 %1 条源记录，其中 %2 条绑定到项目影像，%3 条未匹配。")
                          .arg(imported.records.size())
                          .arg(referenceSet.records().size())
                          .arg(referenceSet.unmatchedRecords().size());
    if (!offsetPath.isEmpty())
    {
        message += tr("\n已保留同目录 GNSS_offset.txt 的杆臂值。");
    }
    message += tr("\n\nWGS84、姿态角和杆臂目前按源值保存；完成坐标系及姿态约定配置前，"
                  "不会作为平差约束使用。");
    if (!imported.warnings.isEmpty())
    {
        message += tr("\n\n警告：\n%1").arg(imported.warnings.join(QLatin1Char('\n')));
    }
    QMessageBox::information(_parentWidget, tr("导入相机参考"), message);
}

void CameraReferenceController::exportReferences()
{
    if (!_repository
        || (_repository->referenceSet().records().isEmpty()
            && _repository->referenceSet().unmatchedRecords().isEmpty()))
    {
        return;
    }
    const QString path = QFileDialog::getSaveFileName(
        _parentWidget,
        tr("导出相机参考源值"),
        QDir(initialDirectory()).filePath(QStringLiteral("camera_references.csv")),
        tr("CSV 文件 (*.csv)"));
    if (path.isEmpty())
    {
        return;
    }

    QString error;
    if (!exportCameraReferenceCsv(_repository->referenceSet(), path, &error))
    {
        QMessageBox::critical(_parentWidget, tr("导出相机参考"), error);
        return;
    }
    QMessageBox::information(
        _parentWidget,
        tr("导出相机参考"),
        tr("已导出 %1 条已匹配和 %2 条未匹配相机参考源值。")
            .arg(_repository->referenceSet().records().size())
            .arg(_repository->referenceSet().unmatchedRecords().size()));
}

void CameraReferenceController::showSettingsSummary()
{
    if (!_repository)
    {
        return;
    }
    const camera_reference::CameraReferenceSet &set = _repository->referenceSet();
    int positionUsable = 0;
    int orientationUsable = 0;
    for (const camera_reference::CameraReferenceRecord &record : set.records())
    {
        positionUsable += record.resolved.positionUsable ? 1 : 0;
        orientationUsable += record.resolved.orientationUsable ? 1 : 0;
    }
    const QString source = set.source().displayName.isEmpty()
        ? tr("未导入")
        : set.source().displayName;
    QMessageBox::information(
        _parentWidget,
        tr("相机参考设置"),
        tr("来源：%1\n源坐标系：%2\n坐标轴：%3\n垂直基准：%4\n姿态约定：%5\n"
           "记录：%6（未匹配 %7）\n可用于求解：位置 %8，姿态 %9\n杆臂记录：%10\n\n"
           "当前版本保留并检查源观测，但尚未提供求解坐标系、姿态轴向和杆臂方向的转换配置；"
           "因此未转换的 WGS84/姿态数据不会静默进入平差。")
            .arg(source,
                 set.source().sourceCrs,
                 set.source().axisOrder,
                 set.source().verticalDatum,
                 set.source().orientationConvention)
            .arg(set.records().size())
            .arg(set.unmatchedRecords().size())
            .arg(positionUsable)
            .arg(orientationUsable)
            .arg(set.leverArms().size()));
}

QString CameraReferenceController::initialDirectory() const
{
    if (!_projectData || _projectData->currentProjectPath().isEmpty())
    {
        return QDir::homePath();
    }
    return QFileInfo(_projectData->currentProjectPath()).absolutePath();
}

} // namespace xjw::gui::reference
