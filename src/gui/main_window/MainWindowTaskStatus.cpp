#include "MainWindow.h"

#include "ui_MainWindow.h"

#include <QApplication>
#include <QAction>
#include <QSplitter>
#include <QDockWidget>
#include <QToolBar>
#include <QMenuBar>
#include <QStatusBar>
#include <QFileInfo>
#include <QDesktopServices>
#include <QUrl>
#include <QJsonArray>
#include <QJsonObject>
#include <QMessageBox>
#include <QProgressDialog>
#include <QSaveFile>
#include <QTabWidget>
#include <QTextStream>
#include <QCloseEvent>
#include <QTimer>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QScopedValueRollback>
#include <QWidgetAction>

#include <algorithm>
#include <utility>

#include "CanvasWidget.h"
#include "ImageViewRotationSettings.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "LogPanel.h"
#include "MainMenu.h"
#include "MenuWorkflowController.h"
#include "ReconstructionWorkflowController.h"
#include "tie_points/CleanTiePointsDialog.h"
#include "tie_points/CreateTiePointsDialog.h"
#include "tie_points/MatchPairSelectorDialog.h"
#include "MatchPhotosTask.h"
#include "camera/ForwardIntersectionCheckDialog.h"
#include "camera/ForwardIntersectionResultsDialog.h"
#include "HenuBrandWidget.h"
#include "ProjectManager.h"
#include "project/ProjectIO.h"
#include "ProjectData.h"
#include "ProjectDashboardWidget.h"
#include "PhotoStripWidget.h"
#include "AppConfigManager.h"
#include "DataTreeWidget.h"
#include "ReferencePanelWidget.h"
#include "SelectionPropertiesWidget.h"
#include "TaskStatusWidget.h"
#include "ObservationNetworkView.h"
#include "graph/ObservationNetworkBuilder.h"
#include "WorkspaceCenterWidget.h"
#include "WorkspacePanelController.h"
#include "ProjectUiHydrator.h"
#include "TiePointWorkflowController.h"
#include "camera/CameraModel3DDialog.h"
#include "tie_points/ThinTiePointsDialog.h"
#include "LayerRenderer.h"
#include "Logger.h"
#include "ModelDropSupport.h"
#include "MarkerWorkspaceController.h"
#include "MarkerReferencePanel.h"
#include "MarkerFocusMeasurementDialog.h"
#include "DetectMarkersDialog.h"
#include "MarkerDetectionReviewDialog.h"
#include "PrintMarkersDialog.h"

void MainWindow::showMatchViewer(const QString &initialImagePath, bool modal)
{
    if (!_projectManager)
    {
        LOG_ERROR(QStringLiteral("无法打开匹配查看：ProjectManager 未初始化"));
        return;
    }

    auto *dialog = new MatchPairSelectorDialog(_projectManager, this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    if (!initialImagePath.trimmed().isEmpty())
    {
        dialog->setInitialImagePath(initialImagePath);
    }

    if (modal)
    {
        dialog->exec();
    }
    else
    {
        dialog->show();
    }
}

void MainWindow::refreshDashboardTaskSnapshots()
{
    if (!_dashboard)
    {
        return;
    }

    QJsonArray tasks;
    auto appendTask = [&tasks](const QString &name, const TaskStatusWidget *widget)
    {
        if (!widget || (!widget->isActive() && !widget->isCancelling()))
        {
            return;
        }

        QJsonObject record;
        record[QStringLiteral("name")] = name;
        record[QStringLiteral("status_text")] = widget->statusText();
        record[QStringLiteral("active")] = widget->isActive();
        record[QStringLiteral("cancelling")] = widget->isCancelling();
        record[QStringLiteral("progress_value")] = widget->progressValue();
        record[QStringLiteral("progress_maximum")] = widget->progressMaximum();
        tasks.append(record);
    };

    appendTask(tr("网格重建"), _meshTaskStatus);
    appendTask(tr("创建点云"), _pointCloudTaskStatus);
    appendTask(tr("空三/光束法平差"), _atTaskStatus);
    appendTask(tr("特征匹配"), _sgTaskStatus);
    appendTask(tr("生成蒙版"), _maskTaskStatus);

    _dashboard->setTaskSnapshots(tasks);
}

bool MainWindow::exportMatchedPairsToLis(QString *outputPath, QString *errorMessage) const
{
    if (outputPath)
    {
        outputPath->clear();
    }
    if (errorMessage)
    {
        errorMessage->clear();
    }

    if (!_projectManager)
    {
        if (errorMessage)
        {
            *errorMessage = tr("项目管理器未初始化");
        }
        return false;
    }

    const QString plascanPath = _projectManager->currentProjectPath();
    if (plascanPath.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("请先打开项目");
        }
        return false;
    }

    const QVector<QPair<QString, QString>> matchedPairs =
        xjw::common::project::collectMatchedImageNamePairs(plascanPath, _projectManager->currentMeta());
    if (matchedPairs.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("当前项目中没有可导出的匹配对");
        }
        return false;
    }

    const QString projectRoot = xjw::common::project::ProjectIO::projectRootFromPlascan(plascanPath);
    const QString exportDirPath = QDir(projectRoot).filePath(QStringLiteral("export"));
    QDir exportDir;
    if (!exportDir.mkpath(exportDirPath))
    {
        if (errorMessage)
        {
            *errorMessage = tr("无法创建导出目录: %1").arg(exportDirPath);
        }
        return false;
    }

    const QString lisPath = QDir(exportDirPath).filePath(QStringLiteral("matched_pairs.lis"));
    QSaveFile outputFile(lisPath);
    if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        if (errorMessage)
        {
            *errorMessage = tr("无法写入导出文件: %1").arg(lisPath);
        }
        return false;
    }

    QTextStream out(&outputFile);
    for (const QPair<QString, QString> &matchedPair : matchedPairs)
    {
        out << matchedPair.first << ' ' << matchedPair.second << '\n';
    }

    if (!outputFile.commit())
    {
        if (errorMessage)
        {
            *errorMessage = tr("保存导出文件失败: %1").arg(lisPath);
        }
        return false;
    }

    if (outputPath)
    {
        *outputPath = lisPath;
    }
    return true;
}

void MainWindow::onExportMatchedPairs()
{
    QString outputPath;
    QString errorMessage;
    if (!exportMatchedPairsToLis(&outputPath, &errorMessage))
    {
        QMessageBox::warning(this, tr("导出匹配对"), errorMessage);
        return;
    }

    statusBar()->showMessage(tr("匹配对已导出到: %1").arg(outputPath), 5000);
    QMessageBox::information(this,
                             tr("导出匹配对"),
                             tr("导出成功，文件已保存到：\n%1").arg(outputPath));
}

void MainWindow::onManualPointCloudPrune()
{
    if (!_workspaceCenter || !_workspaceCenter->modelView())
    {
        return;
    }

    auto *modelView = _workspaceCenter->modelView();
    const bool enable = !modelView->isManualPruneModeEnabled();
    QString errorMessage;
    if (!modelView->setManualPruneModeEnabled(enable, &errorMessage))
    {
        QMessageBox::warning(this, tr("手动点云剔除"), errorMessage);
        return;
    }

    if (enable)
    {
        _workspaceCenter->showModelView();
        QMessageBox::information(this,
                                 tr("手动点云剔除"),
                                 tr("已进入手动剔除模式。\n"
                                                "请在点云视图中按住鼠标右键拖拽框选，高亮目标点。\n"
                                                "按鼠标前进侧键执行删除并覆盖保存。\n"
                                                "误删可按 Ctrl+Z 撤销。\n"
                                    "再次点击 工具-手动点云剔除 可退出。"));
    }
    else
    {
        statusBar()->showMessage(tr("已退出手动点云剔除模式"), 3500);
    }
}


// ============================================================
//  网格重建进度状态栏槽
// ============================================================

void MainWindow::onMeshProgress(const QString &stage, int percent)
{
    if (!_meshTaskStatus)
    {
        return;
    }
    if (!_meshTaskStatus->isActive())
    {
        _meshTaskStatus->begin(stage, 0, 100);
    }
    _meshTaskStatus->updateProgress(stage, percent);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::onMeshFinished(bool success)
{
    if (!_meshTaskStatus)
    {
        return;
    }
    _meshTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(
        success ? tr("网格重建完成") : tr("网格重建失败"), 4000);
}

// ============================================================
//  深度图估计与点云融合进度状态栏槽
// ============================================================

void MainWindow::onPointCloudProgress(const QString &stage, int percent)
{
    if (!_pointCloudTaskStatus)
    {
        return;
    }
    const int clamped_percent = std::clamp(percent, 0, 100);
    const QString status_text = clamped_percent > 0 && clamped_percent < 100
        ? QStringLiteral("%1 %2%").arg(stage).arg(clamped_percent)
        : stage;
    if (!_pointCloudTaskStatus->isActive())
    {
        _pointCloudTaskStatus->begin(status_text, 0, 100);
    }
    _pointCloudTaskStatus->updateProgress(status_text, clamped_percent);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::onPointCloudFinished(bool success)
{
    if (!_pointCloudTaskStatus)
    {
        return;
    }
    _pointCloudTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(
        success ? tr("点云创建完成") : tr("点云创建已取消或失败"), 4000);
}

// ============================================================
//  AT（空三）/光束法平差进度状态栏槽
// ============================================================

void MainWindow::onAtProgress(const QString &stage, int percent)
{
    if (!_atTaskStatus)
    {
        return;
    }
    // 显示阶段信息和百分比，例如 "正在匹配特征点... 50%"
    QString statusText = stage;
    if (percent > 0 && percent < 100)
    {
        statusText = QStringLiteral("%1 %2%").arg(stage).arg(percent);
    }
    if (!_atTaskStatus->isActive())
    {
        _atTaskStatus->begin(statusText, 0, 100);
    }
    _atTaskStatus->updateProgress(statusText, percent);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());   // 清空普通消息，让 permanent widget 露出
}

void MainWindow::onAtFinished(bool success)
{
    if (!_atTaskStatus)
    {
        return;
    }
    _atTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(
        success ? tr("空三/光束法平差完成") : tr("空三/光束法平差已取消或失败"), 4000);
}

void MainWindow::onCancelAt()
{
    if (_projectManager)
    {
        _projectManager->cancelAt();
    }
}

// ============================================================
//  照片蒙版生成进度状态栏槽
// ============================================================

void MainWindow::onMaskGenerationProgress(const QString &stage, int done, int total)
{
    if (!_maskTaskStatus)
    {
        return;
    }

    const int safeTotal = std::max(1, total);
    const int clampedDone = std::clamp(done, 0, safeTotal);
    const QString defaultText = tr("生成蒙版 %1/%2").arg(clampedDone).arg(safeTotal);
    const QString stageText = stage.trimmed();
    const QString statusText = stageText.isEmpty() || stageText == tr("生成蒙版")
        ? defaultText
        : tr("%1 %2/%3").arg(stageText).arg(clampedDone).arg(safeTotal);

    if (!_maskTaskStatus->isActive())
    {
        _maskTaskStatus->begin(statusText, 0, safeTotal);
    }
    _maskTaskStatus->updateProgress(statusText, clampedDone);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::onMaskGenerationFinished(bool success)
{
    if (!_maskTaskStatus)
    {
        return;
    }
    _maskTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(success ? tr("蒙版生成完成") : tr("蒙版生成已取消或失败"), 4000);
}

void MainWindow::showSgProgress(int total)
{
    if (!_sgTaskStatus)
    {
        return;
    }
    _sgTaskStatus->begin(tr("特征匹配 0/%1").arg(total), 0, total);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::updateSgProgress(int done)
{
    if (!_sgTaskStatus)
    {
        return;
    }
    const int total = _sgTaskStatus->progressMaximum();
    const int clampedDone = std::clamp(done, 0, total);
    _sgTaskStatus->updateProgress(
        tr("特征匹配 %1/%2").arg(clampedDone).arg(total), clampedDone);
    refreshDashboardTaskSnapshots();
}

void MainWindow::hideSgProgress(bool ok)
{
    if (!_sgTaskStatus)
    {
        return;
    }
    _sgTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(ok ? tr("匹配完成") : tr("匹配已取消"), 4000);
}

// ============================================================
//  辅助方法
// ============================================================
