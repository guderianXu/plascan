#include "ProjectTerrainProductsManager.h"

#include "ProjectManager.h"
#include "ProjectData.h"
#include "ProjectIO.h"
#include "ProjectMetadataOperations.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowUtils.h"
#include "ProjectCameraImportService.h"
#include "SuperPointRunner.h"
#include "FeatureMatchRunner.h"
#include "Logger.h"
#include "Camera.h"
#include "TerrainPipeline.h"

#include <QDir>
#include <QJsonArray>
#include <QMessageBox>
#include <QtConcurrent>
#include <QFutureWatcher>
#include <QTimer>
#include <opencv2/core.hpp>

using xjw::gui::project::makeDemResultRecord;
using xjw::gui::project::makeDepthResultRecord;
using xjw::gui::project::makeOrthoResultRecord;
using xjw::gui::project::persistProjectMeta;
using xjw::gui::project::findLatestProductionAtResultIndex;
using xjw::gui::project::resolveProjectOutputDir;
using xjw::gui::project::resolveLatestDenseCloudPath;
using xjw::gui::project::runDemProducts;
using xjw::gui::project::runOrthoProduct;
using xjw::gui::project::upsertMetaArrayRecordByPath;

ProjectTerrainProductsManager::ProjectTerrainProductsManager(ProjectManager *owner,
                                                             ProjectData *projectData,
                                                             QWidget *parentWidget,
                                                             QObject *parent)
    : QObject(parent)
    , m_owner(owner)
    , m_projectData(projectData)
    , m_parentWidget(parentWidget)
{
}

bool ProjectTerrainProductsManager::ensureProjectOpen(const QString &message,
                                                      const QString &title) const
{
    if (m_projectData && m_projectData->hasProject())
    {
        return true;
    }
    QMessageBox::warning(m_parentWidget, title, message);
    return false;
}

bool ProjectTerrainProductsManager::runDemProductsOrWarn(const QString &sparsePath,
                                                         const QString &outputDir,
                                                         double demResolution,
                                                         const QString &demType,
                                                         bool genPointCloud,
                                                         const QString &title,
                                                         QJsonObject *result) const
{
    const auto terrainRun = runDemProducts(sparsePath,
                                           outputDir,
                                           demResolution,
                                           demType,
                                           genPointCloud);
    if (!terrainRun.ok)
    {
        QMessageBox::warning(m_parentWidget,
                             title,
                             QStringLiteral("处理失败：%1").arg(terrainRun.error));
        return false;
    }
    if (result)
    {
        *result = terrainRun.payload;
    }
    return true;
}

bool ProjectTerrainProductsManager::runOrthoProductOrWarn(const QStringList &images,
                                                          const QString &demPath,
                                                          const QString &outputPath,
                                                          double resolution,
                                                          const QJsonObject &projectMeta,
                                                          const QString &title,
                                                          QJsonObject *result) const
{
    const auto orthoRun = runOrthoProduct(images, demPath, outputPath, resolution, projectMeta);
    if (!orthoRun.ok)
    {
        QMessageBox::warning(m_parentWidget,
                             title,
                             QStringLiteral("处理失败：%1").arg(orthoRun.error));
        return false;
    }
    if (result)
    {
        *result = orthoRun.payload;
    }
    return true;
}

void ProjectTerrainProductsManager::startStereoAndPoint2DemAsync(const QStringList &images,
                                                                 const QString &outputDir,
                                                                 int threads,
                                                                 bool genPointCloud,
                                                                 double demResolution,
                                                                 const QString &demType,
                                                                 const QString &tSrs)
{
    Q_UNUSED(threads);
    Q_UNUSED(genPointCloud);
    Q_UNUSED(tSrs);
    Q_UNUSED(images);

    if (!ensureProjectOpen())
    {
        return;
    }

    QJsonObject meta = m_projectData->metadata();
    QString denseCloudPath;
    QString denseError;
    if (!resolveLatestDenseCloudPath(m_projectData, &denseCloudPath, &denseError))
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("创建相对 DEM"),
                             QStringLiteral("%1\n请先完成[工作流程 → 创建点云]。")
                                 .arg(denseError.isEmpty()
                                          ? QStringLiteral("未找到可用的密集点云结果")
                                          : denseError));
        return;
    }

    QString outDir = resolveProjectOutputDir(m_owner->currentProjectPath(),
                                             outputDir.trimmed(),
                                             QStringLiteral("assets/dem/relative_dem"));

    QJsonObject terrainResult;
    if (!runDemProductsOrWarn(denseCloudPath,
                              outDir,
                              demResolution,
                              demType,
                              false,
                              QStringLiteral("创建相对 DEM"),
                              &terrainResult))
    {
        return;
    }

    QJsonObject depthResult = makeDepthResultRecord(
        terrainResult.value(QStringLiteral("created_at")).toString(),
        terrainResult.value(QStringLiteral("depth_png")).toString(),
        terrainResult.value(QStringLiteral("grid_width")).toInt(),
        terrainResult.value(QStringLiteral("grid_height")).toInt(),
        denseCloudPath);
    upsertMetaArrayRecordByPath(&meta,
                                QStringLiteral("depth_map_results"),
                                QStringLiteral("depth_png"),
                                depthResult);

    QJsonObject demResult = makeDemResultRecord(
        terrainResult.value(QStringLiteral("created_at")).toString(),
        outDir,
        QString(),
        terrainResult.value(QStringLiteral("dem_tif")).toString(),
        demType,
        demResolution,
        QString(),
        images);
    demResult[QStringLiteral("source_dense_cloud")] = denseCloudPath;
    demResult[QStringLiteral("dem_reference")] = QStringLiteral("relative");
    demResult[QStringLiteral("depth_preview_png")] = terrainResult.value(QStringLiteral("depth_png")).toString();
    demResult[QStringLiteral("relative_z_offset")] = terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0);
    upsertMetaArrayRecordByPath(&meta,
                                QStringLiteral("dem_results"),
                                QStringLiteral("dem_tif"),
                                demResult);

    persistProjectMeta(m_projectData, meta, true);

    QMessageBox::information(m_parentWidget,
                             QStringLiteral("创建相对 DEM"),
                             QStringLiteral("处理完成。\nDEM: %1\n预览图: %2\n参考点云: %3\n高程基准偏移: %4")
                                 .arg(terrainResult.value(QStringLiteral("dem_tif")).toString())
                                 .arg(terrainResult.value(QStringLiteral("depth_png")).toString())
                                 .arg(denseCloudPath)
                                 .arg(terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0), 0, 'f', 6));
}

void ProjectTerrainProductsManager::startFullDemPipelineAsync(const QStringList &images,
                                                             const QString &outputDir,
                                                             const QJsonObject &pipelineSettings)
{
    using namespace xjw::gui::project;

    if (!ensureProjectOpen())
    {
        return;
    }

    if (images.size() < 2)
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("创建相对 DEM"),
                             QStringLiteral("至少需要 2 张影像才能生成立体 DEM。"));
        return;
    }

    // 步骤 1: 注册相机文件（如果提供）
    const QJsonArray camFilesArr = pipelineSettings.value(QStringLiteral("camera_files")).toArray();
    if (!camFilesArr.isEmpty())
    {
        QMap<QString, QJsonObject> cameraMetaByImage;
        QStringList importErrors;

        for (int i = 0; i < qMin(images.size(), camFilesArr.size()); ++i)
        {
            const QString imagePath = images[i];
            const QString tsaiPath = camFilesArr.at(i).toString();

            SingleCameraImportResult importResult;
            const SingleCameraImportStatus status = buildSingleCameraImport(imagePath, tsaiPath, &importResult);

            if (status == SingleCameraImportStatus::Ok)
            {
                cameraMetaByImage[importResult.imageAbsPath] = importResult.cameraMeta;
            }
            else
            {
                importErrors << QStringLiteral("%1: %2").arg(QFileInfo(imagePath).fileName(), importResult.error);
            }
        }

        if (!importErrors.isEmpty())
        {
            QMessageBox::warning(m_parentWidget,
                                 QStringLiteral("创建相对 DEM"),
                                 QStringLiteral("部分相机文件导入失败：\n%1").arg(importErrors.join(QStringLiteral("\n"))));
            return;
        }

        if (!cameraMetaByImage.isEmpty())
        {
            int updatedCount = 0;
            QString writeErr;
            if (!m_projectData->setImageCameras(cameraMetaByImage, &updatedCount, &writeErr))
            {
                QMessageBox::warning(m_parentWidget,
                                     QStringLiteral("创建相对 DEM"),
                                     QStringLiteral("相机参数写入项目失败: %1").arg(writeErr));
                return;
            }
        }
    }

    // 步骤 2-6: 启动异步流水线任务
    DemPipelineContext ctx;
    ctx.images = images;
    for (int i = 0; i < camFilesArr.size(); ++i)
        ctx.cameraPaths << camFilesArr.at(i).toString();
    ctx.outputDir = outputDir;
    ctx.featureAlgorithm = pipelineSettings.value(QStringLiteral("matcher")).toString(QStringLiteral("disk_lightglue"));
    ctx.matchAlgorithm = QStringLiteral("lightglue");
    ctx.demResolution = pipelineSettings.value(QStringLiteral("dem_resolution")).toDouble(0.0);
    ctx.demType = pipelineSettings.value(QStringLiteral("dem_type")).toString(QStringLiteral("float32"));

    // 从 matcher 字段解析特征算法
    if (ctx.featureAlgorithm.contains(QStringLiteral("disk"), Qt::CaseInsensitive))
    {
        ctx.featureAlgorithm = QStringLiteral("disk");
    }
    else
    {
        ctx.featureAlgorithm = QStringLiteral("superpoint");
    }

    // 启动后台流水线（步骤1-2同步，步骤3-5通过信号链在主线程驱动）
    // demPipelineFinished 由信号链末端（DEM完成或失败）发出，不在此处发出
    LOG_INFO(QStringLiteral("[DEM流水线] 启动后台任务（特征提取+匹配）..."));
    (void)QtConcurrent::run([this, ctx]()
    {
        runFullDemPipelineInBackground(ctx);
    });
}

void ProjectTerrainProductsManager::startDemFromDenseCloudAsync(const QString &denseCloudPath,
                                                               const QString &outputDir,
                                                               double demResolution,
                                                               const QString &demType)
{
    if (!ensureProjectOpen())
    {
        return;
    }

    QJsonObject meta = m_projectData->metadata();
    QString resolvedDenseCloud = denseCloudPath.trimmed();

    // 如果未指定密集点云路径，自动查找最新的
    if (resolvedDenseCloud.isEmpty())
    {
        QString denseError;
        if (!resolveLatestDenseCloudPath(m_projectData, &resolvedDenseCloud, &denseError))
        {
            QMessageBox::warning(m_parentWidget,
                                 QStringLiteral("创建相对 DEM"),
                                 QStringLiteral("%1\n请先完成[工作流程 → 创建点云]。")
                                     .arg(denseError.isEmpty()
                                              ? QStringLiteral("未找到可用的密集点云结果")
                                              : denseError));
            return;
        }
    }
    else if (!QFileInfo::exists(resolvedDenseCloud))
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("创建相对 DEM"),
                             QStringLiteral("指定的密集点云文件不存在：\n%1").arg(resolvedDenseCloud));
        return;
    }

    QString outDir = resolveProjectOutputDir(m_owner->currentProjectPath(),
                                             outputDir.trimmed(),
                                             QStringLiteral("assets/dem/relative_dem"));

    QJsonObject terrainResult;
    if (!runDemProductsOrWarn(resolvedDenseCloud,
                              outDir,
                              demResolution,
                              demType,
                              false,
                              QStringLiteral("创建相对 DEM"),
                              &terrainResult))
    {
        return;
    }

    QJsonObject depthResult = makeDepthResultRecord(
        terrainResult.value(QStringLiteral("created_at")).toString(),
        terrainResult.value(QStringLiteral("depth_png")).toString(),
        terrainResult.value(QStringLiteral("grid_width")).toInt(),
        terrainResult.value(QStringLiteral("grid_height")).toInt(),
        resolvedDenseCloud);
    upsertMetaArrayRecordByPath(&meta,
                                QStringLiteral("depth_map_results"),
                                QStringLiteral("depth_png"),
                                depthResult);

    QJsonObject demResult = makeDemResultRecord(
        terrainResult.value(QStringLiteral("created_at")).toString(),
        outDir,
        QString(),
        terrainResult.value(QStringLiteral("dem_tif")).toString(),
        demType,
        demResolution,
        QString(),
        QStringList());
    demResult[QStringLiteral("source_dense_cloud")] = resolvedDenseCloud;
    demResult[QStringLiteral("dem_reference")] = QStringLiteral("relative");
    demResult[QStringLiteral("depth_preview_png")] = terrainResult.value(QStringLiteral("depth_png")).toString();
    demResult[QStringLiteral("relative_z_offset")] = terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0);
    upsertMetaArrayRecordByPath(&meta,
                                QStringLiteral("dem_results"),
                                QStringLiteral("dem_tif"),
                                demResult);

    persistProjectMeta(m_projectData, meta, true);

    QMessageBox::information(m_parentWidget,
                             QStringLiteral("创建相对 DEM"),
                             QStringLiteral("处理完成。\nDEM: %1\n预览图: %2\n参考点云: %3\n高程基准偏移: %4")
                                 .arg(terrainResult.value(QStringLiteral("dem_tif")).toString())
                                 .arg(terrainResult.value(QStringLiteral("depth_png")).toString())
                                 .arg(resolvedDenseCloud)
                                 .arg(terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0), 0, 'f', 6));
}

void ProjectTerrainProductsManager::startMapProjectAsync(const QStringList &images,
                                                         const QString &demPath,
                                                         const QString &outputPath,
                                                         double resolution)
{
    if (!ensureProjectOpen())
    {
        return;
    }

    QJsonObject meta = m_projectData->metadata();
    QString resolvedDem = demPath.trimmed();
    QJsonObject matchedDemRecord;
    if (resolvedDem.isEmpty())
    {
        const QJsonArray demArr = meta.value(QStringLiteral("dem_results")).toArray();
        if (!demArr.isEmpty())
        {
            for (int index = demArr.size() - 1; index >= 0; --index)
            {
                const QJsonObject record = demArr.at(index).toObject();
                const QString candidate = record.value(QStringLiteral("dem_tif")).toString();
                if (!candidate.isEmpty() && QFileInfo::exists(candidate))
                {
                    resolvedDem = candidate;
                    matchedDemRecord = record;
                    if (record.value(QStringLiteral("dem_reference")).toString() == QStringLiteral("relative"))
                    {
                        break;
                    }
                }
            }
        }
    }
    if (matchedDemRecord.isEmpty())
    {
        const QJsonArray demArr = meta.value(QStringLiteral("dem_results")).toArray();
        for (int index = demArr.size() - 1; index >= 0; --index)
        {
            const QJsonObject record = demArr.at(index).toObject();
            if (record.value(QStringLiteral("dem_tif")).toString() == resolvedDem)
            {
                matchedDemRecord = record;
                break;
            }
        }
    }
    if (resolvedDem.isEmpty() || !QFileInfo::exists(resolvedDem))
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("生成正射影像"),
                             QStringLiteral("找不到 DEM 文件，请先执行[创建 DEM]。"));
        return;
    }

    QStringList sourceImages = images;
    if (sourceImages.isEmpty())
    {
        sourceImages = m_owner->getAllImages();
    }
    if (sourceImages.isEmpty())
    {
        QMessageBox::warning(m_parentWidget,
                             QStringLiteral("生成正射影像"),
                             QStringLiteral("项目中没有可用影像。"));
        return;
    }

    QString out = outputPath.trimmed();
    if (out.isEmpty())
    {
        const QString root = ProjectIO::projectRootFromPlascan(m_owner->currentProjectPath());
        out = QDir(root).filePath(QStringLiteral("assets/ortho/relative_dom.tif"));
    }
    out = QDir::cleanPath(out);

    const double requestedResolution = resolution > 0.0 ? resolution : 0.0;

    QJsonObject orthoResult;
    if (!runOrthoProductOrWarn(sourceImages,
                               resolvedDem,
                               out,
                               requestedResolution,
                               meta,
                               QStringLiteral("生成正射影像"),
                               &orthoResult))
    {
        return;
    }

    const double effectiveResolution = orthoResult.value(QStringLiteral("output_resolution"))
                                           .toDouble(requestedResolution > 0.0 ? requestedResolution : 1.0);

    QJsonObject record = makeOrthoResultRecord(
        orthoResult.value(QStringLiteral("created_at")).toString(),
        resolvedDem,
        orthoResult.value(QStringLiteral("output_path")).toString(),
        orthoResult.value(QStringLiteral("source_image_count")).toInt(),
        sourceImages,
        true,
        effectiveResolution);
    record[QStringLiteral("dom_georeferenced")] = orthoResult.value(QStringLiteral("dom_georeferenced")).toBool(false);
    record[QStringLiteral("projection_wkt_present")] =
        orthoResult.value(QStringLiteral("projection_wkt_present")).toBool(false);
    if (!matchedDemRecord.isEmpty())
    {
        record[QStringLiteral("dem_reference")] = matchedDemRecord.value(QStringLiteral("dem_reference")).toString();
    }
    upsertMetaArrayRecordByPath(&meta,
                                QStringLiteral("ortho_results"),
                                QStringLiteral("output_path"),
                                record);
    persistProjectMeta(m_projectData, meta, true);

    QMessageBox::information(m_parentWidget,
                             QStringLiteral("生成正射影像"),
                             QStringLiteral("正射影像已生成：\n%1\n分辨率: %2 m/px\n与 DEM 投影一致: %3")
                                 .arg(record.value(QStringLiteral("output_path")).toString())
                                 .arg(record.value(QStringLiteral("resolution")).toDouble(), 0, 'f', 3)
                                 .arg(record.value(QStringLiteral("dom_georeferenced")).toBool(false)
                                          ? QStringLiteral("是")
                                          : QStringLiteral("否")));
}
void ProjectTerrainProductsManager::runFullDemPipelineInBackground(const DemPipelineContext &ctx)
{
    LOG_INFO(QStringLiteral("[DEM流水线] ========== 流水线启动 =========="));
    LOG_INFO(QStringLiteral("[DEM流水线] 影像数量: %1").arg(ctx.images.size()));
    for (int i = 0; i < ctx.images.size(); ++i)
        LOG_INFO(QStringLiteral("[DEM流水线]   [%1] %2").arg(i).arg(ctx.images[i]));
    LOG_INFO(QStringLiteral("[DEM流水线] 特征算法: %1 | 匹配算法: %2").arg(ctx.featureAlgorithm, ctx.matchAlgorithm));
    LOG_INFO(QStringLiteral("[DEM流水线] 输出目录: %1").arg(ctx.outputDir.isEmpty() ? QStringLiteral("(自动)") : ctx.outputDir));

    // 步骤 1: 特征提取（在后台线程同步执行，SuperPointRunner::run 是纯同步的）
    LOG_INFO(QStringLiteral("[DEM流水线] ── 步骤 1/5: 特征提取 (%1) ──").arg(ctx.featureAlgorithm));
    emit demPipelineProgressChanged(QStringLiteral("特征提取"), 0);

    std::atomic<bool> cancelFlag(false);
    std::atomic<int> progressCount(0);

    QJsonObject featureConfig;
    featureConfig[QStringLiteral("feature_algorithm")] = ctx.featureAlgorithm;
    featureConfig[QStringLiteral("device")] = QStringLiteral("cuda");
    featureConfig[QStringLiteral("max_keypoints")] = 2048;

    const bool featureOk = SuperPointRunner::run(featureConfig, ctx.images, m_owner, cancelFlag, progressCount);
    if (!featureOk)
    {
        LOG_ERROR(QStringLiteral("[DEM流水线] ✗ 特征提取失败（SuperPointRunner::run 返回 false）"));
        QMetaObject::invokeMethod(m_parentWidget, [this]() {
            QMessageBox::warning(m_parentWidget, QStringLiteral("创建相对 DEM"), QStringLiteral("特征提取失败，流水线中止。"));
        }, Qt::QueuedConnection);
        emit demPipelineFinished(false, QStringLiteral("特征提取失败"));
        return;
    }
    LOG_INFO(QStringLiteral("[DEM流水线] ✓ 特征提取完成"));

    // 步骤 2: 特征匹配（同步执行）
    LOG_INFO(QStringLiteral("[DEM流水线] ── 步骤 2/5: 特征匹配 (%1) ──").arg(ctx.matchAlgorithm));
    emit demPipelineProgressChanged(QStringLiteral("特征匹配"), 20);

    QStringList imagePairs;
    for (int i = 0; i < ctx.images.size(); ++i)
        for (int j = i + 1; j < ctx.images.size(); ++j)
            imagePairs << QStringLiteral("%1|%2").arg(ctx.images[i], ctx.images[j]);
    LOG_INFO(QStringLiteral("[DEM流水线] 匹配对数: %1").arg(imagePairs.size()));

    QJsonObject matchConfig;
    matchConfig[QStringLiteral("match_algorithm")] = ctx.matchAlgorithm;
    matchConfig[QStringLiteral("device")] = QStringLiteral("cuda");
    matchConfig[QStringLiteral("outlier_filter")] = QStringLiteral("fundamental_ransac");

    progressCount.store(0);
    FeatureMatchRunner::run(matchConfig, imagePairs, m_owner, cancelFlag, progressCount);
    LOG_INFO(QStringLiteral("[DEM流水线] ✓ 特征匹配完成"));

    // 步骤 3-5: 正式 SfM/BA 稀疏云 → MVS → DEM 需要在主线程通过信号链驱动
    LOG_INFO(QStringLiteral("[DEM流水线] ── 步骤 3/5: 检查正式 SfM/BA 稀疏点云（切换到主线程信号链）──"));
    emit demPipelineProgressChanged(QStringLiteral("检查正式空三结果"), 40);

    const QString demOutputDir = ctx.outputDir;
    const double demResolution = ctx.demResolution;
    const QString demType = ctx.demType;
    const QStringList cameraPaths = ctx.cameraPaths;

    QMetaObject::invokeMethod(this, [this, demOutputDir, demResolution, demType, cameraPaths]()
    {
        const int productionAtIndex = findLatestProductionAtResultIndex(m_projectData->metadata());
        if (productionAtIndex < 0)
        {
            LOG_ERROR(QStringLiteral("[DEM流水线] ✗ 未找到正式 SfM/BA 稀疏点云，拒绝使用两视预览云继续 DEM 流水线"));
            QMessageBox::warning(m_parentWidget,
                                 QStringLiteral("创建相对 DEM"),
                                 QStringLiteral("未找到可用的正式 SfM/BA 稀疏点云结果。\n请先运行[工作流程 → 三维重建/空三]，不要使用两视预览三角化作为 DEM 输入。"));
            emit demPipelineFinished(false, QStringLiteral("缺少正式 SfM/BA 稀疏点云"));
            return;
        }

        LOG_INFO(QStringLiteral("[DEM流水线] ✓ 使用正式 SfM/BA 稀疏点云结果 index=%1，启动 MVS（CUDA PatchMatch）...")
                     .arg(productionAtIndex));
        emit demPipelineProgressChanged(QStringLiteral("密集重建 (MVS)"), 55);

        // 监听元数据变化：dense_cloud_results 出现新记录即表示融合完成
        auto *connMeta = new QMetaObject::Connection();
        *connMeta = connect(m_owner, &ProjectManager::projectMetadataChanged, this,
            [this, connMeta, demOutputDir, demResolution, demType, cameraPaths](const QJsonObject &metaChanged)
            {
                const QJsonArray denseArr = metaChanged.value(QStringLiteral("dense_cloud_results")).toArray();
                LOG_INFO(QStringLiteral("[DEM流水线] projectMetadataChanged: dense_cloud_results 条目数=%1").arg(denseArr.size()));
                if (denseArr.isEmpty())
                    return;
                const QJsonObject lastRecord = denseArr.last().toObject();
                const QString plyPath = lastRecord.value(QStringLiteral("dense_cloud_xyz")).toString();
                LOG_INFO(QStringLiteral("[DEM流水线] 最新 dense_cloud_xyz=%1 exists=%2")
                             .arg(plyPath)
                             .arg(QFileInfo::exists(plyPath) ? QStringLiteral("true") : QStringLiteral("false")));
                if (plyPath.isEmpty() || !QFileInfo::exists(plyPath))
                    return;

                disconnect(*connMeta);
                delete connMeta;

                LOG_INFO(QStringLiteral("[DEM流水线] ✓ 密集点云就绪（%1），启动 DEM 生成...").arg(plyPath));
                emit demPipelineProgressChanged(QStringLiteral("DEM 生成"), 85);

                const QString outDir = resolveProjectOutputDir(m_owner->currentProjectPath(),
                                                               demOutputDir.trimmed(),
                                                               QStringLiteral("assets/dem/relative_dem"));

                // 优先使用深度图直接生成 DEM（覆盖率接近 100%）
                QJsonObject terrainResult;
                bool demOk = false;

                // 尝试从 MVS 输出目录加载深度图
                const QString mvsDir = QFileInfo(plyPath).absolutePath();
                const QString depth0Path = QDir(mvsDir).filePath(QStringLiteral("depth_0.yml.gz"));
                const QString depth1Path = QDir(mvsDir).filePath(QStringLiteral("depth_1.yml.gz"));

                if (QFileInfo::exists(depth0Path) && !cameraPaths.isEmpty())
                {
                    LOG_INFO(QStringLiteral("[DEM流水线] 使用深度图直接生成 DEM（图像空间方法）"));
                    std::vector<cv::Mat> depthMaps;
                    std::vector<xjw::Camera> cameras;

                    cv::FileStorage fs0(depth0Path.toStdString(), cv::FileStorage::READ);
                    if (fs0.isOpened())
                    {
                        cv::Mat d0;
                        fs0["depth"] >> d0;
                        if (!d0.empty())
                            depthMaps.push_back(d0);
                        fs0.release();
                    }
                    if (QFileInfo::exists(depth1Path))
                    {
                        cv::FileStorage fs1(depth1Path.toStdString(), cv::FileStorage::READ);
                        if (fs1.isOpened())
                        {
                            cv::Mat d1;
                            fs1["depth"] >> d1;
                            if (!d1.empty())
                                depthMaps.push_back(d1);
                            fs1.release();
                        }
                    }

                    for (const QString &camPath : cameraPaths)
                    {
                        xjw::Camera cam;
                        if (cam.loadFromFile(camPath.toStdString()))
                            cameras.push_back(cam);
                    }

                    if (!depthMaps.empty() && !cameras.empty())
                    {
                        QString demErr;
                        demOk = xjw::TerrainPipeline::generateDemFromDepthMaps(
                            depthMaps, cameras, outDir, &terrainResult, &demErr);
                        if (!demOk)
                            LOG_ERROR(QStringLiteral("[DEM流水线] 深度图直接 DEM 失败: %1，回退到点云方法").arg(demErr));
                    }
                }

                // 回退：使用点云方法
                if (!demOk)
                {
                    LOG_INFO(QStringLiteral("[DEM流水线] 使用点云方法生成 DEM"));
                    demOk = runDemProductsOrWarn(plyPath, outDir, demResolution, demType, false,
                                                 QStringLiteral("创建相对 DEM"), &terrainResult);
                }

                if (!demOk)
                {
                    LOG_ERROR(QStringLiteral("[DEM流水线] ✗ DEM 生成失败"));
                    emit demPipelineFinished(false, QStringLiteral("DEM 生成失败"));
                    return;
                }

                QJsonObject metaUpdated = m_projectData->metadata();
                QJsonObject depthResult = makeDepthResultRecord(
                    terrainResult.value(QStringLiteral("created_at")).toString(),
                    terrainResult.value(QStringLiteral("depth_png")).toString(),
                    terrainResult.value(QStringLiteral("grid_width")).toInt(),
                    terrainResult.value(QStringLiteral("grid_height")).toInt(),
                    plyPath);
                upsertMetaArrayRecordByPath(&metaUpdated, QStringLiteral("depth_map_results"),
                                            QStringLiteral("depth_png"), depthResult);

                QJsonObject demResult = makeDemResultRecord(
                    terrainResult.value(QStringLiteral("created_at")).toString(),
                    outDir, QString(),
                    terrainResult.value(QStringLiteral("dem_tif")).toString(),
                    demType, demResolution, QString(), QStringList());
                demResult[QStringLiteral("source_dense_cloud")] = plyPath;
                demResult[QStringLiteral("dem_reference")] = QStringLiteral("relative");
                demResult[QStringLiteral("depth_preview_png")] = terrainResult.value(QStringLiteral("depth_png")).toString();
                demResult[QStringLiteral("relative_z_offset")] = terrainResult.value(QStringLiteral("relative_z_offset")).toDouble(0.0);
                upsertMetaArrayRecordByPath(&metaUpdated, QStringLiteral("dem_results"),
                                            QStringLiteral("dem_tif"), demResult);
                persistProjectMeta(m_projectData, metaUpdated, true);

                LOG_INFO(QStringLiteral("[DEM流水线] ✓ DEM 生成完成: %1")
                             .arg(terrainResult.value(QStringLiteral("dem_tif")).toString()));
                emit demPipelineProgressChanged(QStringLiteral("完成"), 100);
                emit demPipelineFinished(true, QStringLiteral("DEM 流水线已完成"));
            });

        // 同时监听 MVS 失败
        auto *connMvsFail = new QMetaObject::Connection();
        *connMvsFail = connect(m_owner, &ProjectManager::mvsProgressFinished, this,
            [this, connMvsFail, connMeta](bool mvsSuccess)
            {
                LOG_INFO(QStringLiteral("[DEM流水线] mvsProgressFinished: success=%1").arg(mvsSuccess));
                if (mvsSuccess)
                    return; // 成功时由 projectMetadataChanged 处理
                disconnect(*connMvsFail);
                delete connMvsFail;
                disconnect(*connMeta);
                delete connMeta;
                LOG_ERROR(QStringLiteral("[DEM流水线] ✗ MVS 失败（mvsProgressFinished(false)）"));
                emit demPipelineFinished(false, QStringLiteral("MVS 密集重建失败"));
            }, Qt::SingleShotConnection);

        LOG_INFO(QStringLiteral("[DEM流水线] 主线程：调用 startGenerateDenseCloudAsync..."));
        QJsonObject mvsSettings;
        mvsSettings[QStringLiteral("pipeline_mode")] = true;
        mvsSettings[QStringLiteral("at_index")] = productionAtIndex;
        // 两视图立体对：全分辨率，不降采样
        mvsSettings[QStringLiteral("resScale")] = 1.0;
        mvsSettings[QStringLiteral("iterations")] = 16;
        mvsSettings[QStringLiteral("cuda")] = true;
        mvsSettings[QStringLiteral("minConsistentViews")] = 1;
        m_owner->startGenerateDenseCloudAsync(mvsSettings);
    }, Qt::QueuedConnection);
}
