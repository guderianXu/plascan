// =============================================================================
// 文件名: BundleAdjustService.cpp
// 描述:   光束法平差服务实现，详细说明见 BundleAdjustService.h。
//
//         本文件专注于算法实现，不依赖任何 Qt Widget。
//         所有 QWidget 级别的错误弹框均由调用方（ProjectManager）负责。
// =============================================================================
#include "BundleAdjustService.h"

#include "FramePinholeCamera.h"
#include "BundleAdjust.h"
#include "LaserConstraintAssociation.h"
#include "LaserConstraintMap.h"
#include "PlanetaryLaserBaAdapter.h"
#include "PlanetaryLaserJson.h"
#include "Logger.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPainter>
#include <QPen>
#include <QColor>
#include <QTextStream>

#include <cmath>
#include <utility>

namespace xjw
{
namespace gui
{

bool mergePlanetaryLaserProjectImageAliases(
    const QJsonObject &meta,
    const QStringList &imagePathByIndex,
    QVector<QStringList> *aliasesByCameraIndex,
    QString *errorMessage)
{
    if (!aliasesByCameraIndex)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("行星激光影像别名输出为空");
        }
        return false;
    }
    if (!aliasesByCameraIndex->isEmpty() &&
        aliasesByCameraIndex->size() != imagePathByIndex.size())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("行星激光影像别名数量与 BA 相机数量不一致");
        }
        return false;
    }
    if (aliasesByCameraIndex->isEmpty())
    {
        aliasesByCameraIndex->resize(imagePathByIndex.size());
    }

    const auto normalize_alias_path = [](const QString &path)
    {
        QString normalized = path.trimmed();
        if (normalized.isEmpty())
        {
            return QString();
        }
        normalized.replace(QLatin1Char('\\'), QLatin1Char('/'));
        return xjw::common::project::normalizePath(normalized);
    };

    QMap<QString, QString> uuidByPath;
    QMap<QString, QString> pathByUuid;
    for (const QJsonValue &value : meta.value(QStringLiteral("images")).toArray())
    {
        const QJsonObject image = value.toObject();
        const QString path = normalize_alias_path(
            image.value(QStringLiteral("path")).toString());
        const QString uuid = image.value(QStringLiteral("image_uuid")).toString().trimmed();
        if (path.isEmpty() || uuid.isEmpty())
        {
            continue;
        }
        if ((uuidByPath.contains(path) && uuidByPath.value(path) != uuid) ||
            (pathByUuid.contains(uuid) && pathByUuid.value(uuid) != path))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "工程影像 path 与 image_uuid 不是一一对应，不能安全建立行星激光别名");
            }
            return false;
        }
        uuidByPath.insert(path, uuid);
        pathByUuid.insert(uuid, path);
    }

    for (int cameraIndex = 0; cameraIndex < imagePathByIndex.size(); ++cameraIndex)
    {
        const QString path = normalize_alias_path(imagePathByIndex.at(cameraIndex));
        const QString uuid = uuidByPath.value(path);
        if (!uuid.isEmpty() && !aliasesByCameraIndex->at(cameraIndex).contains(uuid))
        {
            (*aliasesByCameraIndex)[cameraIndex].append(uuid);
        }
    }
    return true;
}

// ──────────────────────────────────────────────────────────────────────────────
// 匿名命名空间：本文件内部使用的辅助工具
// ──────────────────────────────────────────────────────────────────────────────
namespace
{

// ── 生成 BA 评估对比图：包含 RMS 柱状图 + 相机位移柱状图 ─────────────────────
void generateEvalPlots(
    const xjw::BAResult& baResult,
    const QJsonArray&    cameraPreview,
    const QString&       outputDir,
    QJsonObject*         filesOut       ///< 输出：将图片路径写入此 JSON 对象
)
{
    if (!filesOut)
    {
        return;
    }

    // ── 图1：全局平均重投影误差（RMS），BA 前后对比柱状图 ─────────────────
    const QString rmsPlotPath = QDir(outputDir).filePath(QStringLiteral("ba_rms_compare.png"));
    QImage rmsImg(640, 360, QImage::Format_ARGB32_Premultiplied);
    rmsImg.fill(Qt::white);
    {
        QPainter p(&rmsImg);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setPen(QPen(Qt::black, 2));
        p.drawRect(40, 40, 560, 260);

        const double maxV = std::max(1e-6, std::max(baResult.meanRmsBefore, baResult.meanRmsAfter));
        const int hBefore = static_cast<int>((baResult.meanRmsBefore / maxV) * 220.0);
        const int hAfter  = static_cast<int>((baResult.meanRmsAfter  / maxV) * 220.0);

        // BA 前（红色）
        p.setBrush(QColor(240, 120, 120));
        p.drawRect(180, 300 - hBefore, 100, hBefore);
        // BA 后（蓝色）
        p.setBrush(QColor(120, 180, 240));
        p.drawRect(360, 300 - hAfter, 100, hAfter);

        p.setPen(Qt::black);
        p.drawText(180, 325, QStringLiteral("RMS前"));
        p.drawText(360, 325, QStringLiteral("RMS后"));
        p.drawText(160, 25, QStringLiteral("BA 平均重投影误差对比"));
        p.drawText(145, 300 - hBefore - 8, QString::number(baResult.meanRmsBefore, 'f', 4));
        p.drawText(345, 300 - hAfter  - 8, QString::number(baResult.meanRmsAfter,  'f', 4));
    }
    rmsImg.save(rmsPlotPath);

    // ── 图2：每台相机中心位移量（m），柱状图 ─────────────────────────────
    const QString camDeltaPath = QDir(outputDir).filePath(QStringLiteral("ba_camera_delta.png"));
    QImage camImg(900, 420, QImage::Format_ARGB32_Premultiplied);
    camImg.fill(Qt::white);
    {
        QPainter p(&camImg);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setPen(QPen(Qt::black, 2));
        p.drawRect(40, 40, 820, 300);

        // 收集全部位移值并求最大值（用于归一化高度）
        QVector<double> deltas;
        deltas.reserve(cameraPreview.size());
        double maxD = 1e-6;
        for (const QJsonValue& v : cameraPreview)
        {
            const double d = v.toObject().value(QStringLiteral("delta_c_m")).toDouble();
            deltas.push_back(d);
            if (d > maxD) maxD = d;
        }

        const int n = deltas.size();
        if (n > 0)
        {
            const int barW = std::max(6, 760 / n);
            for (int i = 0; i < n; ++i)
            {
                const int h = static_cast<int>((deltas.at(i) / maxD) * 250.0);
                const int x = 60 + i * barW;
                p.setBrush(QColor(120, 200, 150));
                p.drawRect(x, 340 - h, std::max(4, barW - 2), h);
            }
        }

        p.setPen(Qt::black);
        p.drawText(360, 25, QStringLiteral("相机中心位移量（m）"));
    }
    camImg.save(camDeltaPath);

    // 将生成的图片路径写入输出 JSON
    (*filesOut)[QStringLiteral("rms_plot")]         = rmsPlotPath;
    (*filesOut)[QStringLiteral("camera_delta_plot")] = camDeltaPath;
}

} // namespace（匿名）

// ──────────────────────────────────────────────────────────────────────────────
// BundleAdjustService::run  — 光束法平差核心流程
// ──────────────────────────────────────────────────────────────────────────────
BaServiceResult BundleAdjustService::run(
    const std::vector<xjw::FramePinholeCamera>& cameras,
    std::vector<xjw::BATrack>&      tracks,
    const BaServiceOptions&         opts
)
{
    BaServiceResult result;

    // ── 基础检查 ──────────────────────────────────────────────────────────
    if (cameras.size() < 2)
    {
        result.errorMessage = QStringLiteral("至少需要两台相机");
        return result;
    }
    if (tracks.empty())
    {
        result.errorMessage = QStringLiteral("轨迹列表为空，无法执行平差");
        return result;
    }
    if (opts.outputDir.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("输出目录未指定");
        return result;
    }

    const QString outDir = QDir::cleanPath(opts.outputDir);
    QDir().mkpath(outDir);

    const auto makeDryRunResult = [&tracks]()
    {
        BaServiceResult dryResult;
        QJsonObject dryObj;
        dryObj[QStringLiteral("track_count")]     = static_cast<int>(tracks.size());
        dryObj[QStringLiteral("optimized_count")] = 0;
        dryObj[QStringLiteral("mean_rms_before")] = 0.0;
        dryObj[QStringLiteral("mean_rms_after")]  = 0.0;
        QJsonObject files;
        files[QStringLiteral("summary_txt")] = QStringLiteral("[DryRun] 未生成文件");
        files[QStringLiteral("points_csv")]  = QStringLiteral("[DryRun] 未生成文件");
        files[QStringLiteral("camera_csv")]  = QStringLiteral("[DryRun] 未生成文件");
        dryObj[QStringLiteral("files")]      = files;
        dryResult.success = true;
        dryResult.resultJson = dryObj;
        return dryResult;
    };

    // 行星测距 dry-run 仍需解析格式、检查传感器模型和完成严格影像关联。
    if (opts.dryRun && !opts.enablePlanetaryLaserRangeConstraints)
    {
        return makeDryRunResult();
    }

    // ── LiDAR 点到面约束预处理 ────────────────────────────────────────────
    xjw::BAOptions baOptions = opts.baOpt;
    xjw::lidar::LaserAssociationSummary laserAssociationSummary;
    int laserMapSampleCount = 0;
    double effectiveLaserWeight = 0.0;
    xjw::lidar::PlanetaryLaserDataset planetaryLaserDataset;
    xjw::lidar::PlanetaryLaserBaAdapterSummary planetaryLaserSummary;
    QStringList planetaryLaserCameraPaths;
    if (opts.enableLaserConstraints && opts.enablePlanetaryLaserRangeConstraints)
    {
        result.errorMessage = QStringLiteral(
            "扫描点云点到面约束与行星稀疏激光测距是两种独立观测模型；"
            "当前一次运行只能选择其中一种");
        return result;
    }
    if (opts.enableLaserConstraints)
    {
        if (opts.laserConstraintCloudPath.trimmed().isEmpty())
        {
            result.errorMessage = QStringLiteral("LiDAR 约束点云路径未指定");
            return result;
        }
        if (!QFileInfo::exists(opts.laserConstraintCloudPath))
        {
            result.errorMessage = QStringLiteral("LiDAR 约束点云不存在: %1").arg(opts.laserConstraintCloudPath);
            return result;
        }

        xjw::lidar::LaserConstraintMapOptions mapOptions;
        mapOptions.maxCurvature = opts.laserMaxCurvature;
        mapOptions.voxelSizeMeters = opts.laserVoxelSizeMeters;
        mapOptions.maxSamples = opts.laserMaxSamples;
        mapOptions.useMissingNormalsAsHeightPlanes = opts.laserUseMissingNormalsAsHeightPlanes;

        xjw::lidar::LaserConstraintMap laserMap;
        std::string laserError;
        if (!laserMap.loadPly(xjw::common::io::toUtf8Path(opts.laserConstraintCloudPath), mapOptions, &laserError))
        {
            result.errorMessage = QStringLiteral("读取 LiDAR 约束点云失败: %1")
                                      .arg(QString::fromStdString(laserError));
            return result;
        }
        laserMapSampleCount = static_cast<int>(laserMap.size());

        xjw::lidar::LaserAssociationOptions associationOptions;
        associationOptions.maxDistanceMeters = opts.laserAssociationMaxDistanceMeters;
        associationOptions.weight = 1.0;
        associationOptions.enableQualityWeighting = true;
        associationOptions.maxCurvatureForWeighting = opts.laserMaxCurvature;
        associationOptions.minQualityWeight = 0.05;
        laserAssociationSummary = xjw::lidar::attachLaserPlaneConstraints(laserMap, &tracks, associationOptions);
        if (laserAssociationSummary.associatedTracks <= 0)
        {
            result.errorMessage = QStringLiteral(
                "LiDAR 点云未关联到任何 BA track；请检查坐标系和最大关联距离（当前 %1 m）")
                                      .arg(opts.laserAssociationMaxDistanceMeters, 0, 'g', 8);
            return result;
        }

        baOptions.enableLaserPlaneConstraints = true;
        const double sigmaMeters = std::max(1.0e-9, opts.laserSigmaMeters);
        effectiveLaserWeight = opts.laserWeight > 0.0
            ? opts.laserWeight
            : 1.0 / (sigmaMeters * sigmaMeters);
        baOptions.laserPlaneWeight = effectiveLaserWeight;
        baOptions.laserHuberDeltaMeters = opts.laserHuberDeltaMeters;
    }

    // ── 行星稀疏激光测距 shot 预处理 ────────────────────────────────────
    if (opts.enablePlanetaryLaserRangeConstraints)
    {
        if (opts.planetaryLaserDataPath.trimmed().isEmpty())
        {
            result.errorMessage = QStringLiteral("行星激光测距 JSON 路径未指定");
            return result;
        }
        if (!QFileInfo::exists(opts.planetaryLaserDataPath))
        {
            result.errorMessage = QStringLiteral("行星激光测距 JSON 不存在: %1")
                                      .arg(opts.planetaryLaserDataPath);
            return result;
        }
        if (!std::isfinite(opts.planetaryLaserRangeWeight) ||
            opts.planetaryLaserRangeWeight <= 0.0 ||
            !std::isfinite(opts.planetaryLaserRangeHuberDeltaSigma) ||
            opts.planetaryLaserRangeHuberDeltaSigma < 0.0)
        {
            result.errorMessage = QStringLiteral(
                "行星激光测距全局权重必须为正，Huber 阈值必须有限且非负");
            return result;
        }

        std::string laserError;
        if (!xjw::lidar::loadPlanetaryLaserJsonFile(
                xjw::common::io::toUtf8Path(opts.planetaryLaserDataPath),
                opts.planetaryLaserParseOptions,
                &planetaryLaserDataset,
                &laserError))
        {
            result.errorMessage = QStringLiteral("读取行星激光测距数据失败: %1")
                                      .arg(QString::fromStdString(laserError));
            return result;
        }

        if (opts.imagePathByIndex.size() != static_cast<int>(cameras.size()))
        {
            result.errorMessage = QStringLiteral(
                "行星激光 shot 关联要求 imagePathByIndex 与 BA 相机数量一致");
            return result;
        }
        planetaryLaserCameraPaths = opts.imagePathByIndex;
        if (!opts.planetaryLaserImageAliasesByCameraIndex.isEmpty() &&
            opts.planetaryLaserImageAliasesByCameraIndex.size() !=
                static_cast<int>(cameras.size()))
        {
            result.errorMessage = QStringLiteral(
                "行星激光额外影像别名必须与 BA 相机数量一致");
            return result;
        }

        xjw::lidar::PlanetaryLaserBaAdapterOptions adapterOptions;
        adapterOptions.imageAliasesByCameraIndex.resize(cameras.size());
        for (int cameraIndex = 0;
             cameraIndex < planetaryLaserCameraPaths.size();
             ++cameraIndex)
        {
            adapterOptions.imageAliasesByCameraIndex[static_cast<std::size_t>(cameraIndex)] = {
                planetaryLaserCameraPaths[cameraIndex].toUtf8().toStdString(),
                QStringLiteral("camera_index:%1").arg(cameraIndex).toStdString(),
            };
            if (!opts.planetaryLaserImageAliasesByCameraIndex.isEmpty())
            {
                for (const QString &alias :
                     opts.planetaryLaserImageAliasesByCameraIndex[cameraIndex])
                {
                    const QString normalizedAlias = alias.trimmed();
                    if (!normalizedAlias.isEmpty())
                    {
                        adapterOptions.imageAliasesByCameraIndex[
                            static_cast<std::size_t>(cameraIndex)]
                            .push_back(normalizedAlias.toUtf8().toStdString());
                    }
                }
            }
        }
        adapterOptions.cameraCoordinateFrame =
            opts.planetaryLaserCameraCoordinateFrame.trimmed().toUtf8().toStdString();
        adapterOptions.cameraSensorFrame =
            opts.planetaryLaserCameraSensorFrame.trimmed().toUtf8().toStdString();
        adapterOptions.confirmUnknownSensorModelIsFrame =
            opts.planetaryLaserConfirmUnknownSensorIsFrame;
        adapterOptions.confirmUnknownRangeTypeIsOneWay =
            opts.planetaryLaserConfirmUnknownRangeIsOneWay;
        adapterOptions.allowUnmappedShots = opts.planetaryLaserAllowUnmappedShots;
        adapterOptions.allowUnmappedMeasuredImages =
            opts.planetaryLaserAllowUnmappedMeasuredImages;

        std::vector<xjw::BALaserRangeConstraint> rangeConstraints;
        if (!xjw::lidar::buildPlanetaryLaserRangeConstraints(
                planetaryLaserDataset,
                adapterOptions,
                &rangeConstraints,
                &planetaryLaserSummary,
                &laserError))
        {
            result.errorMessage = QStringLiteral("行星激光测距数据无法接入 BA: %1")
                                      .arg(QString::fromStdString(laserError));
            return result;
        }

        baOptions.enableLaserRangeConstraints = true;
        baOptions.laserRangeWeight = opts.planetaryLaserRangeWeight;
        baOptions.laserRangeHuberDelta = opts.planetaryLaserRangeHuberDeltaSigma;
        baOptions.laserRangeConstraints = std::move(rangeConstraints);
    }

    if (opts.dryRun)
    {
        return makeDryRunResult();
    }

    // ── 执行光束法平差 ─────────────────────────────────────────────────────
    // xjw::BundleAdjust::optimizePoints 内部使用 Levenberg-Marquardt 算法，
    // 对所有相机与所有点交替迭代，最小化重投影误差的 Huber 加权和。
    const xjw::BAResult baResult = xjw::BundleAdjust::optimizePoints(cameras, tracks, baOptions);
    if (baOptions.cancelFlag && baOptions.cancelFlag->load())
    {
        result.errorMessage = QStringLiteral("用户取消了光束法平差");
        return result;
    }

    // ── 准备各输出文件路径 ─────────────────────────────────────────────────
    const QString tsaiDir       = QDir(outDir).filePath(QStringLiteral("refined_tsai"));
    const QString summaryTxtPath = QDir(outDir).filePath(QStringLiteral("ba_summary.txt"));
    const QString pointsCsvPath  = QDir(outDir).filePath(QStringLiteral("ba_points_metrics.csv"));
    const QString camerasCsvPath = QDir(outDir).filePath(QStringLiteral("ba_camera_metrics.csv"));
    const QString runJsonPath    = QDir(outDir).filePath(QStringLiteral("ba_run_summary.json"));

    if (opts.exportTsai)
    {
        QDir().mkpath(tsaiDir);
    }

    // ── 构建输出 JSON 骨架 ─────────────────────────────────────────────────
    QJsonObject saveObj;
    saveObj[QStringLiteral("created_at")]          = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    saveObj[QStringLiteral("output_dir")]           = outDir;
    saveObj[QStringLiteral("selected_images")]      = QJsonArray::fromStringList(opts.selectedImages);
    saveObj[QStringLiteral("camera_count")]         = static_cast<int>(cameras.size());
    saveObj[QStringLiteral("track_count")]          = baResult.totalTracks;
    saveObj[QStringLiteral("optimized_count")]      = baResult.optimizedTracks;
    saveObj[QStringLiteral("mean_rms_before")]      = baResult.meanRmsBefore;
    saveObj[QStringLiteral("mean_rms_after")]       = baResult.meanRmsAfter;
    saveObj[QStringLiteral("threads")]              = opts.threads;
    saveObj[QStringLiteral("refined_camera_count")] = baResult.refinedCameraCount;
    saveObj[QStringLiteral("ba_requested_backend")] =
        QString::fromLatin1(xjw::BundleAdjust::backendName(baResult.requestedBackend));
    saveObj[QStringLiteral("ba_used_backend")] =
        QString::fromLatin1(xjw::BundleAdjust::backendName(baResult.usedBackend));
    saveObj[QStringLiteral("ba_used_gpu")] = baResult.usedGpu;
    saveObj[QStringLiteral("ba_backend_fallback")] = baResult.backendFallback;
    saveObj[QStringLiteral("ba_backend_message")] =
        QString::fromUtf8(baResult.backendMessage.c_str());
    saveObj[QStringLiteral("ba_backend_selection_reason")] =
        QString::fromUtf8(baResult.backendSelectionReason.c_str());
    saveObj[QStringLiteral("ba_quality_gate_rejected")] = baResult.qualityGateRejected;
    saveObj[QStringLiteral("ba_solve_status")] =
        QString::fromLatin1(xjw::BundleAdjust::solveStatusName(baResult.solveStatus));
    saveObj[QStringLiteral("ba_solution_usable")] = baResult.solutionUsable;
    saveObj[QStringLiteral("ba_quality_gate_message")] =
        QString::fromUtf8(baResult.qualityGateMessage.c_str());
    saveObj[QStringLiteral("ba_valid_track_ratio")] = baResult.validTrackRatio;
    saveObj[QStringLiteral("ba_ceres_linear_solver")] =
        QString::fromStdString(baResult.ceresLinearSolverName);
    saveObj[QStringLiteral("ba_ceres_estimated_cuda_bytes")] =
        static_cast<qint64>(baResult.ceresEstimatedCudaBytes);
    saveObj[QStringLiteral("ba_ceres_cuda_free_bytes")] =
        static_cast<qint64>(baResult.ceresCudaFreeBytes);
    saveObj[QStringLiteral("ba_ceres_initial_cost")] = baResult.ceresInitialCost;
    saveObj[QStringLiteral("ba_ceres_final_cost")] = baResult.ceresFinalCost;
    saveObj[QStringLiteral("ba_ceres_successful_steps")] = baResult.ceresSuccessfulSteps;
    saveObj[QStringLiteral("ba_ceres_unsuccessful_steps")] = baResult.ceresUnsuccessfulSteps;
    saveObj[QStringLiteral("ba_ceres_rejected_initial_tracks")] =
        baResult.ceresRejectedInitialTracks;
    saveObj[QStringLiteral("ba_setup_seconds")] = baResult.setupSeconds;
    saveObj[QStringLiteral("ba_solve_seconds")] = baResult.solveSeconds;
    saveObj[QStringLiteral("ba_postprocess_seconds")] =
        baResult.postprocessSeconds;
    saveObj[QStringLiteral("ba_total_seconds")] = baResult.totalSeconds;
    saveObj[QStringLiteral("ba_observation_count")] = baResult.observationCount;
    saveObj[QStringLiteral("ba_native_cuda_initial_cost")] = baResult.nativeCudaInitialCost;
    saveObj[QStringLiteral("ba_native_cuda_final_cost")] = baResult.nativeCudaFinalCost;
    saveObj[QStringLiteral("ba_native_cuda_accepted_steps")] = baResult.nativeCudaAcceptedSteps;
    saveObj[QStringLiteral("ba_native_cuda_rejected_steps")] = baResult.nativeCudaRejectedSteps;
    saveObj[QStringLiteral("ba_native_cuda_active_cameras")] = baResult.nativeCudaActiveCameras;
    saveObj[QStringLiteral("ba_native_cuda_active_tracks")] = baResult.nativeCudaActiveTracks;
    saveObj[QStringLiteral("ba_native_cuda_active_observations")] = baResult.nativeCudaActiveObservations;
    saveObj[QStringLiteral("ba_native_cuda_upload_seconds")] = baResult.nativeCudaUploadSeconds;
    saveObj[QStringLiteral("ba_native_cuda_kernel_seconds")] = baResult.nativeCudaKernelSeconds;
    saveObj[QStringLiteral("ba_native_cuda_download_seconds")] = baResult.nativeCudaDownloadSeconds;
    saveObj[QStringLiteral("ba_native_cuda_host_cost_seconds")] = baResult.nativeCudaHostCostSeconds;
    saveObj[QStringLiteral("ba_native_cuda_device_select_seconds")] = baResult.nativeCudaDeviceSelectSeconds;
    saveObj[QStringLiteral("ba_native_cuda_staging_seconds")] = baResult.nativeCudaStagingSeconds;
    saveObj[QStringLiteral("ba_native_cuda_release_seconds")] = baResult.nativeCudaReleaseSeconds;
    saveObj[QStringLiteral("ba_plamatrix_initial_cost")] = baResult.plaMatrixInitialCost;
    saveObj[QStringLiteral("ba_plamatrix_final_cost")] = baResult.plaMatrixFinalCost;
    saveObj[QStringLiteral("ba_plamatrix_accepted_steps")] = baResult.plaMatrixAcceptedSteps;
    saveObj[QStringLiteral("ba_plamatrix_rejected_steps")] = baResult.plaMatrixRejectedSteps;
    saveObj[QStringLiteral("ba_plamatrix_linearizations")] =
        baResult.plaMatrixLinearizations;
    saveObj[QStringLiteral("ba_plamatrix_objective_evaluations")] =
        baResult.plaMatrixObjectiveEvaluations;
    saveObj[QStringLiteral("ba_plamatrix_rejected_initial_tracks")] =
        baResult.plaMatrixRejectedInitialTracks;
    saveObj[QStringLiteral("ba_plamatrix_linear_solver")] =
        QString::fromStdString(baResult.plaMatrixLinearSolverName);
    saveObj[QStringLiteral("ba_plamatrix_device_name")] =
        QString::fromStdString(baResult.plaMatrixDeviceName);
    saveObj[QStringLiteral("ba_plamatrix_linear_iterations")] =
        baResult.plaMatrixLinearIterations;
    saveObj[QStringLiteral("ba_plamatrix_schur_pattern_builds")] =
        baResult.plaMatrixSchurPatternBuilds;
    saveObj[QStringLiteral("ba_plamatrix_schur_pattern_reuses")] =
        baResult.plaMatrixSchurPatternReuses;
    saveObj[QStringLiteral("ba_plamatrix_schur_assembly_on_device")] =
        baResult.plaMatrixSchurAssemblyOnDevice;
    saveObj[QStringLiteral("ba_plamatrix_mixed_precision_used")] =
        baResult.plaMatrixMixedPrecisionUsed;
    saveObj[QStringLiteral("ba_plamatrix_small_block_inverse_seconds")] =
        baResult.plaMatrixSmallBlockInverseSeconds;
    saveObj[QStringLiteral("ba_plamatrix_schur_accumulation_seconds")] =
        baResult.plaMatrixSchurAccumulationSeconds;
    saveObj[QStringLiteral("ba_plamatrix_csr_conversion_seconds")] =
        baResult.plaMatrixCsrConversionSeconds;
    saveObj[QStringLiteral("ba_plamatrix_schur_assembly_seconds")] =
        baResult.plaMatrixSchurAssemblySeconds;
    saveObj[QStringLiteral("ba_plamatrix_cholesky_factorization_seconds")] =
        baResult.plaMatrixCholeskyFactorizationSeconds;
    saveObj[QStringLiteral("ba_plamatrix_triangular_solve_seconds")] =
        baResult.plaMatrixTriangularSolveSeconds;
    saveObj[QStringLiteral("ba_plamatrix_residual_check_seconds")] =
        baResult.plaMatrixResidualCheckSeconds;
    saveObj[QStringLiteral("ba_plamatrix_linear_solve_seconds")] =
        baResult.plaMatrixLinearSolveSeconds;
    saveObj[QStringLiteral("ba_plamatrix_back_substitution_seconds")] =
        baResult.plaMatrixBackSubstitutionSeconds;

    // BA 选项回存（便于复现）
    {
        QJsonObject optObj;
        optObj[QStringLiteral("ba_backend")] =
            QString::fromLatin1(xjw::BundleAdjust::backendName(baOptions.backend));
        optObj[QStringLiteral("ba_cuda_device")] = baOptions.ceresCudaDevice;
        optObj[QStringLiteral("ba_plamatrix_device")] = baOptions.plaMatrixDevice;
        optObj[QStringLiteral("ba_min_cuda_cameras")] = baOptions.minPlaMatrixGpuCameras;
        optObj[QStringLiteral("ba_min_cuda_observations")] =
            baOptions.minPlaMatrixGpuObservations;
        optObj[QStringLiteral("ba_native_cuda_device")] = baOptions.nativeCudaDevice;
        optObj[QStringLiteral("ba_native_cuda_max_point_step")] =
            baOptions.nativeCudaMaxPointStepNorm;
        optObj[QStringLiteral("ba_min_cpu_observations")] = baOptions.minCeresCpuObservations;
        optObj[QStringLiteral("ba_max_ceres_point_only_observations")] =
            baOptions.maxCeresPointOnlyObservations;
        optObj[QStringLiteral("ba_max_ceres_initial_track_rms")] =
            baOptions.maxCeresInitialTrackRms;
        optObj[QStringLiteral("ba_max_dense_schur_cameras")] =
            baOptions.maxDenseSchurCameras;
        optObj[QStringLiteral("ba_max_sparse_schur_cameras")] =
            baOptions.maxSparseSchurCameras;
        optObj[QStringLiteral("ba_max_ceres_cuda_memory_fraction")] =
            baOptions.maxCeresCudaMemoryFraction;
        optObj[QStringLiteral("ba_allow_backend_fallback")] = baOptions.allowBackendFallback;
        optObj[QStringLiteral("ba_enable_backend_quality_gate")] = baOptions.enableBackendQualityGate;
        optObj[QStringLiteral("ba_max_accepted_rms_growth")] = baOptions.maxAcceptedRmsGrowth;
        optObj[QStringLiteral("ba_min_accepted_valid_track_ratio")] =
            baOptions.minAcceptedValidTrackRatio;
        optObj[QStringLiteral("ba_max_accepted_constraint_rms_growth")] =
            baOptions.maxAcceptedConstraintRmsGrowth;
        optObj[QStringLiteral("ba_compare_auto_backend_with_legacy")] =
            baOptions.compareAutoBackendWithLegacy;
        optObj[QStringLiteral("max_iterations")]       = opts.baOpt.maxIterations;
        optObj[QStringLiteral("max_point_iterations")] = opts.baOpt.maxPointIterations;
        optObj[QStringLiteral("max_camera_iterations")]= opts.baOpt.maxCameraIterations;
        optObj[QStringLiteral("refine_camera_pose")]   = opts.baOpt.refineCameraPose;
        optObj[QStringLiteral("huber_delta")]           = opts.baOpt.huberDelta;
        optObj[QStringLiteral("finite_diff_eps")]       = opts.baOpt.finiteDiffEps;
        optObj[QStringLiteral("damping")]               = opts.baOpt.damping;
        optObj[QStringLiteral("step_tolerance")]        = opts.baOpt.stepTolerance;
        optObj[QStringLiteral("enable_laser_constraints")] = opts.enableLaserConstraints;
        optObj[QStringLiteral("laser_constraint_cloud_path")] = opts.laserConstraintCloudPath;
        optObj[QStringLiteral("laser_association_max_distance_m")] = opts.laserAssociationMaxDistanceMeters;
        optObj[QStringLiteral("laser_voxel_size_m")] = opts.laserVoxelSizeMeters;
        optObj[QStringLiteral("laser_max_curvature")] = opts.laserMaxCurvature;
        optObj[QStringLiteral("laser_max_samples")] = opts.laserMaxSamples;
        optObj[QStringLiteral("laser_missing_normals_as_height_planes")] =
            opts.laserUseMissingNormalsAsHeightPlanes;
        optObj[QStringLiteral("laser_weight")] = opts.laserWeight;
        optObj[QStringLiteral("laser_sigma_m")] = opts.laserSigmaMeters;
        optObj[QStringLiteral("laser_effective_weight")] = effectiveLaserWeight;
        optObj[QStringLiteral("laser_huber_delta_m")] = opts.laserHuberDeltaMeters;
        optObj[QStringLiteral("enable_planetary_laser_range_constraints")] =
            opts.enablePlanetaryLaserRangeConstraints;
        optObj[QStringLiteral("planetary_laser_data_path")] = opts.planetaryLaserDataPath;
        optObj[QStringLiteral("planetary_laser_camera_coordinate_frame")] =
            opts.planetaryLaserCameraCoordinateFrame;
        optObj[QStringLiteral("planetary_laser_camera_sensor_frame")] =
            opts.planetaryLaserCameraSensorFrame;
        optObj[QStringLiteral("planetary_laser_confirm_unknown_sensor_is_frame")] =
            opts.planetaryLaserConfirmUnknownSensorIsFrame;
        optObj[QStringLiteral("planetary_laser_confirm_unknown_range_is_one_way")] =
            opts.planetaryLaserConfirmUnknownRangeIsOneWay;
        optObj[QStringLiteral("planetary_laser_allow_unmapped_shots")] =
            opts.planetaryLaserAllowUnmappedShots;
        optObj[QStringLiteral("planetary_laser_allow_unmapped_measured_images")] =
            opts.planetaryLaserAllowUnmappedMeasuredImages;
        QJsonArray planetaryLaserAliases;
        for (int cameraIndex = 0;
             cameraIndex < opts.planetaryLaserImageAliasesByCameraIndex.size();
             ++cameraIndex)
        {
            planetaryLaserAliases.append(QJsonObject{
                {QStringLiteral("camera_index"), cameraIndex},
                {QStringLiteral("aliases"), QJsonArray::fromStringList(
                     opts.planetaryLaserImageAliasesByCameraIndex.at(cameraIndex))},
            });
        }
        optObj[QStringLiteral("planetary_laser_image_aliases_by_camera_index")] =
            planetaryLaserAliases;
        optObj[QStringLiteral("planetary_laser_range_weight")] =
            opts.planetaryLaserRangeWeight;
        optObj[QStringLiteral("planetary_laser_range_huber_delta_sigma")] =
            opts.planetaryLaserRangeHuberDeltaSigma;
        optObj[QStringLiteral("export_observation_details")] = opts.exportObservationDetails;
        optObj[QStringLiteral("enable_control_point_constraints")] = baOptions.enableControlPointConstraints;
        optObj[QStringLiteral("control_point_weight")] = baOptions.controlPointWeight;
        optObj[QStringLiteral("control_point_huber_delta_m")] = baOptions.controlPointHuberDeltaMeters;
        optObj[QStringLiteral("enable_scale_bar_constraints")] = baOptions.enableScaleBarConstraints;
        optObj[QStringLiteral("scale_bar_weight")] = baOptions.scaleBarWeight;
        optObj[QStringLiteral("scale_bar_huber_delta_m")] = baOptions.scaleBarHuberDeltaMeters;
        optObj[QStringLiteral("enable_reference_terrain_prior")] = opts.enableReferenceTerrainPrior;
        optObj[QStringLiteral("reference_terrain_dem_path")] = opts.referenceTerrainDemPath;
        optObj[QStringLiteral("reference_terrain_sigma_m")] = opts.referenceTerrainSigmaMeters;
        optObj[QStringLiteral("reference_terrain_max_association_distance_m")] =
            opts.referenceTerrainMaxAssociationDistanceMeters;
        optObj[QStringLiteral("reference_terrain_huber_delta_m")] = opts.referenceTerrainHuberDeltaMeters;
        saveObj[QStringLiteral("options")] = optObj;
    }

    if (opts.enableReferenceTerrainPrior || !opts.referenceTerrainPriorSummary.isEmpty())
    {
        saveObj[QStringLiteral("reference_terrain_prior_summary")] =
            opts.referenceTerrainPriorSummary.isEmpty()
                ? QJsonObject{{QStringLiteral("enabled"), opts.enableReferenceTerrainPrior}}
                : opts.referenceTerrainPriorSummary;
    }

    if (opts.enableLaserConstraints)
    {
        QJsonObject laserSummary;
        laserSummary[QStringLiteral("enabled")] = true;
        laserSummary[QStringLiteral("cloud_path")] = opts.laserConstraintCloudPath;
        laserSummary[QStringLiteral("missing_normals_as_height_planes")] =
            opts.laserUseMissingNormalsAsHeightPlanes;
        laserSummary[QStringLiteral("map_sample_count")] = laserMapSampleCount;
        laserSummary[QStringLiteral("total_tracks")] = laserAssociationSummary.totalTracks;
        laserSummary[QStringLiteral("associated_tracks")] = laserAssociationSummary.associatedTracks;
        laserSummary[QStringLiteral("rejected_by_distance")] = laserAssociationSummary.rejectedByDistance;
        laserSummary[QStringLiteral("rejected_invalid_track")] = laserAssociationSummary.rejectedInvalidTrack;
        laserSummary[QStringLiteral("laser_constraint_count")] = baResult.laserConstraintCount;
        laserSummary[QStringLiteral("laser_rms_before_m")] = baResult.laserRmsBeforeMeters;
        laserSummary[QStringLiteral("laser_rms_after_m")] = baResult.laserRmsAfterMeters;
        laserSummary[QStringLiteral("laser_median_before_m")] = baResult.laserMedianBeforeMeters;
        laserSummary[QStringLiteral("laser_median_after_m")] = baResult.laserMedianAfterMeters;
        laserSummary[QStringLiteral("laser_sigma_m")] = opts.laserSigmaMeters;
        laserSummary[QStringLiteral("laser_effective_weight")] = effectiveLaserWeight;
        laserSummary[QStringLiteral("laser_huber_delta_m")] = opts.laserHuberDeltaMeters;
        saveObj[QStringLiteral("laser_constraints_summary")] = laserSummary;
    }

    if (opts.enablePlanetaryLaserRangeConstraints)
    {
        QJsonObject rangeSummary;
        rangeSummary[QStringLiteral("enabled")] = true;
        rangeSummary[QStringLiteral("mode")] = QStringLiteral("planetary_laser_range_shots");
        rangeSummary[QStringLiteral("data_path")] = opts.planetaryLaserDataPath;
        rangeSummary[QStringLiteral("source_format")] =
            planetaryLaserDataset.sourceFormat ==
                    xjw::lidar::PlanetaryLaserSourceFormat::IsisLidarDataJson
                ? QStringLiteral("isis_lidar_data_json")
                : QStringLiteral("plascan_si_json_v1");
        rangeSummary[QStringLiteral("target")] =
            QString::fromStdString(planetaryLaserDataset.reference.targetName);
        rangeSummary[QStringLiteral("body_fixed_frame")] =
            QString::fromStdString(planetaryLaserDataset.reference.bodyFixedFrame);
        rangeSummary[QStringLiteral("laser_frame")] =
            QString::fromStdString(planetaryLaserDataset.reference.laserFrame);
        rangeSummary[QStringLiteral("sensor_model")] = QString::fromLatin1(
            xjw::lidar::planetaryLaserSensorModelName(planetaryLaserDataset.sensorModel));
        rangeSummary[QStringLiteral("range_type")] = QString::fromLatin1(
            xjw::lidar::planetaryLaserRangeTypeName(planetaryLaserDataset.rangeType));
        rangeSummary[QStringLiteral("total_shots")] = planetaryLaserSummary.totalShots;
        rangeSummary[QStringLiteral("accepted_shots")] = planetaryLaserSummary.acceptedShots;
        rangeSummary[QStringLiteral("fixed_point_shots")] = planetaryLaserSummary.fixedPointShots;
        rangeSummary[QStringLiteral("constrained_point_shots")] =
            planetaryLaserSummary.constrainedPointShots;
        rangeSummary[QStringLiteral("free_point_shots")] = planetaryLaserSummary.freePointShots;
        rangeSummary[QStringLiteral("skipped_unmapped_shots")] =
            planetaryLaserSummary.skippedUnmappedShots;
        rangeSummary[QStringLiteral("measured_image_observations")] =
            planetaryLaserSummary.measuredImageObservations;
        rangeSummary[QStringLiteral("ignored_projected_measures")] =
            planetaryLaserSummary.ignoredProjectedMeasures;
        rangeSummary[QStringLiteral("ignored_unmapped_measured_images")] =
            planetaryLaserSummary.ignoredUnmappedMeasuredImages;
        rangeSummary[QStringLiteral("range_constraint_count")] =
            baResult.laserRangeConstraintCount;
        rangeSummary[QStringLiteral("range_rms_before_m")] =
            baResult.laserRangeRmsBeforeMeters;
        rangeSummary[QStringLiteral("range_rms_after_m")] =
            baResult.laserRangeRmsAfterMeters;
        rangeSummary[QStringLiteral("range_weight")] = opts.planetaryLaserRangeWeight;
        rangeSummary[QStringLiteral("range_huber_delta_sigma")] =
            opts.planetaryLaserRangeHuberDeltaSigma;

        QJsonArray shotResults;
        for (std::size_t shotIndex = 0;
             shotIndex < baResult.laserRangeShots.size() &&
             shotIndex < baOptions.laserRangeConstraints.size();
             ++shotIndex)
        {
            const xjw::BARefinedLaserRangeShot &shot = baResult.laserRangeShots[shotIndex];
            const xjw::BALaserRangeConstraint &constraint =
                baOptions.laserRangeConstraints[shotIndex];
            QJsonObject shotObject;
            shotObject[QStringLiteral("id")] = QString::fromStdString(shot.shotId);
            shotObject[QStringLiteral("source_index")] = shot.sourceIndex;
            shotObject[QStringLiteral("ephemeris_time_s")] = shot.ephemerisTimeSeconds;
            shotObject[QStringLiteral("valid")] = shot.valid;
            shotObject[QStringLiteral("point_mode")] =
                shot.pointMode == xjw::BALaserPointMode::Constrained
                    ? QStringLiteral("constrained")
                    : (shot.pointMode == xjw::BALaserPointMode::Free
                           ? QStringLiteral("free")
                           : QStringLiteral("fixed"));
            shotObject[QStringLiteral("camera_index")] = constraint.cameraIndex;
            if (constraint.cameraIndex >= 0 &&
                constraint.cameraIndex < planetaryLaserCameraPaths.size())
            {
                shotObject[QStringLiteral("camera_image")] =
                    planetaryLaserCameraPaths[constraint.cameraIndex];
            }
            shotObject[QStringLiteral("observed_range_m")] = constraint.observedRangeMeters;
            shotObject[QStringLiteral("range_sigma_m")] = constraint.sigmaRangeMeters;
            shotObject[QStringLiteral("lever_arm_camera_m")] = QJsonArray{
                constraint.leverArmCameraMeters[0],
                constraint.leverArmCameraMeters[1],
                constraint.leverArmCameraMeters[2]};
            shotObject[QStringLiteral("computed_range_before_m")] =
                shot.computedRangeBeforeMeters;
            shotObject[QStringLiteral("computed_range_after_m")] =
                shot.computedRangeAfterMeters;
            shotObject[QStringLiteral("residual_before_m")] = shot.residualBeforeMeters;
            shotObject[QStringLiteral("residual_after_m")] = shot.residualAfterMeters;
            shotObject[QStringLiteral("normalized_residual_after")] =
                shot.normalizedResidualAfter;
            shotObject[QStringLiteral("refined_point_m")] = QJsonArray{
                shot.point[0], shot.point[1], shot.point[2]};
            shotObject[QStringLiteral("measured_image_observation_count")] =
                static_cast<int>(constraint.measuredImageObservations.size());
            shotResults.append(shotObject);
        }
        rangeSummary[QStringLiteral("shots")] = shotResults;
        saveObj[QStringLiteral("planetary_laser_range_summary")] = rangeSummary;
    }

    if (baOptions.enableControlPointConstraints || baResult.controlPointConstraintCount > 0)
    {
        QJsonObject controlSummary;
        controlSummary[QStringLiteral("enabled")] = baOptions.enableControlPointConstraints;
        controlSummary[QStringLiteral("control_point_constraint_count")] = baResult.controlPointConstraintCount;
        controlSummary[QStringLiteral("control_point_rms_before_m")] = baResult.controlPointRmsBeforeMeters;
        controlSummary[QStringLiteral("control_point_rms_after_m")] = baResult.controlPointRmsAfterMeters;
        controlSummary[QStringLiteral("control_point_weight")] = baOptions.controlPointWeight;
        controlSummary[QStringLiteral("control_point_huber_delta_m")] = baOptions.controlPointHuberDeltaMeters;
        saveObj[QStringLiteral("control_point_constraints_summary")] = controlSummary;
    }

    if (baOptions.enableScaleBarConstraints || baResult.scaleBarConstraintCount > 0)
    {
        QJsonObject scaleBarSummary;
        scaleBarSummary[QStringLiteral("enabled")] = baOptions.enableScaleBarConstraints;
        scaleBarSummary[QStringLiteral("scale_bar_constraint_count")] = baResult.scaleBarConstraintCount;
        scaleBarSummary[QStringLiteral("scale_bar_rms_before_m")] = baResult.scaleBarRmsBeforeMeters;
        scaleBarSummary[QStringLiteral("scale_bar_rms_after_m")] = baResult.scaleBarRmsAfterMeters;
        scaleBarSummary[QStringLiteral("scale_bar_weight")] = baOptions.scaleBarWeight;
        scaleBarSummary[QStringLiteral("scale_bar_huber_delta_m")] = baOptions.scaleBarHuberDeltaMeters;
        saveObj[QStringLiteral("scale_bar_constraints_summary")] = scaleBarSummary;
    }

    if (!opts.markerTrackQualityInputs.isEmpty() || !opts.markerScaleBarQualityInputs.isEmpty())
    {
        control_points::ControlNetworkResult marker_network;
        marker_network.ok = true;
        QJsonArray marker_details;
        for (const BaServiceOptions::MarkerTrackQualityInput &input : opts.markerTrackQualityInputs)
        {
            if (input.trackIndex < 0
                || input.trackIndex >= static_cast<int>(baResult.points.size()))
            {
                continue;
            }
            const std::array<double, 3> &point =
                baResult.points[static_cast<std::size_t>(input.trackIndex)].point;
            control_points::MarkerResidual residual;
            residual.markerId = input.markerId.toStdString();
            residual.role = input.role;
            double sigma_sum_squared = 0.0;
            for (int axis = 0; axis < 3; ++axis)
            {
                residual.delta[axis] = point[axis] - input.referencePoint[axis];
                residual.total += residual.delta[axis] * residual.delta[axis];
                sigma_sum_squared += input.sigma[axis] * input.sigma[axis];
            }
            residual.total = std::sqrt(residual.total);
            const double sigma = std::sqrt(sigma_sum_squared / 3.0);
            residual.normalized = residual.total / std::max(1.0e-9, sigma);
            residual.inlier = input.usedAsConstraint;
            if (input.role == control_points::MarkerRole::ControlPoint)
            {
                marker_network.controlResiduals.push_back(residual);
            }
            else if (input.role == control_points::MarkerRole::CheckPoint)
            {
                marker_network.checkPointResiduals.push_back(residual);
            }

            QJsonObject detail;
            detail[QStringLiteral("marker_id")] = input.markerId;
            detail[QStringLiteral("role")] = input.role == control_points::MarkerRole::ControlPoint
                ? QStringLiteral("control") : QStringLiteral("check");
            detail[QStringLiteral("track_index")] = input.trackIndex;
            detail[QStringLiteral("used_as_constraint")] = input.usedAsConstraint;
            detail[QStringLiteral("dx")] = residual.delta[0];
            detail[QStringLiteral("dy")] = residual.delta[1];
            detail[QStringLiteral("dz")] = residual.delta[2];
            detail[QStringLiteral("total")] = residual.total;
            detail[QStringLiteral("normalized")] = residual.normalized;
            marker_details.append(detail);
        }

        QVector<control_points::ScaleBarResidual> scale_residuals;
        QJsonArray scale_details;
        for (const BaServiceOptions::MarkerScaleBarQualityInput &input :
             opts.markerScaleBarQualityInputs)
        {
            if (input.trackIndexA < 0 || input.trackIndexB < 0
                || input.trackIndexA >= static_cast<int>(baResult.points.size())
                || input.trackIndexB >= static_cast<int>(baResult.points.size()))
            {
                continue;
            }
            const auto &first = baResult.points[static_cast<std::size_t>(input.trackIndexA)].point;
            const auto &second = baResult.points[static_cast<std::size_t>(input.trackIndexB)].point;
            const double dx = first[0] - second[0];
            const double dy = first[1] - second[1];
            const double dz = first[2] - second[2];
            const double estimated = std::sqrt(dx * dx + dy * dy + dz * dz);
            scale_residuals.push_back({input.scaleBarId,
                                       input.role,
                                       input.measuredDistance,
                                       estimated,
                                       estimated - input.measuredDistance});
            QJsonObject detail;
            detail[QStringLiteral("scale_bar_id")] = input.scaleBarId;
            detail[QStringLiteral("role")] = input.role == control_points::ScaleBarRole::Control
                ? QStringLiteral("control") : QStringLiteral("check");
            detail[QStringLiteral("measured_distance")] = input.measuredDistance;
            detail[QStringLiteral("estimated_distance")] = estimated;
            detail[QStringLiteral("residual")] = estimated - input.measuredDistance;
            scale_details.append(detail);
        }

        const control_points::MarkerQualityReport marker_report =
            control_points::buildMarkerQualityReport(marker_network, scale_residuals);
        const auto statistics_json = [](const control_points::ResidualStatistics &statistics)
        {
            return QJsonObject{
                {QStringLiteral("count"), statistics.totalCount},
                {QStringLiteral("inlier_count"), statistics.inlierCount},
                {QStringLiteral("mean"), statistics.mean},
                {QStringLiteral("rms"), statistics.rms},
                {QStringLiteral("maximum"), statistics.maximum},
            };
        };
        QJsonObject quality;
        quality[QStringLiteral("controls")] = statistics_json(marker_report.controls);
        quality[QStringLiteral("check_points")] = statistics_json(marker_report.checkPoints);
        quality[QStringLiteral("control_scale_bars")] =
            statistics_json(marker_report.controlScaleBars);
        quality[QStringLiteral("check_scale_bars")] =
            statistics_json(marker_report.checkScaleBars);
        quality[QStringLiteral("marker_residuals")] = marker_details;
        quality[QStringLiteral("scale_bar_residuals")] = scale_details;
        saveObj[QStringLiteral("marker_quality_report")] = quality;
    }

    // ── 点位精度统计写入 JSON 数组 ─────────────────────────────────────────
    QJsonArray pointsArr;
    for (int i = 0; i < static_cast<int>(baResult.points.size()); ++i)
    {
        const auto& p = baResult.points.at(static_cast<size_t>(i));
        QJsonObject one;
        one[QStringLiteral("index")]     = i;
        one[QStringLiteral("valid")]     = p.valid;
        one[QStringLiteral("converged")] = p.converged;
        one[QStringLiteral("iterations")]= p.iterations;
        one[QStringLiteral("rms_before")]= p.rmsBefore;
        one[QStringLiteral("rms_after")] = p.rmsAfter;
        one[QStringLiteral("track_len")] = static_cast<int>(tracks[static_cast<size_t>(i)].observations.size());
        QJsonArray xyz;
        xyz.append(p.point[0]);
        xyz.append(p.point[1]);
        xyz.append(p.point[2]);
        one[QStringLiteral("point_xyz")] = xyz;
        QJsonArray observations;
        if (opts.exportObservationDetails)
        {
            for (const BAObservation &observation : tracks[static_cast<size_t>(i)].observations)
            {
                if (observation.cameraIndex < 0
                    || observation.cameraIndex >= static_cast<int>(cameras.size())
                    || observation.cameraIndex >= opts.imagePathByIndex.size())
                {
                    continue;
                }
                const FramePinholeCamera &camera = observation.cameraIndex < static_cast<int>(baResult.refinedCameras.size())
                    ? baResult.refinedCameras[static_cast<size_t>(observation.cameraIndex)]
                    : cameras[static_cast<size_t>(observation.cameraIndex)];
                double projected[2] = {0.0, 0.0};
                const bool projectedOk = camera.projectWorldPoint(p.point.data(), projected)
                    || camera.projectWorldPointSigned(p.point.data(), projected);
                if (!projectedOk || !std::isfinite(projected[0]) || !std::isfinite(projected[1]))
                {
                    continue;
                }
                const double residualX = projected[0] - observation.u;
                const double residualY = projected[1] - observation.v;
                QJsonObject observationObject;
                observationObject[QStringLiteral("image_id")] = observation.cameraIndex;
                observationObject[QStringLiteral("image_path")] =
                    opts.imagePathByIndex.at(observation.cameraIndex);
                observationObject[QStringLiteral("xy")] = QJsonArray{observation.u, observation.v};
                observationObject[QStringLiteral("projected_xy")] = QJsonArray{projected[0], projected[1]};
                observationObject[QStringLiteral("residual_xy")] = QJsonArray{residualX, residualY};
                observationObject[QStringLiteral("residual_norm_px")] = std::hypot(residualX, residualY);
                observations.append(observationObject);
            }
        }
        one[QStringLiteral("observations")] = observations;
        pointsArr.append(one);
    }
    saveObj[QStringLiteral("points")] = pointsArr;

    // ── 逐相机统计：位移量 + 欧拉角变化 + 每台相机 RMS ───────────────────
    // 同时构建：
    //   - pendingCamUpdates：待确认写入项目的相机 JSON（由调用方决定是否应用）
    //   - cameraPreview：GUI 预览列表（显示给用户确认）
    //   - refinedCameras：输出到 JSON 的精化相机表
    QMap<QString, QJsonObject> pendingCamUpdates;
    QJsonArray refinedCameras;
    QJsonArray cameraPreview;

    // 打开相机 CSV 文件（若需要导出）
    QFile cameraCsv(camerasCsvPath);
    const bool csvOpened = opts.exportCameraCsv
        && cameraCsv.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);
    QTextStream cs(&cameraCsv);
    if (csvOpened)
    {
        cs << "image_path,image_name,delta_c_m"
              ",yaw_before,yaw_after,pitch_before,pitch_after,roll_before,roll_after"
              ",mean_rms_before,mean_rms_after\n";
    }

    for (size_t i = 0;
         i < baResult.refinedCameras.size() && i < static_cast<size_t>(opts.imagePathByIndex.size());
         ++i)
    {
        const auto&   camBefore = cameras[i];
        const auto&   camAfter  = baResult.refinedCameras[i];
        const QString imgPath   = opts.imagePathByIndex.at(static_cast<int>(i));
        const QString imgName   = QFileInfo(imgPath).fileName();

        // 计算相机中心三维位移量（单位：与输入坐标系相同，通常为米）
        const auto   c0 = camBefore.cameraCenter();
        const auto   c1 = camAfter.cameraCenter();
        const double dC = std::sqrt(
            (c1[0]-c0[0])*(c1[0]-c0[0]) +
            (c1[1]-c0[1])*(c1[1]-c0[1]) +
            (c1[2]-c0[2])*(c1[2]-c0[2]));

        const QJsonObject beforeJson  = opts.beforeCamMeta.value(imgPath);
        const QJsonObject afterJson = xjw::common::project::cameraToJson(camAfter);

        // 收集待提交的相机更新
        pendingCamUpdates.insert(imgPath, afterJson);

        // 从 JSON 读取欧拉角（BA 前，已在原始文件中计算存储）
        const double yawBefore   = beforeJson.value(QStringLiteral("yaw_deg")).toDouble();
        const double pitchBefore = beforeJson.value(QStringLiteral("pitch_deg")).toDouble();
        const double rollBefore  = beforeJson.value(QStringLiteral("roll_deg")).toDouble();
        // BA 后欧拉角直接从序列化 JSON 读取（避免重复计算）
        const double yawAfter    = afterJson.value(QStringLiteral("yaw_deg")).toDouble();
        const double pitchAfter  = afterJson.value(QStringLiteral("pitch_deg")).toDouble();
        const double rollAfter   = afterJson.value(QStringLiteral("roll_deg")).toDouble();

        // ── 计算该相机参与的观测对应的 RMS（小于全局 RMS 时还可发现坏相机）──
        double beforeSum2 = 0.0, afterSum2 = 0.0;
        int    obsCnt     = 0;
        for (int t = 0;
             t < static_cast<int>(tracks.size()) && t < static_cast<int>(baResult.points.size());
             ++t)
        {
            const auto& tr = tracks[static_cast<size_t>(t)];
            const auto& pt = baResult.points[static_cast<size_t>(t)];
            if (!pt.valid) continue;

            for (const auto& obs : tr.observations)
            {
                if (obs.cameraIndex != static_cast<int>(i)) continue;

                const double world[3] = {pt.point[0], pt.point[1], pt.point[2]};
                double uv0[2] = {0.0, 0.0}, uv1[2] = {0.0, 0.0};

                if (camBefore.projectWorldPoint(world, uv0))
                {
                    const double du = uv0[0] - obs.u;
                    const double dv = uv0[1] - obs.v;
                    beforeSum2 += du*du + dv*dv;
                }
                if (camAfter.projectWorldPoint(world, uv1))
                {
                    const double du = uv1[0] - obs.u;
                    const double dv = uv1[1] - obs.v;
                    afterSum2 += du*du + dv*dv;
                }
                obsCnt += 2;
            }
        }
        const double rmsBefore = (obsCnt > 0) ? std::sqrt(beforeSum2 / obsCnt) : 0.0;
        const double rmsAfter  = (obsCnt > 0) ? std::sqrt(afterSum2  / obsCnt) : 0.0;

        // 写入 CSV 行
        if (csvOpened)
        {
            cs << '"' << imgPath << "\",\"" << imgName << "\"," << dC
               << ',' << yawBefore   << ',' << yawAfter
               << ',' << pitchBefore << ',' << pitchAfter
               << ',' << rollBefore  << ',' << rollAfter
               << ',' << rmsBefore   << ',' << rmsAfter << "\n";
        }

        // 构建 GUI 预览条目
        QJsonObject preview;
        preview[QStringLiteral("image_path")]    = imgPath;
        preview[QStringLiteral("image_name")]    = imgName;
        preview[QStringLiteral("delta_c_m")]     = dC;
        preview[QStringLiteral("yaw_before")]    = yawBefore;
        preview[QStringLiteral("yaw_after")]     = yawAfter;
        preview[QStringLiteral("pitch_before")]  = pitchBefore;
        preview[QStringLiteral("pitch_after")]   = pitchAfter;
        preview[QStringLiteral("roll_before")]   = rollBefore;
        preview[QStringLiteral("roll_after")]    = rollAfter;
        preview[QStringLiteral("mean_rms_before")] = rmsBefore;
        preview[QStringLiteral("mean_rms_after")]  = rmsAfter;
        cameraPreview.append(preview);

        // 导出精化后的 .tsai 相机文件（供外部工具验证）
        QString tsaiPath;
        if (opts.exportTsai)
        {
            tsaiPath = QDir(tsaiDir).filePath(
                QFileInfo(imgPath).completeBaseName() + QStringLiteral(".ba.tsai"));
            camAfter.saveToFile(xjw::common::io::toUtf8Path(tsaiPath));
        }

        // 追加到 JSON 精化相机列表
        QJsonObject one;
        one[QStringLiteral("index")]      = static_cast<int>(i);
        one[QStringLiteral("image_path")] = imgPath;
        one[QStringLiteral("tsai_path")]  = tsaiPath;
        one[QStringLiteral("camera")]     = afterJson;
        refinedCameras.append(one);
    }

    if (csvOpened)
    {
        cameraCsv.close();
    }

    // ── 点位精度 CSV ───────────────────────────────────────────────────────
    if (opts.exportPointsCsv)
    {
        QFile ptsFile(pointsCsvPath);
        if (ptsFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
        {
            QTextStream ps(&ptsFile);
            ps << "index,valid,converged,iterations,rms_before,rms_after,improve,dx,dy,dz\n";
            for (int i = 0; i < static_cast<int>(baResult.points.size()); ++i)
            {
                const auto& p    = baResult.points.at(static_cast<size_t>(i));
                const auto& init = tracks.at(static_cast<size_t>(i)).initialPoint;
                ps << i << ',' << (p.valid ? 1 : 0) << ',' << (p.converged ? 1 : 0) << ','
                   << p.iterations << ',' << p.rmsBefore << ',' << p.rmsAfter << ','
                   << (p.rmsBefore - p.rmsAfter) << ','
                   << (p.point[0] - init[0]) << ','
                   << (p.point[1] - init[1]) << ','
                   << (p.point[2] - init[2]) << "\n";
            }
            ptsFile.close();
        }
    }

    // ── 文字摘要报告 ───────────────────────────────────────────────────────
    if (opts.exportSummaryTxt)
    {
        QFile sumFile(summaryTxtPath);
        if (sumFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
        {
            QTextStream ts(&sumFile);
            ts << "光束法平差 - 运行摘要\n";
            ts << "==============================\n";
            ts << "输出目录: "       << outDir                         << "\n";
            ts << "相机数量: "       << cameras.size()                  << "\n";
            ts << "轨迹总数: "       << baResult.totalTracks            << "\n";
            ts << "有效优化轨迹: "   << baResult.optimizedTracks        << "\n";
            ts << "有效轨迹比例: "   << baResult.validTrackRatio        << "\n";
            ts << "平均 RMS（前）: " << baResult.meanRmsBefore          << "\n";
            ts << "平均 RMS（后）: " << baResult.meanRmsAfter           << "\n";
            ts << "请求 BA 后端: "   << QString::fromLatin1(xjw::BundleAdjust::backendName(baResult.requestedBackend))
               << "\n";
            ts << "实际 BA 后端: "   << QString::fromLatin1(xjw::BundleAdjust::backendName(baResult.usedBackend))
               << "\n";
            ts << "实际使用 GPU: "   << (baResult.usedGpu ? QStringLiteral("是") : QStringLiteral("否"))
               << "\n";
            ts << "Ceres 线性求解器: " << QString::fromStdString(baResult.ceresLinearSolverName) << "\n";
            ts << "Ceres CUDA 显存估算/空闲(bytes): "
               << baResult.ceresEstimatedCudaBytes << "/"
               << baResult.ceresCudaFreeBytes << "\n";
            ts << "Ceres 目标函数: " << baResult.ceresInitialCost << " -> "
               << baResult.ceresFinalCost << "\n";
            ts << "Ceres 接受/拒绝步: " << baResult.ceresSuccessfulSteps << "/"
               << baResult.ceresUnsuccessfulSteps << "\n";
            ts << "Ceres 初始粗差剔除 track: " << baResult.ceresRejectedInitialTracks << "\n";
            ts << "观测数量: "       << baResult.observationCount          << "\n";
            ts << "native CUDA 活动相机/轨迹/观测: "
               << baResult.nativeCudaActiveCameras << "/"
               << baResult.nativeCudaActiveTracks << "/"
               << baResult.nativeCudaActiveObservations << "\n";
            ts << "native CUDA 点块代价: " << baResult.nativeCudaInitialCost
               << " -> " << baResult.nativeCudaFinalCost
               << ", 接受步: " << baResult.nativeCudaAcceptedSteps
               << ", 拒绝步: " << baResult.nativeCudaRejectedSteps << "\n";
            ts << "native CUDA 分解耗时(s): 上传 " << baResult.nativeCudaUploadSeconds
               << ", kernel " << baResult.nativeCudaKernelSeconds
               << ", 下载 " << baResult.nativeCudaDownloadSeconds
               << ", host cost " << baResult.nativeCudaHostCostSeconds
               << ", staging " << baResult.nativeCudaStagingSeconds
               << ", release " << baResult.nativeCudaReleaseSeconds << "\n";
            ts << "后端总耗时(s): "  << baResult.totalSeconds             << "\n";
            ts << "问题构建耗时(s): " << baResult.setupSeconds             << "\n";
            ts << "求解耗时(s): "    << baResult.solveSeconds             << "\n";
            ts << "质量检查耗时(s): " << baResult.postprocessSeconds       << "\n";
            if (opts.enablePlanetaryLaserRangeConstraints)
            {
                ts << "行星激光测距 shot: " << baResult.laserRangeConstraintCount << "\n";
                ts << "行星激光 range RMS(m): "
                   << baResult.laserRangeRmsBeforeMeters << " -> "
                   << baResult.laserRangeRmsAfterMeters << "\n";
                ts << "行星激光目标/坐标系: "
                   << QString::fromStdString(planetaryLaserDataset.reference.targetName)
                   << "/"
                   << QString::fromStdString(planetaryLaserDataset.reference.bodyFixedFrame)
                   << "\n";
            }
            if (!baResult.backendSelectionReason.empty())
            {
                ts << "后端选择说明: " << QString::fromUtf8(baResult.backendSelectionReason.c_str()) << "\n";
            }
            if (baResult.qualityGateRejected || !baResult.qualityGateMessage.empty())
            {
                ts << "质量门控: "
                   << (baResult.qualityGateRejected ? QStringLiteral("拒绝候选后端") : QStringLiteral("通过"))
                   << "\n";
                if (!baResult.qualityGateMessage.empty())
                {
                    ts << "质量门控说明: " << QString::fromUtf8(baResult.qualityGateMessage.c_str()) << "\n";
                }
            }
            if (baResult.backendFallback)
            {
                ts << "后端回退: 是\n";
                ts << "回退说明: " << QString::fromUtf8(baResult.backendMessage.c_str()) << "\n";
            }
            ts << "\n主要输出文件：\n";
            ts << "  " << summaryTxtPath  << "\n";
            ts << "  " << pointsCsvPath   << "\n";
            ts << "  " << camerasCsvPath  << "\n";
            ts << "  " << runJsonPath     << "\n";
            sumFile.close();
        }
    }

    // ── 汇总文件路径字段 ───────────────────────────────────────────────────
    saveObj[QStringLiteral("refined_cameras")]  = refinedCameras;
    saveObj[QStringLiteral("camera_preview")]   = cameraPreview;

    QJsonObject filesObj;
    filesObj[QStringLiteral("summary_txt")] = summaryTxtPath;
    filesObj[QStringLiteral("points_csv")]  = pointsCsvPath;
    filesObj[QStringLiteral("camera_csv")]  = camerasCsvPath;
    filesObj[QStringLiteral("run_json")]    = runJsonPath;
    filesObj[QStringLiteral("tsai_dir")]    = tsaiDir;
    // 导出标志回写（便于结果复现校验）
    filesObj[QStringLiteral("export_tsai")]        = opts.exportTsai;
    filesObj[QStringLiteral("export_summary_txt")] = opts.exportSummaryTxt;
    filesObj[QStringLiteral("export_points_csv")]  = opts.exportPointsCsv;
    filesObj[QStringLiteral("export_camera_csv")]  = opts.exportCameraCsv;
    filesObj[QStringLiteral("export_run_json")]    = opts.exportRunJson;
    filesObj[QStringLiteral("export_eval_plot")]   = opts.exportEvalPlot;

    // ── 生成评估图 ─────────────────────────────────────────────────────────
    if (opts.exportEvalPlot)
    {
        generateEvalPlots(baResult, cameraPreview, outDir, &filesObj);
    }

    saveObj[QStringLiteral("files")] = filesObj;

    // ── 写入 JSON 文件 ─────────────────────────────────────────────────────
    if (opts.exportRunJson)
    {
        QFile runJson(runJsonPath);
        if (runJson.open(QIODevice::WriteOnly | QIODevice::Truncate))
        {
            runJson.write(QJsonDocument(saveObj).toJson(QJsonDocument::Indented));
            runJson.close();
        }
    }

    // ── 组装并返回结果 ─────────────────────────────────────────────────────
    const bool missingActiveLaserConstraints =
        opts.enableLaserConstraints && baResult.laserConstraintCount <= 0;
    const bool missingActivePlanetaryLaserConstraints =
        opts.enablePlanetaryLaserRangeConstraints &&
        baResult.laserRangeConstraintCount <= 0;
    result.success = baResult.solutionUsable &&
                     !missingActiveLaserConstraints &&
                     !missingActivePlanetaryLaserConstraints;
    if (result.success)
    {
        result.pendingCamUpdates = pendingCamUpdates;
    }
    else if (missingActiveLaserConstraints)
    {
        result.errorMessage = QStringLiteral(
            "LiDAR 约束在求解或质量过滤后全部失效，未写回相机；"
            "请检查坐标系、关联距离和影像匹配粗差");
    }
    else if (missingActivePlanetaryLaserConstraints)
    {
        result.errorMessage = QStringLiteral(
            "行星激光测距 shot 在求解中全部失效，未写回相机；"
            "请检查 frame camera 声明、同期影像映射、坐标系、range 和 sigma");
    }
    else
    {
        result.errorMessage = QStringLiteral("BA 求解结果不可写回（%1）: %2")
                                  .arg(QString::fromLatin1(
                                           xjw::BundleAdjust::solveStatusName(
                                               baResult.solveStatus)),
                                       QString::fromUtf8(baResult.backendMessage.c_str()));
    }
    result.resultJson        = saveObj;
    return result;
}

} // namespace gui
} // namespace xjw
