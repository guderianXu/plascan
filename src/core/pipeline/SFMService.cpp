// =============================================================================
// 文件名: SFMService.cpp
// 描述:   增量式 SFM 一站式服务实现，详细说明见 SFMService.h。
//
//         全自动流水线：SuperPoint 特征提取 → SuperGlue 匹配 → 增量式 SFM
//         本文件不依赖任何 Qt Widget，所有 GUI 提示由调用方负责。
// =============================================================================

// ── LibTorch / OpenCV 头文件必须在 Qt 头文件之前引入，避免宏冲突 ────────
#include "compat/QtTorchMacroGuard.h"

#include "SuperPoint.h"
#include "FeatureData.h"
#include "FeatureFileIO.h"
#include "SuperGlueMatcher.h"
#include "SuperGlueMatchIO.h"
#include "MatchOutlierRejector.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// ── 项目 / 服务头文件 ──────────────────────────────────────────────────────
#include "SFMService.h"
#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "Logger.h"
#include "Camera.h"
#include "pipeline/IncrementalSfm.h"
#include "common/SfmTypes.h"
#include "data/PointCloud.h"
#include "io/PointCloudIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTextStream>
#include <QDateTime>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QSet>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>

namespace xjw
{
namespace gui
{

// ══════════════════════════════════════════════════════════════════════════════
// 匿名命名空间：文件内部辅助工具
// ══════════════════════════════════════════════════════════════════════════════
namespace
{

// ── 路径工具 ─────────────────────────────────────────────────────────────────

QString normalizePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

QString canonicalPairKey(const QString &pathA, const QString &pathB)
{
    const QString normA = normalizePath(pathA);
    const QString normB = normalizePath(pathB);
    if (normA.isEmpty() || normB.isEmpty() || normA == normB)
    {
        return QString();
    }
    return (normA < normB)
        ? (normA + QStringLiteral("\n") + normB)
        : (normB + QStringLiteral("\n") + normA);
}

QJsonArray ipmatchResultsFromMeta(const QJsonObject &meta)
{
    const QJsonArray direct = meta.value(QStringLiteral("ipmatch_results")).toArray();
    if (!direct.isEmpty())
    {
        return direct;
    }

    const QJsonObject resultsObj = meta.value(QStringLiteral("project_results")).toObject();
    if (!resultsObj.isEmpty())
    {
        return resultsObj.value(QStringLiteral("ipmatch_results")).toArray();
    }

    return QJsonArray();
}

QString resolvePathFromProjectRoot(const QString &projectRoot, const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }

    if (QDir::isAbsolutePath(trimmed))
    {
        return QDir::cleanPath(trimmed);
    }

    if (projectRoot.isEmpty())
    {
        return QDir::cleanPath(trimmed);
    }

    return QDir(projectRoot).filePath(trimmed);
}

QString resolveImagePathTokenFromMeta(const QString &token, const QJsonObject &projectMeta)
{
    const QString trimmed = token.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }

    const QString resolved = xjw::gui::project::resolveProjectImagePathFromToken(trimmed, projectMeta);
    if (!resolved.isEmpty())
    {
        return normalizePath(resolved);
    }

    return normalizePath(trimmed);
}

QString canonicalNamePairKey(const QString &nameA, const QString &nameB)
{
    const QString a = nameA.trimmed().toLower();
    const QString b = nameB.trimmed().toLower();
    if (a.isEmpty() || b.isEmpty() || a == b)
    {
        return QString();
    }
    return (a < b) ? (a + QStringLiteral("\n") + b) : (b + QStringLiteral("\n") + a);
}

// ── 相机 JSON 序列化 ─────────────────────────────────────────────────────────

QJsonObject cameraToJson(const Camera &camera)
{
    const auto intrinsics = camera.intrinsics();
    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();

    QJsonObject obj;
    obj[QStringLiteral("model")]       = QStringLiteral("tsai");
    obj[QStringLiteral("imported_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    obj[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
    obj[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    obj[QStringLiteral("pitch")] = camera.pixelPitch();
    obj[QStringLiteral("fu")] = camera.focalXMillimeters();
    obj[QStringLiteral("fv")] = camera.focalYMillimeters();
    obj[QStringLiteral("cu")] = camera.principalXMillimeters();
    obj[QStringLiteral("cv")] = camera.principalYMillimeters();
    obj[QStringLiteral("u_direction")] = intrinsics.uAxisSign;
    obj[QStringLiteral("v_direction")] = intrinsics.vAxisSign;
    obj[QStringLiteral("depth_axis_flipped")] = camera.depthAxisFlipped();

    QJsonArray cArr;
    cArr.append(center[0]);
    cArr.append(center[1]);
    cArr.append(center[2]);
    obj[QStringLiteral("C")] = cArr;

    QJsonArray rArr;
    for (int i = 0; i < 9; ++i)
    {
        rArr.append(rotation[i]);
    }
    obj[QStringLiteral("R")] = rArr;

    // 欧拉角（供显示用）
    const double r00 = rotation[0], r10 = rotation[3], r20 = rotation[6];
    const double r21 = rotation[7], r22 = rotation[8];
    const double pitch = std::asin(std::clamp(-r20, -1.0, 1.0));
    double yaw = 0.0, roll = 0.0;
    if (std::abs(std::cos(pitch)) > 1e-8)
    {
        yaw  = std::atan2(r10, r00);
        roll = std::atan2(r21, r22);
    }
    else
    {
        yaw = std::atan2(-rotation[1], rotation[4]);
    }
    constexpr double kRad2Deg = 180.0 / M_PI;
    obj[QStringLiteral("yaw_deg")]   = yaw   * kRad2Deg;
    obj[QStringLiteral("pitch_deg")] = pitch * kRad2Deg;
    obj[QStringLiteral("roll_deg")]  = roll  * kRad2Deg;

    return obj;
}

// ── 模型文件查找 ─────────────────────────────────────────────────────────────

QString findModelFile(const QString &modelName)
{
    QStringList candidates;

    const QString envModelDir = qEnvironmentVariable("PLASCAN_MODEL_DIR").trimmed();
    if (!envModelDir.isEmpty())
    {
        candidates.append(QDir(envModelDir).filePath(modelName));
    }

#ifdef PLASCAN_SOURCE_DIR
    candidates.append(
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("resources/models/%1").arg(modelName)));
#endif

    const QString exeDir = QCoreApplication::applicationDirPath();
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../models/%1").arg(modelName)));
    candidates.append(QStringLiteral("models/%1").arg(modelName));

    for (const QString &candidate : candidates)
    {
        if (QFile::exists(candidate))
        {
            return candidate;
        }
    }

    return QString();
}

// ── 精度等级参数预设 ─────────────────────────────────────────────────────────

struct QualityPresets
{
    // SFM
    int   initMinNumMatches;
    int   initMinNumInliers;
    int   localBAInterval;
    int   globalBAInterval;
    // SuperPoint
    float spDetectionThreshold;   ///< 检测阈值，越低特征点越多
    int   spMaxKeypoints;         ///< 最大关键点数，<=0 不限制
    int   spNmsRadius;            ///< NMS 半径（像素），越小特征越密
    int   spRemoveBorders;        ///< 边界移除宽度（像素）
    int   spNeighborhoodRadius;   ///< 邻域黑边检测半径，0=禁用
    float spNeighborhoodThresh;   ///< 邻域检测灰度阈值
    int   spMaxImageDim;          ///< 输入图像最长边限制，<=0 不缩放
    // SuperGlue
    float sgMatchThreshold;       ///< 匹配置信度阈值，越低匹配越多
    int   sgSinkhornIters;        ///< Sinkhorn 迭代次数
    // 粗差剔除
    double outlierReprojThresh;   ///< RANSAC 重投影误差阈值（像素）
    int    sgMinInliers;          ///< 粗差剔除后最少内点数；低于此阈值则舍弃该影像对
};

QualityPresets presetsForLevel(int quality)
{
    // 注意：initMinNumMatches / initMinNumInliers 不宜过高——
    // 自动流水线使用估算内参（非精确标定），相对定向内点率会偏低。
    // 阈值过高会导致完全无法初始化。
    //
    // SuperPoint 原始论文使用 640-1600px 输入，大图需缩放否则特征稀疏。
    // spMaxImageDim 控制最长边上限：先缩放再提特征，坐标自动映射回原图。
    //
    //                           SFM                              SuperPoint                                                    SuperGlue          Outlier
    //                   minMatch inl  lBA gBA   thresh  maxKp nms brd  nbR  nbT   maxDim  sgThr  sink  reproj  minInl
    switch (quality)
    {
    case 0: return {       15,    5,  5,  15,   0.005f, 2048,  3,  4,   3,  0.05f, 1200,  0.20f, 100,  4.0,    15  };   // 低精度 — 快速
    case 1: return {       20,    5,  4,  12,   0.003f, 4096,  2,  3,   2,  0.03f, 1600,  0.15f, 120,  3.0,    20  };   // 中精度
    case 2: return {       25,    5,  3,  10,   0.002f, 8192,  2,  2,   0,  0.00f, 2000,  0.10f, 150,  2.5,    20  };   // 高精度
    case 3:
    default: return {      30,    5,  3,  10,   0.001f,   -1,  2,  2,   0,  0.00f,    0,  0.20f, 200,  2.0,    25  };   // 最高精度 — 不缩放
    }
}

// ── 内部数据结构 ─────────────────────────────────────────────────────────────

/// 单张影像的缓存特征数据（含描述子，Phase 2 匹配需要）
struct ImageFeatureCache
{
    SuperPointOutput spOutput;      ///< keypoints, scores, descriptors
    int imgH = 0;                   ///< 图像高度（SuperGlue 位置编码需要）
    int imgW = 0;                   ///< 图像宽度
};

/// 一对影像的匹配数据
struct PairMatchData
{
    ImageId idA = 0;
    ImageId idB = 0;
    std::vector<FeatureMatch> matches;
    bool loaded = false;
};

} // anonymous namespace

// ══════════════════════════════════════════════════════════════════════════════
// SFMService::run  — 一站式增量 SFM 全自动主入口（同步阻塞）
// ══════════════════════════════════════════════════════════════════════════════
SFMServiceResult SFMService::run(const SFMServiceOptions &opts)
{
    SFMServiceResult result;
    QElapsedTimer elapsedTimer;
    elapsedTimer.start();

    auto reportProgress = [&](const QString &stage, int percent)
    {
        if (opts.progressFn)
        {
            opts.progressFn(stage, percent);
        }
    };

    // 取消检查辅助 lambda
    auto isCancelled = [&]() -> bool
    {
        return opts.cancelFlag && opts.cancelFlag->load();
    };

    // ══════════════════════════════════════════════════════════════════════════
    // 0. 基本校验
    // ══════════════════════════════════════════════════════════════════════════
    if (opts.images.size() < 2)
    {
        result.errorMessage = QStringLiteral("至少需要选择两张影像");
        result.summary      = result.errorMessage;
        return result;
    }

    const QualityPresets presets = presetsForLevel(opts.quality);

    // ══════════════════════════════════════════════════════════════════════════
    // 1. 确定计算设备
    // ══════════════════════════════════════════════════════════════════════════
    bool useCuda = false;
    if (opts.device == QStringLiteral("cuda") ||
        (opts.device == QStringLiteral("auto") && torch::cuda::is_available()))
    {
        useCuda = true;
    }
    LOG_INFO(QStringLiteral("SFM 流水线: 使用 %1 设备, 精度等级 %2")
        .arg(useCuda ? QStringLiteral("CUDA") : QStringLiteral("CPU"))
        .arg(opts.quality));

    // ══════════════════════════════════════════════════════════════════════════
    // 2. 设置目录
    // ══════════════════════════════════════════════════════════════════════════
    const QString assetsDir = ProjectIO::projectAssetsDir(opts.plascanPath);
    const QString projectRootDir = ProjectIO::projectRootFromPlascan(opts.plascanPath);
    const QString ipDir     = QDir(assetsDir).filePath(QStringLiteral("ip"));
    const QString matchDir  = QDir(assetsDir).filePath(QStringLiteral("matches"));
    const QString outDir    = QDir::cleanPath(opts.outputDir);
    QDir().mkpath(ipDir);
    QDir().mkpath(matchDir);
    if (!outDir.isEmpty())
    {
        QDir().mkpath(outDir);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 3. 分配 ImageId
    // ══════════════════════════════════════════════════════════════════════════
    QMap<QString, ImageId> imageIdMap;   // normPath → id
    QMap<ImageId, QString> idToPath;     // id → normPath
    ImageId nextId = 0;
    for (const QString &imgPath : opts.images)
    {
        const QString norm = normalizePath(imgPath);
        imageIdMap[norm] = nextId;
        idToPath[nextId] = norm;
        ++nextId;
    }
    const int N = static_cast<int>(opts.images.size());

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 1: 确保所有影像的特征文件存在
    // ══════════════════════════════════════════════════════════════════════════
    reportProgress(QStringLiteral("检查特征文件..."), 2);
    LOG_INFO(QStringLiteral("SFM Phase 1: 检查 / 提取特征..."));

    QMap<ImageId, ImageFeatureCache> featureCache;
    QMap<ImageId, QString> spFilePaths;     // id → .sp 文件路径
    QVector<ImageId> missingSpIds;

    // 1a. 查找已有的 .sp 文件
    for (auto it = imageIdMap.constBegin(); it != imageIdMap.constEnd(); ++it) {
        const QString &imgPath = it.key();
        const ImageId id       = it.value();
        const QString spPath   = ProjectIO::findSpForImage(opts.plascanPath, imgPath);
        if (!spPath.isEmpty()) {
            spFilePaths[id] = spPath;
        } else {
            missingSpIds.append(id);
        }
    }

    LOG_INFO(QStringLiteral("  已有特征: %1/%2, 需提取: %3")
        .arg(spFilePaths.size()).arg(N).arg(missingSpIds.size()));

    // 1b. 提取缺失的 SuperPoint 特征
    if (!missingSpIds.isEmpty()) 
    {
        LOG_INFO(QStringLiteral("  启动 SuperPoint 特征提取..."));

        const QString spModelName = useCuda
            ? QStringLiteral("superpoint_v6_cuda.pt")
            : QStringLiteral("superpoint_v6_cpu.pt");
        const QString spModelPath = findModelFile(spModelName);
        if (spModelPath.isEmpty()) 
        {
            result.errorMessage = QStringLiteral("未找到 SuperPoint 模型文件: %1").arg(spModelName);
            result.summary      = result.errorMessage;
            return result;
        }

        try 
        {
            SuperPointConfig spCfg;
            spCfg.device                   = useCuda ? torch::kCUDA : torch::kCPU;
            spCfg.detection_threshold      = presets.spDetectionThreshold;
            spCfg.max_num_keypoints        = presets.spMaxKeypoints;
            spCfg.nms_radius               = presets.spNmsRadius;
            spCfg.remove_borders           = presets.spRemoveBorders;
            spCfg.neighborhood_check_radius = presets.spNeighborhoodRadius;
            spCfg.neighborhood_threshold   = presets.spNeighborhoodThresh;
            spCfg.allow_device_fallback    = true;

            SuperPoint sp(spModelPath.toStdString(), spCfg);

            int spDoneCount = 0;
            const int spTotalCount = static_cast<int>(missingSpIds.size());
            for (const ImageId id : missingSpIds) 
            {
                // 取消检查
                if (isCancelled())
                {
                    result.errorMessage = QStringLiteral("用户取消");
                    result.summary = result.errorMessage;
                    return result;
                }

                // 报告特征提取进度（2%~35% 区间映射）
                {
                    int pct = 2 + (spDoneCount * 33) / std::max(1, spTotalCount);
                    reportProgress(QStringLiteral("正在查找特征点... %1/%2")
                        .arg(spDoneCount + 1).arg(spTotalCount), pct);
                }

                const QString &imgPath = idToPath[id];
                const QFileInfo fi(imgPath);

                cv::Mat image = cv::imread(imgPath.toStdString(), cv::IMREAD_GRAYSCALE);
                if (image.empty()) 
                {
                    LOG_WARN(QStringLiteral("  无法读取图像: %1").arg(fi.fileName()));
                    continue;
                }

                // ── 大图自适应缩放 ──────────────────────────────────
                // SuperPoint 训练分辨率约 480-640px，大图需缩放到合理尺度
                // 以提高特征密度。提取后将坐标自动映射回原图分辨率。
                const int origH = image.rows;
                const int origW = image.cols;
                float scale = 1.0f;
                cv::Mat inputImg = image;

                if (presets.spMaxImageDim > 0) 
                {
                    const int maxDim = std::max(origH, origW);
                    if (maxDim > presets.spMaxImageDim) 
                    {
                        scale = static_cast<float>(presets.spMaxImageDim) / static_cast<float>(maxDim);
                        const int newW = static_cast<int>(std::round(origW * scale));
                        const int newH = static_cast<int>(std::round(origH * scale));
                        cv::resize(image, inputImg, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
                        LOG_INFO(QStringLiteral("  图像缩放: %1x%2 → %3x%4 (scale=%5)")
                            .arg(origW).arg(origH).arg(newW).arg(newH)
                            .arg(static_cast<double>(scale), 0, 'f', 3));
                    }
                }

                SuperPointOutput spOut = sp.detect(inputImg);

                // 若进行了缩放，将关键点坐标映射回原图分辨率
                if (scale < 1.0f && !spOut.keypoints.empty()) 
                {
                    const float invScale = 1.0f / scale;
                    for (auto &kp : spOut.keypoints)  
                    {
                        kp.pt.x *= invScale;
                        kp.pt.y *= invScale;
                    }
                }

                const QString spPath = QDir(ipDir).filePath(fi.completeBaseName() + QStringLiteral(".sp"));
                if (!FeatureFileIO::write(spPath, fi.fileName(), spOut)) 
                {
                    LOG_WARN(QStringLiteral("  保存特征文件失败: %1").arg(spPath));
                    continue;
                }

                LOG_INFO(QStringLiteral("  提取 %1 个特征点: %2")
                    .arg(spOut.keypoints.size()).arg(fi.fileName()));

                spFilePaths[id] = spPath;

                // 缓存特征及图像尺寸（Phase 2 匹配可直接使用）
                ImageFeatureCache &fc = featureCache[id];
                fc.spOutput = std::move(spOut);
                fc.imgH     = image.rows;
                fc.imgW     = image.cols;

                // 记录新生成的文件
                result.newSpFiles.append({imgPath, spPath});
                ++spDoneCount;
            }
        } 
        catch (const std::exception &e) 
        {
            result.errorMessage = QStringLiteral("SuperPoint 特征提取失败: %1")
                .arg(QString::fromStdString(e.what()));
            result.summary = result.errorMessage;
            return result;
        }
    }

    // 1c. 加载已有 .sp 文件到缓存（仅加载尚未缓存的）
    for (auto it = spFilePaths.constBegin(); it != spFilePaths.constEnd(); ++it) 
    {
        const ImageId id     = it.key();
        const QString &spPath = it.value();
        if (featureCache.contains(id)) continue;   // 刚提取的已在缓存中

        QString imageName;
        SuperPointOutput spOut;
        if (!FeatureFileIO::read(spPath, imageName, spOut)) 
        {
            LOG_WARN(QStringLiteral("  读取特征文件失败: %1").arg(spPath));
            continue;
        }

        ImageFeatureCache &fc = featureCache[id];
        fc.spOutput = std::move(spOut);
        // 图像尺寸未知，Phase 2 匹配时按需读取
    }

    // 1d. 检查可用特征数量
    if (featureCache.size() < 2) 
    {
        result.errorMessage = QStringLiteral("有效影像不足 2 张（仅 %1 张有特征）")
            .arg(featureCache.size());
        result.summary = result.errorMessage;
        return result;
    }

    // 1e. 取消检查
    if (isCancelled())
    {
        result.errorMessage = QStringLiteral("用户取消");
        result.summary = result.errorMessage;
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 2: 确保所有影像对的匹配结果存在
    // ══════════════════════════════════════════════════════════════════════════
    reportProgress(QStringLiteral("检查匹配文件..."), 35);
    LOG_INFO(QStringLiteral("SFM Phase 2: 检查 / 生成匹配..."));

    // 有效 Id 列表（只保留有特征的）
    QVector<ImageId> validIds;
    for (auto it = featureCache.constBegin(); it != featureCache.constEnd(); ++it)
        validIds.append(it.key());
    std::sort(validIds.begin(), validIds.end());

    // ── 加载 no_match_pairs.json：上次已确认无匹配的影像对 ──────────────────────
    // key: "normPath0|normPath1"（两方向均存），value 未使用（以文件 mtime 比较代替）
    QSet<QString> noMatchSet;
    const QString noMatchFilePath = QDir(matchDir).filePath(QStringLiteral("no_match_pairs.json"));
    const QDateTime noMatchFileMtime = QFileInfo(noMatchFilePath).lastModified();
    {
        QFile nmf(noMatchFilePath);
        if (nmf.open(QIODevice::ReadOnly)) 
        {
            const QJsonArray arr = QJsonDocument::fromJson(nmf.readAll()).array();
            nmf.close();
            for (const QJsonValue &v : arr) 
            {
                const QJsonObject o = v.toObject();
                const QString p0 = normalizePath(o.value(QStringLiteral("image0")).toString());
                const QString p1 = normalizePath(o.value(QStringLiteral("image1")).toString());
                if (!p0.isEmpty() && !p1.isEmpty()) 
                {
                    noMatchSet.insert(p0 + QStringLiteral("|") + p1);
                    noMatchSet.insert(p1 + QStringLiteral("|") + p0);
                }
            }
            LOG_INFO(QStringLiteral("  已加载 no_match_pairs.json: %1 条无匹配记录")
                .arg(noMatchSet.size() / 2));
        }
    }

    QSet<QString> allowedPairSet;
    for (const QString &pairKey : opts.allowedPairs)
    {
        const QString trimmedKey = pairKey.trimmed();
        if (!trimmedKey.isEmpty())
        {
            allowedPairSet.insert(trimmedKey);
        }
    }

    if (opts.restrictPairs)
    {
        LOG_INFO(QStringLiteral("  匹配对约束已启用: %1 对").arg(allowedPairSet.size()));
    }

    // 从项目元数据复用已有匹配记录（覆盖“仅按 assets/matches 规范命名扫描”的限制）
    // 适配场景：ipmatch 使用了自定义 output_dir，文件不在 assets/matches 下。
    QMap<QString, QString> metaMatchPathByPair;
    QMap<QString, QString> metaMatchPathByBasePair;
    {
        const QJsonArray ipmatchResults = ipmatchResultsFromMeta(opts.projectMeta);
        for (const QJsonValue &val : ipmatchResults)
        {
            const QJsonObject rec = val.toObject();
            const QString rawOutput = rec.value(QStringLiteral("output")).toString();
            const QString outputPath = resolvePathFromProjectRoot(projectRootDir, rawOutput);
            if (outputPath.isEmpty() || !QFile::exists(outputPath))
            {
                continue;
            }

            QString image0 = resolveImagePathTokenFromMeta(rec.value(QStringLiteral("image0")).toString(),
                                                           opts.projectMeta);
            QString image1 = resolveImagePathTokenFromMeta(rec.value(QStringLiteral("image1")).toString(),
                                                           opts.projectMeta);

            if (image0.isEmpty() || image1.isEmpty())
            {
                const QJsonArray imageFiles = rec.value(QStringLiteral("settings"))
                                                 .toObject()
                                                 .value(QStringLiteral("image_files"))
                                                 .toArray();
                if (imageFiles.size() >= 2)
                {
                    image0 = resolveImagePathTokenFromMeta(imageFiles.at(0).toString(), opts.projectMeta);
                    image1 = resolveImagePathTokenFromMeta(imageFiles.at(1).toString(), opts.projectMeta);
                }
            }

            if (image0.isEmpty() || image1.isEmpty())
            {
                const QString sidecarPath = outputPath + QStringLiteral(".json");
                QFile sidecarFile(sidecarPath);
                if (sidecarFile.open(QIODevice::ReadOnly))
                {
                    const QJsonObject sidecarObj = QJsonDocument::fromJson(sidecarFile.readAll()).object();
                    sidecarFile.close();
                    image0 = resolveImagePathTokenFromMeta(sidecarObj.value(QStringLiteral("image0_path")).toString(),
                                                           opts.projectMeta);
                    image1 = resolveImagePathTokenFromMeta(sidecarObj.value(QStringLiteral("image1_path")).toString(),
                                                           opts.projectMeta);
                }
            }

            const QString pairKey = canonicalPairKey(image0, image1);
            const QString basePairKey = canonicalNamePairKey(QFileInfo(image0).completeBaseName(),
                                                             QFileInfo(image1).completeBaseName());
            if (pairKey.isEmpty() && basePairKey.isEmpty())
            {
                continue;
            }

            auto upsertNewest = [&](QMap<QString, QString> *pathMap, const QString &key)
            {
                if (!pathMap || key.isEmpty())
                {
                    return;
                }
                const QString existingPath = pathMap->value(key);
                if (existingPath.isEmpty())
                {
                    pathMap->insert(key, outputPath);
                    return;
                }

                const QDateTime oldTime = QFileInfo(existingPath).lastModified();
                const QDateTime newTime = QFileInfo(outputPath).lastModified();
                if (newTime > oldTime)
                {
                    (*pathMap)[key] = outputPath;
                }
            };

            upsertNewest(&metaMatchPathByPair, pairKey);
            upsertNewest(&metaMatchPathByBasePair, basePairKey);
        }
        if (!metaMatchPathByPair.isEmpty())
        {
            LOG_INFO(QStringLiteral("  元数据可复用匹配: %1 对").arg(metaMatchPathByPair.size()));
        }
    }

    int metaFoundPathCount = 0;
    int metaReadOkCount = 0;
    int metaBaseHitCount = 0;

    // 生成所有需要处理的影像对并检查已有匹配文件
    QVector<PairMatchData> allPairs;
    QVector<int> missingPairIndices;

    for (int i = 0; i < validIds.size(); ++i) 
    {
        for (int j = i + 1; j < validIds.size(); ++j) 
        {
            const ImageId idA = validIds[i];
            const ImageId idB = validIds[j];

            if (opts.restrictPairs)
            {
                const QString pairKey = canonicalPairKey(idToPath.value(idA), idToPath.value(idB));
                if (pairKey.isEmpty() || !allowedPairSet.contains(pairKey))
                {
                    continue;
                }
            }

            const QString baseA = QFileInfo(idToPath[idA]).completeBaseName();
            const QString baseB = QFileInfo(idToPath[idB]).completeBaseName();

            PairMatchData pd;
            pd.idA = idA;
            pd.idB = idB;

            // 在 matches 目录下查找已有 .match 文件（两种命名顺序）
            const QString pathAB = QDir(matchDir).filePath(
                QStringLiteral("%1__%2.match").arg(baseA, baseB));
            const QString pathBA = QDir(matchDir).filePath(
                QStringLiteral("%1__%2.match").arg(baseB, baseA));

            QString foundPath;
            if (QFile::exists(pathAB))      foundPath = pathAB;
            else if (QFile::exists(pathBA)) foundPath = pathBA;
            else
            {
                const QString pairKey = canonicalPairKey(idToPath.value(idA), idToPath.value(idB));
                const QString metaPath = metaMatchPathByPair.value(pairKey);
                if (!metaPath.isEmpty() && QFile::exists(metaPath))
                {
                    foundPath = metaPath;
                    ++metaFoundPathCount;
                }
                else
                {
                    const QString basePairKey = canonicalNamePairKey(baseA, baseB);
                    const QString metaBasePath = metaMatchPathByBasePair.value(basePairKey);
                    if (!metaBasePath.isEmpty() && QFile::exists(metaBasePath))
                    {
                        foundPath = metaBasePath;
                        ++metaFoundPathCount;
                        ++metaBaseHitCount;
                    }
                }
            }

            // 检查 .match 文件是否过期：仅当 .sp 特征文件比 .match 更新时才视为过期
            // （即特征点重新提取后，旧匹配结果失效）
            // 注意：不再检查 match_threshold，避免因参数变化引发全量重匹配
            if (!foundPath.isEmpty()) 
            {
                const QDateTime matchTime = QFileInfo(foundPath).lastModified();
                bool stale = false;
                if (spFilePaths.contains(idA) &&
                    QFileInfo(spFilePaths[idA]).lastModified() > matchTime)
                    stale = true;
                if (spFilePaths.contains(idB) &&
                    QFileInfo(spFilePaths[idB]).lastModified() > matchTime)
                    stale = true;

                if (stale)
                {
                    if (opts.autoGenerateMissingMatches)
                    {
                        LOG_INFO(QStringLiteral("  匹配缓存过期(特征已更新), 重新生成: %1")
                                                     .arg(foundPath));
                        QFile::remove(foundPath);
                        QFile::remove(foundPath + QStringLiteral(".json"));
                    }
                    else
                    {
                        LOG_WARN(QStringLiteral("  匹配缓存过期(特征已更新), 自动补匹配已禁用，跳过该对: %1")
                                                     .arg(foundPath));
                    }
                    foundPath.clear();
                }
            }

            if (!foundPath.isEmpty()) 
            {
                // 读取已有匹配
                QString img0name, img1name;
                xjw::feature_match::MatchResult mr;
                if (SuperGlueMatchIO::read(foundPath, img0name, img1name, mr)
                    && !mr.cvMatches.empty())
                {
                    // 判断方向：image0 (query) 对应哪张影像
                    bool direct = true;     // 默认 idA = image0
                    const QString sidecarPath = foundPath + QStringLiteral(".json");
                    if (QFile::exists(sidecarPath)) 
                    {
                        QFile f(sidecarPath);
                        if (f.open(QIODevice::ReadOnly)) 
                        {
                            const QJsonObject sc = QJsonDocument::fromJson(f.readAll()).object();
                            f.close();
                            const QString si0 = normalizePath(sc.value(QStringLiteral("image0_path")).toString());
                            const QString si1 = normalizePath(sc.value(QStringLiteral("image1_path")).toString());
                            if (si0 == idToPath[idA] && si1 == idToPath[idB])
                                direct = true;
                            else if (si0 == idToPath[idB] && si1 == idToPath[idA])
                                direct = false;
                        }
                    } 
                    else 
                    {
                        // 无 sidecar：通过文件中的影像名推断
                        if (img0name == baseB) direct = false;
                    }

                    pd.matches.reserve(mr.cvMatches.size());
                    for (const auto &dm : mr.cvMatches) 
                    {
                        FeatureMatch fm;
                        if (direct) 
                        {
                            fm.idx1 = static_cast<FeatureIdx>(dm.queryIdx);
                            fm.idx2 = static_cast<FeatureIdx>(dm.trainIdx);
                        } 
                        else 
                        {
                            fm.idx1 = static_cast<FeatureIdx>(dm.trainIdx);
                            fm.idx2 = static_cast<FeatureIdx>(dm.queryIdx);
                        }
                        pd.matches.push_back(fm);
                    }
                    pd.loaded = true;
                    ++metaReadOkCount;
                }
                else if (metaMatchPathByPair.contains(canonicalPairKey(idToPath.value(idA), idToPath.value(idB))))
                {
                    LOG_WARN(QStringLiteral("  元数据命中但读取匹配失败或为空: %1").arg(foundPath));
                }
            }

            // ── 无匹配记录检查：若已知该对无匹配，且 sp 文件在记录后未更新，跳过 ──
            if (!pd.loaded) 
            {
                const QString pA = normalizePath(idToPath[idA]);
                const QString pB = normalizePath(idToPath[idB]);
                const QString nmKey = pA + QStringLiteral("|") + pB;
                if (noMatchSet.contains(nmKey) && noMatchFileMtime.isValid()) 
                {
                    // 以 no_match_pairs.json 文件的修改时间为基准
                    // 若两张影像的 sp 文件均早于该时间 → 特征未变 → 直接跳过
                    bool spNewer = false;
                    if (spFilePaths.contains(idA) &&
                        QFileInfo(spFilePaths[idA]).lastModified() > noMatchFileMtime)
                        spNewer = true;
                    if (spFilePaths.contains(idB) &&
                        QFileInfo(spFilePaths[idB]).lastModified() > noMatchFileMtime)
                        spNewer = true;
                    if (!spNewer)
                        pd.loaded = true;   // 已知无匹配且特征未更新，跳过
                }
            }

            const int pairIdx = allPairs.size();
            allPairs.append(pd);
            if (!pd.loaded) 
            {
                missingPairIndices.append(pairIdx);
            }
        }
    }

    if (allPairs.isEmpty())
    {
        result.errorMessage = opts.restrictPairs
            ? QStringLiteral("所选影像中没有可用的已生成匹配对，请先创建连接点或检查 .lis 配对范围")
            : QStringLiteral("未找到可用影像对");
        result.summary = result.errorMessage;
        return result;
    }

    LOG_INFO(QStringLiteral("  总配对: %1, 已有匹配: %2, 需生成: %3")
        .arg(allPairs.size())
        .arg(allPairs.size() - missingPairIndices.size())
        .arg(missingPairIndices.size()));
    if (metaFoundPathCount > 0)
    {
        LOG_INFO(QStringLiteral("  元数据命中: %1 对(含基名兜底 %2), 读取成功: %3 对")
            .arg(metaFoundPathCount)
            .arg(metaBaseHitCount)
            .arg(metaReadOkCount));
    }

    // 2a. 对缺失配对执行 SuperGlue 匹配（可按选项禁用）
    if (!missingPairIndices.isEmpty() && opts.autoGenerateMissingMatches)
    {
        LOG_INFO(QStringLiteral("  启动 SuperGlue 特征匹配..."));

        const QString sgModelName = QStringLiteral("superglue_%1_%2.pt")
            .arg(opts.sgModelType, useCuda ? QStringLiteral("cuda") : QStringLiteral("cpu"));
        const QString sgModelPath = findModelFile(sgModelName);
        if (sgModelPath.isEmpty()) 
        {
            result.errorMessage = QStringLiteral("未找到 SuperGlue 模型文件: %1").arg(sgModelName);
            result.summary      = result.errorMessage;
            return result;
        }

        // ── 粗差剔除配置（USAC_MAGSAC，最优粗差剔除）──────────────────────────────────
        superglue::OutlierFilterConfig outlierCfg;
        outlierCfg.method          = superglue::OutlierMethod::FundamentalUsacMagsac;
        outlierCfg.reprojThreshold = presets.outlierReprojThresh;
        outlierCfg.confidence      = 0.9999;
        outlierCfg.maxIters        = 10000;
        outlierCfg.minInliers      = presets.sgMinInliers;

        // ── 预加载所有影像尺寸，消除并行阶段的写竞争 ──────────────────────
        {
            QSet<ImageId> needDim;
            for (const int pi : missingPairIndices) 
            {
                needDim.insert(allPairs[pi].idA);
                needDim.insert(allPairs[pi].idB);
            }
            for (const ImageId id : needDim) 
            {
                if (!featureCache.contains(id)) continue;
                ImageFeatureCache &fc = featureCache[id];
                if (fc.imgH == 0 || fc.imgW == 0) 
                {
                    cv::Mat img = cv::imread(idToPath[id].toStdString(), cv::IMREAD_GRAYSCALE);
                    if (!img.empty()) { fc.imgH = img.rows; fc.imgW = img.cols; }
                    else              { fc.imgH = 1000;     fc.imgW = 1000;     }
                }
            }
        }

        // CUDA 模式：由用户指定并行对数（每个线程独立持有一个 Matcher 实例占用显存）
        // CPU 模式：按线程数并行
        const int numMatchThreads = useCuda
            ? std::max(1, opts.cudaParallelPairs)
            : std::max(1, opts.threads);
        LOG_INFO(QStringLiteral("  匹配线程数: %1 (%2 模式)")
            .arg(numMatchThreads)
            .arg(useCuda ? QStringLiteral("CUDA") : QStringLiteral("CPU-并行")));

        // 每个工作线程独立持有一个 SuperGlueMatcher，避免模型权重并发写入问题
        superglue::SuperGlueConfig sgCfg;
        sgCfg.model_path          = sgModelPath.toStdString();
        sgCfg.use_cuda            = useCuda;
        sgCfg.match_threshold     = presets.sgMatchThreshold;
        sgCfg.sinkhorn_iterations = presets.sgSinkhornIters;

        // ── 验证模型可加载（用单个测试实例） ─────────────────────────────
        {
            superglue::SuperGlueMatcher testMatcher(sgCfg);
            if (!testMatcher.isLoaded()) 
            {
                result.errorMessage = QStringLiteral("SuperGlue 模型加载失败");
                result.summary      = result.errorMessage;
                return result;
            }
        }

        // ── 共享错误状态（线程间）────────────────────────────────────────
        std::atomic<bool>         matchErrorFlag{false};
        std::string               matchErrorMsg;
        std::mutex                writeMutex;       // 保护对 result 的追加操作
        std::atomic<int>          pairCursor{0};    // 原子索引，各线程竞争取下一个待处理 pair
        const int                 totalMissing = static_cast<int>(missingPairIndices.size());

        // 省去重复读取：featureCache/idToPath/spFilePaths 在预加载后为纯只读
        // 使用 constFind 进行线程安全只读访问（无写入时并发读取安全）
        auto matchWorker = [&]() 
        {
            try 
            {
                // 每线程自有 matcher 实例
                std::unique_ptr<xjw::feature_match::IFeatureMatcher> localMatcher;
                {
                    auto sgMatcher = std::make_unique<superglue::SuperGlueMatcher>(sgCfg);
                    if (!sgMatcher->isLoaded()) return;
                    localMatcher = std::move(sgMatcher);
                }

                while (!matchErrorFlag.load()) 
                {
                    // 取消检查
                    if (opts.cancelFlag && opts.cancelFlag->load()) break;

                    const int localIdx = pairCursor.fetch_add(1);
                    if (localIdx >= totalMissing) break;

                    // 报告匹配进度（35%~70% 区间映射）
                    {
                        int pct = 35 + (localIdx * 35) / std::max(1, totalMissing);
                        reportProgress(QStringLiteral("正在匹配特征点... %1/%2")
                            .arg(localIdx + 1).arg(totalMissing), pct);
                    }

                    const int pi      = missingPairIndices[localIdx];
                    PairMatchData &pd = allPairs[pi];   // pi 唯一，无写竞争
                    const ImageId idA = pd.idA;
                    const ImageId idB = pd.idB;

                    auto itA = featureCache.constFind(idA);
                    auto itB = featureCache.constFind(idB);
                    if (itA == featureCache.constEnd() || itB == featureCache.constEnd()) continue;

                    const ImageFeatureCache &fcA = *itA;
                    const ImageFeatureCache &fcB = *itB;

                    auto fdA = xjw::feature_extractors::FeatureData::fromSuperPointOutput(fcA.spOutput, "superpoint");
                    auto fdB = xjw::feature_extractors::FeatureData::fromSuperPointOutput(fcB.spOutput, "superpoint");
                    fdA.imageWidth = fcA.imgW;
                    fdA.imageHeight = fcA.imgH;
                    fdB.imageWidth = fcB.imgW;
                    fdB.imageHeight = fcB.imgH;

                    // 执行匹配
                    xjw::feature_match::MatchResult mr = localMatcher->match(fdA, fdB);

                    // 粗差剔除
                    int inlierCount = mr.numMatches;
                    mr = superglue::MatchOutlierRejector::filter(
                        mr, fcA.spOutput.keypoints, fcB.spOutput.keypoints,
                        outlierCfg, &inlierCount);

                    const QString baseA    = QFileInfo(idToPath.value(idA)).completeBaseName();
                    const QString baseB    = QFileInfo(idToPath.value(idB)).completeBaseName();
                    const QString pairName = QStringLiteral("%1__%2").arg(baseA, baseB);

                    // ── 最小内点数检测：内点不足时记录到 failedPairs（不写文件）──
                    if (mr.numMatches < presets.sgMinInliers) 
                    {
                        LOG_INFO(
                            QStringLiteral("  跳过 %1: 内点数 %2 < 阈值 %3（已记录为无匹配对）")
                            .arg(pairName).arg(mr.numMatches).arg(presets.sgMinInliers));
                        {
                            std::lock_guard<std::mutex> lk(writeMutex);
                            FailedPairRecord fpr;
                            fpr.imagePath0 = idToPath.value(idA);
                            fpr.imagePath1 = idToPath.value(idB);
                            result.failedPairs.append(fpr);
                        }
                        // pd.loaded 保持 false，不写任何文件
                        continue;
                    }

                    LOG_INFO(QStringLiteral("  匹配 %1: %2 对匹配点")
                        .arg(pairName).arg(mr.numMatches));

                    // ── 保存 .match 文件 ────────────────────────────────────
                    const QString matchPath = QDir(matchDir).filePath(pairName + QStringLiteral(".match"));
                    SuperGlueMatchIO::write(matchPath, baseA, baseB, mr);

                    // ── 保存 sidecar JSON ───────────────────────────────────
                    QJsonObject sidecar;
                    sidecar[QStringLiteral("match_file")]      = matchPath;
                    sidecar[QStringLiteral("image0_name")]     = baseA;
                    sidecar[QStringLiteral("image1_name")]     = baseB;
                    sidecar[QStringLiteral("image0_path")]     = idToPath.value(idA);
                    sidecar[QStringLiteral("image1_path")]     = idToPath.value(idB);
                    sidecar[QStringLiteral("sp0_path")]        = spFilePaths.value(idA);
                    sidecar[QStringLiteral("sp1_path")]        = spFilePaths.value(idB);
                    sidecar[QStringLiteral("num_matches")]     = mr.numMatches;
                    sidecar[QStringLiteral("match_threshold")] = static_cast<double>(presets.sgMatchThreshold);

                    QJsonArray pts0, pts1;
                    for (const auto &dm : mr.cvMatches) 
                    {
                        const int qi = dm.queryIdx, ti = dm.trainIdx;
                        if (qi >= 0 && qi < static_cast<int>(fcA.spOutput.keypoints.size()) &&
                            ti >= 0 && ti < static_cast<int>(fcB.spOutput.keypoints.size()))
                        {
                            const auto &kp0 = fcA.spOutput.keypoints[qi];
                            const auto &kp1 = fcB.spOutput.keypoints[ti];
                            QJsonArray p0; p0.append(kp0.pt.x); p0.append(kp0.pt.y);
                            QJsonArray p1; p1.append(kp1.pt.x); p1.append(kp1.pt.y);
                            pts0.append(p0);
                            pts1.append(p1);
                        }
                    }
                    sidecar[QStringLiteral("matched_points0")] = pts0;
                    sidecar[QStringLiteral("matched_points1")] = pts1;

                    const QString sidecarPath = matchPath + QStringLiteral(".json");
                    QFile sf(sidecarPath);
                    if (sf.open(QIODevice::WriteOnly | QIODevice::Truncate)) 
                    {
                        sf.write(QJsonDocument(sidecar).toJson(QJsonDocument::Compact));
                        sf.close();
                    }

                    // ── 构建匹配记录 ─────────────────────────────────────────
                    MatchFileRecord mfr;
                    mfr.pairName    = pairName;
                    mfr.matchPath   = matchPath;
                    mfr.sidecarPath = sidecarPath;

                    QJsonObject pairSettings;
                    QJsonArray imageFiles;
                    imageFiles.append(idToPath.value(idA));
                    imageFiles.append(idToPath.value(idB));
                    pairSettings[QStringLiteral("image_files")]  = imageFiles;
                    pairSettings[QStringLiteral("pair_name")]    = pairName;
                    pairSettings[QStringLiteral("sidecar_json")] = sidecarPath;
                    pairSettings[QStringLiteral("sp0_path")]     = spFilePaths.value(idA);
                    pairSettings[QStringLiteral("sp1_path")]     = spFilePaths.value(idB);
                    mfr.settings = pairSettings;

                    // ── 转换为 SFM 需要的 FeatureMatch ──────────────────────
                    pd.matches.reserve(mr.cvMatches.size());
                    for (const auto &dm : mr.cvMatches) {
                        FeatureMatch fm;
                        fm.idx1 = static_cast<FeatureIdx>(dm.queryIdx);
                        fm.idx2 = static_cast<FeatureIdx>(dm.trainIdx);
                        pd.matches.push_back(fm);
                    }
                    pd.loaded = true;

                    // ── 追加到 result（需加锁）─────────────────────────────
                    {
                        std::lock_guard<std::mutex> lk(writeMutex);
                        result.newMatchFiles.append(mfr);
                    }

                    // ── 实时回调：通知 UI 本对已完成（无需加锁，调用方负责线程转发）
                    if (opts.pairMatchedFn) 
                    {
                        opts.pairMatchedFn(idToPath.value(idA), idToPath.value(idB),
                                           matchPath, mr.numMatches);
                    }
                }
            } 
            catch (const std::exception &ex) 
            {
                std::lock_guard<std::mutex> lk(writeMutex);
                if (!matchErrorFlag.exchange(true)) 
                {
                    matchErrorMsg = ex.what();
                }
            }
        };

        // ── 启动工作线程 ──────────────────────────────────────────────────
        std::vector<std::thread> workerThreads;
        workerThreads.reserve(numMatchThreads);
        for (int t = 0; t < numMatchThreads; ++t)
            workerThreads.emplace_back(matchWorker);
        for (auto &th : workerThreads) th.join();

        if (matchErrorFlag.load()) 
        {
            result.errorMessage = QStringLiteral("SuperGlue 匹配失败: %1")
                .arg(QString::fromStdString(matchErrorMsg));
            result.summary = result.errorMessage;
            return result;
        }

        // ── 将无匹配的影像对追加写入 no_match_pairs.json ─────────────────────
        if (!result.failedPairs.isEmpty()) 
        {
            QJsonArray existing;
            {
                QFile nmf(noMatchFilePath);
                if (nmf.open(QIODevice::ReadOnly))
                    existing = QJsonDocument::fromJson(nmf.readAll()).array();
            }
            // 构建已存在条目的集合（避免重复追加）
            QSet<QString> existingKeys;
            for (const QJsonValue &v : existing) 
            {
                const QJsonObject o = v.toObject();
                existingKeys.insert(o[QStringLiteral("image0")].toString() +
                                    QStringLiteral("|") +
                                    o[QStringLiteral("image1")].toString());
            }
            const QString ts = QDateTime::currentDateTime().toString(Qt::ISODate);
            for (const FailedPairRecord &fpr : result.failedPairs) {
                const QString key  = fpr.imagePath0 + QStringLiteral("|") + fpr.imagePath1;
                const QString keyR = fpr.imagePath1 + QStringLiteral("|") + fpr.imagePath0;
                if (!existingKeys.contains(key) && !existingKeys.contains(keyR)) 
                {
                    QJsonObject o;
                    o[QStringLiteral("image0")]      = fpr.imagePath0;
                    o[QStringLiteral("image1")]      = fpr.imagePath1;
                    o[QStringLiteral("checked_at")]  = ts;
                    existing.append(o);
                    existingKeys.insert(key);
                }
            }
            QFile nmfOut(noMatchFilePath);
            if (nmfOut.open(QIODevice::WriteOnly | QIODevice::Truncate)) 
            {
                nmfOut.write(QJsonDocument(existing).toJson(QJsonDocument::Compact));
                nmfOut.close();  // 显式关闭以确保 mtime 立即生效
            }
            LOG_INFO(QStringLiteral("  无匹配对 %1 条已记录到 %2")
                .arg(result.failedPairs.size()).arg(noMatchFilePath));
        }
    }
    else if (!missingPairIndices.isEmpty())
    {
        LOG_INFO(QStringLiteral("  自动补匹配已禁用，跳过 %1 对缺失配对").arg(missingPairIndices.size()));
    }

    // 2b. 统计有效匹配
    int loadedMatches = 0;
    for (const auto &pd : allPairs) 
    {
        if (pd.loaded && !pd.matches.empty()) ++loadedMatches;
    }

    LOG_INFO(QStringLiteral("  有效匹配对: %1").arg(loadedMatches));

    if (loadedMatches < 1) 
    {
        result.errorMessage = QStringLiteral("未找到有效匹配对，无法执行重建");
        result.summary      = result.errorMessage;
        return result;
    }

    // 取消检查
    if (isCancelled())
    {
        result.errorMessage = QStringLiteral("用户取消");
        result.summary = result.errorMessage;
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // baOnly 模式：Phase 1+2 已完成，直接返回，由外部执行 BA
    // ══════════════════════════════════════════════════════════════════════════
    if (opts.baOnly) 
    {
        result.featureMatchesReady = true;
        result.summary = QStringLiteral("特征检测与匹配完成，可执行光束法平差");
        reportProgress(QStringLiteral("特征/匹配准备完毕"), 100);
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 3: 执行增量式 SFM 重建
    // ══════════════════════════════════════════════════════════════════════════
    reportProgress(QStringLiteral("执行 SFM 重建..."), 70);
    LOG_INFO(QStringLiteral("SFM Phase 3: 增量式重建..."));

    IncrementalSfmOptions sfmOpts;
    sfmOpts.initMinNumMatches = presets.initMinNumMatches;
    sfmOpts.initMinNumInliers = presets.initMinNumInliers;
    sfmOpts.localBAInterval   = presets.localBAInterval;
    sfmOpts.globalBAInterval  = presets.globalBAInterval;

    // 根据精度等级调整过滤参数：高精度/最高精度更严格
    if (opts.quality >= 2) 
    {
        sfmOpts.filterMaxReprojError = 1.5;      // 更严格的重投影误差阈值
        sfmOpts.filterMinTriAngle    = 2.0;
        sfmOpts.iterativeBARounds    = 4;         // 更多迭代精化轮数
    }

    IncrementalSfm sfm(sfmOpts);

    // 3a. 添加影像（使用缓存中的特征点 + 内参）
    // 内参来源优先级：cameraPaths(.tsai文件) > projectMeta(项目元数据) > userFu/Fv > 自动估算
    const bool hasUserIntrinsics = (opts.userFu > 0 && opts.userFv > 0);
    const bool hasUserPitch = (opts.userPitch > 0);
    const bool hasCameraPaths = (!opts.cameraPaths.isEmpty()
                                 && opts.cameraPaths.size() == opts.images.size());

    // 从 projectMeta 中预构建 imagePath → cameraJson 映射
    QMap<QString, QJsonObject> projectCameraMap;
    if (!opts.projectMeta.isEmpty()) 
    {
        const QJsonArray images = opts.projectMeta.value(QStringLiteral("images")).toArray();
        for (int i = 0; i < images.size(); ++i) 
        {
            if (!images[i].isObject()) continue;
            const QJsonObject imgObj = images[i].toObject();
            const QString imgPath = normalizePath(imgObj.value(QStringLiteral("path")).toString());
            const QJsonObject camObj = imgObj.value(QStringLiteral("camera")).toObject();
            if (!camObj.isEmpty() && camObj.contains(QStringLiteral("fu"))) 
            {
                projectCameraMap.insert(imgPath, camObj);
            }
        }
    }

    if (hasCameraPaths) 
    {
        LOG_INFO(QStringLiteral("  使用相机文件路径列表 (%1 个)")
            .arg(opts.cameraPaths.size()));
    } 
    else if (!projectCameraMap.isEmpty()) 
    {
        LOG_INFO(QStringLiteral("  从项目元数据加载相机参数 (%1 个)")
            .arg(projectCameraMap.size()));
    } else if (hasUserIntrinsics) {
        LOG_INFO(QStringLiteral("  使用用户提供的内参: fu=%1 fv=%2 cu=%3 cv=%4")
            .arg(opts.userFu, 0, 'f', 2).arg(opts.userFv, 0, 'f', 2)
            .arg(opts.userCu, 0, 'f', 2).arg(opts.userCv, 0, 'f', 2));
        if (hasUserPitch) {
            LOG_INFO(QStringLiteral("  像元大小 pitch=%1 mm")
                .arg(opts.userPitch, 0, 'f', 6));
        }
    } else {
        LOG_INFO(QStringLiteral("  使用自动估算内参 (focal ≈ max(w,h)×1.2)"));
    }

    int addedImages = 0;
    int cameraFromFile = 0;
    int cameraFromMeta = 0;
    int cameraFromUser = 0;
    int cameraEstimated = 0;

    for (auto it = featureCache.constBegin(); it != featureCache.constEnd(); ++it)
    {
        const ImageId id = it.key();
        const ImageFeatureCache &fc = it.value();

        std::vector<FeatureKeypoint> kpts;
        kpts.reserve(fc.spOutput.keypoints.size());
        for (const auto &kp : fc.spOutput.keypoints)
        {
            FeatureKeypoint fkp;
            fkp.x = kp.pt.x;
            fkp.y = kp.pt.y;
            kpts.push_back(fkp);
        }

        // 获取图像尺寸（优先用缓存，否则读取）
        int imgW = fc.imgW;
        int imgH = fc.imgH;
        if (imgW == 0 || imgH == 0)
        {
            cv::Mat img = cv::imread(idToPath[id].toStdString(), cv::IMREAD_GRAYSCALE);
            if (!img.empty())
            {
                imgW = img.cols;
                imgH = img.rows;
            }
            else
            {
                imgW = 1920;
                imgH = 1080;
            }
        }

        bool added = false;

        // 方式 1: 从 cameraPaths 列表加载 .tsai 文件
        if (!added && hasCameraPaths && id < static_cast<ImageId>(opts.cameraPaths.size()))
        {
            const QString &camPath = opts.cameraPaths[static_cast<int>(id)];
            if (!camPath.isEmpty() && QFileInfo::exists(camPath))
            {
                sfm.addImage(id, idToPath[id].toStdString(),
                             camPath.toStdString(), kpts);
                added = true;
                ++cameraFromFile;
            }
        }

        // 方式 2: 从项目元数据中加载相机参数
        if (!added)
        {
            const QString normImgPath = normalizePath(idToPath[id]);
            auto metaIt = projectCameraMap.find(normImgPath);
            if (metaIt != projectCameraMap.end())
            {
                Camera cam;
                if (xjw::gui::project::cameraFromJson(metaIt.value(), &cam) && cam.isValid())
                {
                    sfm.addImageWithCamera(id, idToPath[id].toStdString(), cam, kpts);
                    added = true;
                    ++cameraFromMeta;
                }
            }
        }

        // 方式 3: 使用用户提供的内参
        if (!added && hasUserIntrinsics)
        {
            Camera estimatedCam;
            double fuPx = opts.userFu;
            double fvPx = opts.userFv;
            double cuPx = opts.userCu;
            double cvPx = opts.userCv;
            if (opts.userPitch > 0)
            {
                fuPx = opts.userFu / opts.userPitch;
                fvPx = opts.userFv / opts.userPitch;
                if (cuPx > 0)
                {
                    cuPx = opts.userCu / opts.userPitch;
                }
                if (cvPx > 0)
                {
                    cvPx = opts.userCv / opts.userPitch;
                }
            }
            if (cuPx <= 0)
            {
                cuPx = imgW * 0.5;
            }
            if (cvPx <= 0)
            {
                cvPx = imgH * 0.5;
            }
            estimatedCam.setIntrinsics(fuPx, fvPx, cuPx, cvPx);

            if (addedImages == 0)
            {
                LOG_INFO(QStringLiteral("  像素内参: fu=%1 fv=%2 cu=%3 cv=%4")
                    .arg(fuPx, 0, 'f', 2).arg(fvPx, 0, 'f', 2)
                    .arg(cuPx, 0, 'f', 2).arg(cvPx, 0, 'f', 2));
            }
            sfm.addImageWithCamera(id, idToPath[id].toStdString(), estimatedCam, kpts);
            added = true;
            ++cameraFromUser;
        }

        // 方式 4: 自动估算焦距
        if (!added)
        {
            Camera estimatedCam;
            const double focalPx = std::max(imgW, imgH) * 1.2;
            estimatedCam.setIntrinsics(focalPx, focalPx,
                                       imgW * 0.5, imgH * 0.5);
            sfm.addImageWithCamera(id, idToPath[id].toStdString(), estimatedCam, kpts);
            ++cameraEstimated;
        }

        ++addedImages;
    }

    LOG_INFO(QStringLiteral("  相机来源: file=%1, meta=%2, user=%3, estimated=%4")
        .arg(cameraFromFile).arg(cameraFromMeta).arg(cameraFromUser).arg(cameraEstimated));

    // 3b. 添加匹配
    for (const auto &pd : allPairs) {
        if (!pd.loaded || pd.matches.empty()) continue;
        sfm.addMatches(pd.idA, pd.idB, pd.matches);
    }

    // 保存关键点位置（轻量，不含描述子），供导出步骤做颜色采样
    QMap<ImageId, std::vector<cv::KeyPoint>> kptPositions;
    for (auto it = featureCache.constBegin(); it != featureCache.constEnd(); ++it)
        kptPositions[it.key()] = it.value().spOutput.keypoints;

    // 释放特征缓存（描述子张量等大块内存），SFM 不再需要
    featureCache.clear();

    LOG_INFO(QStringLiteral("  影像: %1, 匹配对: %2, 开始重建...")
        .arg(addedImages).arg(loadedMatches));

    // 3c. 运行 SFM
    auto sfmResult = sfm.run([&reportProgress, &isCancelled](int registered, int total, const std::string &msg) -> bool {
        LOG_INFO(QStringLiteral("  SFM 进度: [%1/%2] %3")
            .arg(registered).arg(total).arg(QString::fromStdString(msg)));
        if (isCancelled())
            return false;
        int pct = 70;
        if (total > 0)
            pct = 70 + static_cast<int>(20.0 * registered / total);
        reportProgress(QStringLiteral("正在重建... %1/%2").arg(registered).arg(total), pct);
        return true;
    });

    // 3d. 取消检查
    if (isCancelled())
    {
        result.success = false;
        result.errorMessage = QStringLiteral("用户取消");
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 4: 处理结果
    // ══════════════════════════════════════════════════════════════════════════
    reportProgress(QStringLiteral("整理结果..."), 90);
    result.numRegisteredImages = sfmResult.numRegisteredImages;
    result.numPoints3D         = sfmResult.numPoints3D;
    result.meanReprojError     = sfmResult.meanReprojError;
    // BA 统计
    result.baRmsBefore       = sfmResult.baRmsBefore;
    result.baRmsAfter        = sfmResult.baRmsAfter;
    result.baTracksTotal     = sfmResult.baTracksTotal;
    result.baTracksOptimized = sfmResult.baTracksOptimized;
    result.baTracksFiltered  = sfmResult.baTracksFiltered;
    result.durationSeconds   = elapsedTimer.elapsed() / 1000.0;

    if (sfmResult.success && sfmResult.reconstruction) {
        result.success = true;

        LOG_INFO(QStringLiteral("SFM 完成: 注册 %1 张影像, %2 个三维点, 平均重投影误差 %3 px")
            .arg(sfmResult.numRegisteredImages)
            .arg(sfmResult.numPoints3D)
            .arg(sfmResult.meanReprojError, 0, 'f', 2));

        const auto *recon = sfmResult.reconstruction.get();

        // 4a. 收集相机参数更新
        for (auto it = imageIdMap.constBegin(); it != imageIdMap.constEnd(); ++it) {
            const ImageId id = it.value();
            if (!recon->isRegistered(id)) continue;
            const Camera &cam = recon->camera(id);
            result.pendingCamUpdates.insert(it.key(), cameraToJson(cam));
        }

        // 4a-2. 收集逐相机残差（供报告使用，一次遍历所有三维点）
        {
            // 先统计每相机的误差累计
            std::unordered_map<ImageId, std::pair<double,int>> camErr;  // id → (sumErr, count)
            for (Point3DId pid : recon->allPoint3DIds()) {
                if (!recon->hasPoint3D(pid)) continue;
                const auto &pt = recon->point3D(pid);
                for (const auto &elem : pt.track.elements) {
                    camErr[elem.imageId].first  += pt.error;
                    camErr[elem.imageId].second += 1;
                }
            }

            for (auto it = imageIdMap.constBegin(); it != imageIdMap.constEnd(); ++it) {
                const ImageId id  = it.value();
                const bool    reg = recon->isRegistered(id);
                double        res = 0.0;
                if (reg) {
                    auto eit = camErr.find(id);
                    if (eit != camErr.end() && eit->second.second > 0)
                        res = eit->second.first / eit->second.second;
                    else
                        res = sfmResult.meanReprojError;
                }
                QJsonObject camObj;
                camObj["path"]        = it.key();
                camObj["registered"]  = reg;
                camObj["residual_px"] = res;
                result.perCameraResiduals.append(camObj);
            }
        }

        // 4b. 导出稀疏点云（使用 xjw::pointcloud::PointCloud 写 PLY 二进制）
        if (!outDir.isEmpty()) {
            const QString plyPath = QDir(outDir).filePath(QStringLiteral("sfm_sparse.ply"));
            const auto ptIds = recon->allPoint3DIds();

            xjw::pointcloud::PointCloud cloud;
            cloud.reserve(ptIds.size());

            // 懒加载彩色图像用于颜色采样（每幅图像只加载一次）
            QMap<ImageId, cv::Mat> imgColorCache;

            for (Point3DId pid : ptIds) {
                if (!recon->hasPoint3D(pid)) continue;
                const auto &pt = recon->point3D(pid);
                // 过滤离群点：重投影误差 > 4px 或 track 长度 < 2
                if (pt.error > 4.0) continue;
                if (pt.track.length() < 2) continue;

                // 从轨迹元素采样像素颜色
                uint8_t cr = 128, cg = 128, cb = 128;  // 默认中灰
                for (const auto &elem : pt.track.elements) {
                    auto kptIt = kptPositions.find(elem.imageId);
                    if (kptIt == kptPositions.end()) continue;
                    const auto &kpts = kptIt.value();
                    if (elem.featureIdx >= static_cast<FeatureIdx>(kpts.size())) continue;

                    // 懒加载彩色图像
                    if (!imgColorCache.contains(elem.imageId)) {
                        const QString imgPath = idToPath.value(elem.imageId);
                        if (!imgPath.isEmpty())
                            imgColorCache[elem.imageId] = cv::imread(imgPath.toStdString(), cv::IMREAD_COLOR);
                    }
                    const cv::Mat &img = imgColorCache.value(elem.imageId);
                    if (img.empty()) continue;

                    const int px = static_cast<int>(std::round(kpts[elem.featureIdx].pt.x));
                    const int py = static_cast<int>(std::round(kpts[elem.featureIdx].pt.y));
                    if (px >= 0 && px < img.cols && py >= 0 && py < img.rows) {
                        const cv::Vec3b &pix = img.at<cv::Vec3b>(py, px);
                        cb = pix[0]; cg = pix[1]; cr = pix[2];  // OpenCV BGR 顺序
                        break;
                    }
                }

                cloud.addPoint(
                    xjw::Point3f{static_cast<float>(pt.xyz[0]),
                                 static_cast<float>(pt.xyz[1]),
                                 static_cast<float>(pt.xyz[2])},
                    xjw::ColorRGBA{cr, cg, cb, 255});
            }

            xjw::pointcloud::PointCloudWriteOptions writeOpts;
            writeOpts.format = xjw::pointcloud::PointCloudFileFormat::PlyBinaryLittleEndian;
            writeOpts.writeNormals = false;
            writeOpts.writeTextureCoordinates = false;
            writeOpts.writeFaces = false;

            if (xjw::pointcloud::writePointCloud(plyPath.toStdString(), cloud, writeOpts)) {
                result.sparseCloudPath = plyPath;
                LOG_INFO(QStringLiteral("SFM: 稀疏点云已保存 %1 个点（原始 %2 个）→ %3")
                    .arg((int)cloud.size()).arg((int)ptIds.size()).arg(plyPath));
            } else {
                LOG_INFO(QStringLiteral("SFM: 稀疏点云写出失败 → %1").arg(plyPath));
            }
        }

        result.summary = QStringLiteral("SFM 重建成功：注册 %1/%2 张影像，%3 个三维点")
            .arg(sfmResult.numRegisteredImages)
            .arg(opts.images.size())
            .arg(sfmResult.numPoints3D);
        result.durationSeconds = elapsedTimer.elapsed() / 1000.0;
        reportProgress(QStringLiteral("完成"), 100);

    } else {
        result.errorMessage = QStringLiteral("SFM 重建失败: %1")
            .arg(QString::fromStdString(sfmResult.summary));
        result.summary = result.errorMessage;
        LOG_WARN(result.errorMessage);
    }

    return result;
}

} // namespace gui
} // namespace xjw
