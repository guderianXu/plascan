// =============================================================================
// 文件名: BundleAdjustService.cpp
// 描述:   光束法平差服务实现，详细说明见 BundleAdjustService.h。
//
//         本文件专注于算法实现，不依赖任何 Qt Widget。
//         所有 QWidget 级别的错误弹框均由调用方（ProjectManager）负责。
// =============================================================================
#include "BundleAdjustService.h"

#include "Camera.h"
#include "BundleAdjust.h"
#include "Logger.h"
#include "ProjectSupportUtils.h"

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

namespace xjw
{
namespace gui
{

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
    const std::vector<xjw::Camera>& cameras,
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

    // ── [DryRun] 仅统计，不执行实际计算 ──────────────────────────────────
    if (opts.dryRun)
    {
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
        result.success    = true;
        result.resultJson = dryObj;
        return result;
    }

    // ── 执行光束法平差 ─────────────────────────────────────────────────────
    // xjw::BundleAdjust::optimizePoints 内部使用 Levenberg-Marquardt 算法，
    // 对所有相机与所有点交替迭代，最小化重投影误差的 Huber 加权和。
    const xjw::BAResult baResult = xjw::BundleAdjust::optimizePoints(cameras, tracks, opts.baOpt);
    if (opts.baOpt.cancelFlag && opts.baOpt.cancelFlag->load())
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

    // BA 选项回存（便于复现）
    {
        QJsonObject optObj;
        optObj[QStringLiteral("max_iterations")]       = opts.baOpt.maxIterations;
        optObj[QStringLiteral("max_point_iterations")] = opts.baOpt.maxPointIterations;
        optObj[QStringLiteral("max_camera_iterations")]= opts.baOpt.maxCameraIterations;
        optObj[QStringLiteral("refine_camera_pose")]   = opts.baOpt.refineCameraPose;
        optObj[QStringLiteral("huber_delta")]           = opts.baOpt.huberDelta;
        optObj[QStringLiteral("finite_diff_eps")]       = opts.baOpt.finiteDiffEps;
        optObj[QStringLiteral("damping")]               = opts.baOpt.damping;
        optObj[QStringLiteral("step_tolerance")]        = opts.baOpt.stepTolerance;
        saveObj[QStringLiteral("options")] = optObj;
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
        const QJsonObject afterJson = xjw::gui::project::cameraToJson(camAfter);

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
            camAfter.saveToFile(tsaiPath.toStdString());
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
            ts << "平均 RMS（前）: " << baResult.meanRmsBefore          << "\n";
            ts << "平均 RMS（后）: " << baResult.meanRmsAfter           << "\n";
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
    result.success          = true;
    result.pendingCamUpdates = pendingCamUpdates;
    result.resultJson        = saveObj;
    return result;
}

} // namespace gui
} // namespace xjw
