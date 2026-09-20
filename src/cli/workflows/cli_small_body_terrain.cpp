#include "cli_common.h"
#include "CliJsonIO.h"

#include "TerrainPipeline.h"
#include <QCoreApplication>
#ifdef PLASCAN_HAS_TERRAIN_REPORT
#include "GlobalTerrainReportRenderer.h"
#include <QGuiApplication>
#endif
#include <QJsonObject>
#include <QString>

#include <cstdio>
#include <atomic>
#include <csignal>
#include <string>
#include <memory>

namespace
{

std::atomic_bool gCancellationRequested{false};
static_assert(std::atomic_bool::is_always_lock_free,
              "CLI signal cancellation requires lock-free atomic_bool");

void requestCancellation(int)
{
    gCancellationRequested.store(true, std::memory_order_relaxed);
}

class ScopedCancellationSignals final
{
public:
    ScopedCancellationSignals()
    {
        gCancellationRequested.store(false, std::memory_order_relaxed);
        _previousInterrupt = std::signal(SIGINT, requestCancellation);
        _previousTerminate = std::signal(SIGTERM, requestCancellation);
    }

    ~ScopedCancellationSignals()
    {
        if (_previousInterrupt != SIG_ERR)
        {
            std::signal(SIGINT, _previousInterrupt);
        }
        if (_previousTerminate != SIG_ERR)
        {
            std::signal(SIGTERM, _previousTerminate);
        }
    }

private:
    using SignalHandler = void (*)(int);
    SignalHandler _previousInterrupt = SIG_DFL;
    SignalHandler _previousTerminate = SIG_DFL;
};

QString smallBodyGlobalStageText(xjw::SmallBodyGlobalStage stage)
{
    switch (stage)
    {
    case xjw::SmallBodyGlobalStage::LoadSurface:
        return QStringLiteral("读取体固连表面模型");
    case xjw::SmallBodyGlobalStage::BuildSpatialIndex:
        return QStringLiteral("建立三角网 BVH");
    case xjw::SmallBodyGlobalStage::RasterizeGlobalProducts:
        return QStringLiteral("生成全球径向 DEM/DOM");
    case xjw::SmallBodyGlobalStage::WriteProducts:
        return QStringLiteral("写出全球 GeoTIFF");
    case xjw::SmallBodyGlobalStage::Completed:
        return QStringLiteral("全球 DEM/DOM 与报告完成");
    }
    return QStringLiteral("处理小天体全球 DEM/DOM");
}

} // namespace

int main(int argc, char *argv[])
{
#ifdef PLASCAN_HAS_TERRAIN_REPORT
    CLI::App app{"PlaScan 原生小天体全球径向 DEM/DOM 生成工具（默认生成 PNG 预览）"};
#else
    CLI::App app{"PlaScan 原生小天体全球径向 DEM/DOM 生成工具（当前构建仅输出 GeoTIFF/JSON）"};
#endif
    cli::configureApp(app);
    argv = app.ensure_utf8(argv);
    std::string surface;
    std::string output_dir;
    std::string target = "Small Body";
    std::string body_frame = "MODEL_LOCAL_BODY_FIXED";
    std::string surface_unit = "m";
    double angular_resolution = 0.25;
    double reference_radius = 0.0;
    double central_meridian = 0.0;
    double center_x = 0.0;
    double center_y = 0.0;
    double center_z = 0.0;
    long long maximum_pixels = 25000000;
    bool manual_center = false;
    bool no_preview = false;
    bool preview_requested = false;

    app.add_option("--surface", surface, "带三角面的体固连 PLY/OBJ 表面模型")->required();
    app.add_option("--output-dir", output_dir, "全球地形产品输出目录")->required();
    app.add_option("--target", target, "目标天体名称");
    app.add_option("--body-fixed-frame", body_frame, "体固连坐标系名称；禁止 J2000/ICRF");
    app.add_option("--surface-unit", surface_unit, "PLY/OBJ 顶点坐标单位：m 或 km");
    app.add_option("--angular-resolution-deg", angular_resolution, "经纬网角分辨率（度）");
    app.add_option("--reference-radius-m", reference_radius, "参考半径（米）；0 表示顶点半径中位数");
    app.add_option("--central-meridian-deg", central_meridian, "0° 栅格列对应的体固连经度");
    app.add_flag("--manual-center", manual_center, "使用显式体心；默认自动估计顶点均值");
    app.add_option("--center-x", center_x, "手动体心 X（米）");
    app.add_option("--center-y", center_y, "手动体心 Y（米）");
    app.add_option("--center-z", center_z, "手动体心 Z（米）");
    app.add_option("--maximum-pixels", maximum_pixels, "全球栅格最大像元数");
    app.add_flag("--no-preview", no_preview, "不生成四联图 PNG（GeoTIFF/JSON 仍生成）");
    app.add_flag("--preview", preview_requested, "生成四联图 PNG（需要启用 Qt 报告绘制）");

    CLI11_PARSE(app, argc, argv);

    if (no_preview && preview_requested)
    {
        cli::fatal("--preview 与 --no-preview 不能同时指定", cli::EXIT_ARG_ERR);
    }
    std::unique_ptr<QCoreApplication> qt_application;
    xjw::SmallBodyPreviewWriter preview_writer;
#ifdef PLASCAN_HAS_TERRAIN_REPORT
    if (!no_preview)
    {
        if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
        {
            qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
        }
        qt_application = std::make_unique<QGuiApplication>(argc, argv);
        preview_writer = xjw::GlobalTerrainReportRenderer::writePreview;
    }
#else
    if (preview_requested)
    {
        cli::fatal("本构建未启用地形报告绘制；请使用 PLASCAN_BUILD_QT_PRESENTATION=ON 的配置", cli::EXIT_ARG_ERR);
    }
    no_preview = true;
#endif
    if (!qt_application)
    {
        qt_application = std::make_unique<QCoreApplication>(argc, argv);
    }

    xjw::SmallBodyGlobalOptions options;
    options.targetName = QString::fromUtf8(target);
    options.bodyFixedFrame = QString::fromUtf8(body_frame);
    options.surfaceCoordinateUnit = QString::fromUtf8(surface_unit);
    options.automaticCenter = !manual_center;
    options.bodyCenter = cv::Vec3d(center_x, center_y, center_z);
    options.referenceRadiusM = reference_radius;
    options.angularResolutionDeg = angular_resolution;
    options.centralMeridianDeg = central_meridian;
    options.maximumPixelCount = maximum_pixels;
    options.writeReportPreview = !no_preview;

    QJsonObject result;
    QString error;
    ScopedCancellationSignals cancellation_signals;
    const auto progress = [](const xjw::SmallBodyGlobalProgress &progressEvent)
    {
        const QByteArray line = QStringLiteral("[%1%] %2\n")
                                    .arg(progressEvent.overallPercent)
                                    .arg(smallBodyGlobalStageText(progressEvent.stage))
                                    .toUtf8();
        std::fwrite(line.constData(), 1, static_cast<std::size_t>(line.size()), stderr);
        std::fflush(stderr);
    };
    if (!xjw::TerrainPipeline::generateSmallBodyGlobalProducts(QString::fromUtf8(surface),
                                                               QString::fromUtf8(output_dir),
                                                               options,
                                                               &result,
                                                               &error,
                                                               &gCancellationRequested,
                                                               progress,
                                                               preview_writer))
    {
        result[QStringLiteral("ok")] = false;
        result[QStringLiteral("error")] = error;
        xjw::cli::writeJson(stderr, result);
        return cli::EXIT_ALGO_ERR;
    }

    result[QStringLiteral("ok")] = true;
    xjw::cli::writeJson(stdout, result);
    return cli::EXIT_OK;
}
