#include "cli_common.h"
#include "CliJsonIO.h"

#include "TerrainPipeline.h"

#include <QGuiApplication>
#include <QJsonObject>
#include <QString>

#include <cstdio>
#include <atomic>
#include <csignal>
#include <string>

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

} // namespace

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 原生小天体全球径向 DEM/DOM 生成工具"};
    cli::configureApp(app);
    argv = app.ensure_utf8(argv);
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
    {
        qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
    }
    QGuiApplication qt_application(argc, argv);

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

    CLI11_PARSE(app, argc, argv);

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
    const auto progress = [](const QString &stage, int percent)
    {
        const QByteArray line = QStringLiteral("[%1%] %2\n").arg(percent).arg(stage).toUtf8();
        std::fwrite(line.constData(), 1, static_cast<std::size_t>(line.size()), stderr);
        std::fflush(stderr);
    };
    if (!xjw::TerrainPipeline::generateSmallBodyGlobalProducts(
            QString::fromUtf8(surface), QString::fromUtf8(output_dir),
            options, &result, &error, &gCancellationRequested, progress))
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
