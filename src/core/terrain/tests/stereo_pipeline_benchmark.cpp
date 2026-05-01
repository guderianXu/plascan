#include "AspPointCloudMetrics.h"
#include "StereoDenseCloudPipeline.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QString>
#include <QTextStream>

#include <cstdio>
#include <string>

using namespace xjw;
using namespace xjw::mvs;

namespace
{

const char *IMAGE1 = "/home/guderian/code/plascan/data/stereo_test_20260426/20260413T174329163_NAS_PAN_L2b.tif";
const char *IMAGE2 = "/home/guderian/code/plascan/data/stereo_test_20260426/20260413T174419164_NAS_PAN_L2b.tif";
const char *CAMERA1 = "/home/guderian/code/plascan/data/stereo_test_20260426/ba-tsai_20260413T174329163_NAS_PAN_L2b.tsai";
const char *CAMERA2 = "/home/guderian/code/plascan/data/stereo_test_20260426/ba-tsai_20260413T174419164_NAS_PAN_L2b.tsai";
const char *ASP_PC = "/home/guderian/code/plascan/data/stereo_test_20260426/run-PC.tif";
const char *OUTPUT_DIR = "/home/guderian/code/plascan/data/test_stereo/stereo_pipeline_benchmark";

StereoPipelineConfig strictBenchmarkConfig()
{
    StereoPipelineConfig cfg;
    cfg.patchMatch.numIterations = 32;
    cfg.patchMatch.patchHalf = 7;
    cfg.patchMatch.confidenceThresh = 0.0001f;
    cfg.patchMatch.downsampleFactor = 1;
    cfg.patchMatch.numSourceViews = 1;
    cfg.triangulation.maxTriangulationError = 0.01f;
    cfg.subpixel.mode = 1;
    cfg.outputTif = true;
    cfg.outputPly = true;
    return cfg;
}

StereoPipelineConfig configForProfile(const QString &profile)
{
    StereoPipelineConfig cfg = strictBenchmarkConfig();
    if (profile == QStringLiteral("no-iqr"))
    {
        cfg.filters.enableIqrFilter = false;
    }
    else if (profile == QStringLiteral("no-local"))
    {
        cfg.filters.enableLocalDepthConsistency = false;
    }
    else if (profile == QStringLiteral("no-lr"))
    {
        cfg.filters.enableLeftRightDepthCheck = false;
    }
    else if (profile == QStringLiteral("asp-rectified"))
    {
        cfg.geometryMode = StereoPipelineGeometryMode::RectifiedDisparity;
    }
    else if (profile == QStringLiteral("asp-rectified-no-disp-filter"))
    {
        cfg.geometryMode = StereoPipelineGeometryMode::RectifiedDisparity;
        cfg.filters.enableIqrFilter = false;
        cfg.filters.enableLocalDepthConsistency = false;
        cfg.filters.enableLeftRightDepthCheck = false;
        cfg.disparityFilter.medianFilterSize = 0;
        cfg.disparityFilter.speckleSize = 0;
        cfg.triangulation.maxTriangulationError = 0.01f;
    }
    else if (profile == QStringLiteral("wide-depth"))
    {
        cfg.depthRange.nearScale = 0.4;
        cfg.depthRange.farScale = 2.2;
    }
    else if (profile == QStringLiteral("loose-all"))
    {
        cfg.filters.leftRightDepthRatio = 0.20f;
        cfg.filters.localDepthRatio = 0.10f;
        cfg.filters.iqrMultiplier = 4.0f;
        cfg.triangulation.maxTriangulationError = 0.05f;
    }
    return cfg;
}

bool writeMetricsJson(const QString &path, const AspPointCloudMetricsResult &metrics)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        return false;
    }

    QTextStream stream(&file);
    stream << QString::fromStdString(AspPointCloudMetrics::toJson(metrics));
    return true;
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    Q_UNUSED(app);
    QDir().mkpath(QString::fromUtf8(OUTPUT_DIR));

    const QString profile = argc > 1 ? QString::fromLocal8Bit(argv[1]) : QStringLiteral("strict");
    std::printf("[benchmark] profile=%s\n", profile.toUtf8().constData());

    StereoDenseCloudPipeline pipeline;
    pipeline.setConfig(configForProfile(profile));
    QObject::connect(&pipeline, &StereoDenseCloudPipeline::progressChanged,
                     [](const QString &stage, float ratio)
    {
        std::printf("[benchmark] %s %.1f%%\n", stage.toUtf8().constData(), ratio * 100.0f);
    });

    StereoPipelineResult pipelineResult;
    const bool ok = pipeline.run(IMAGE1, IMAGE2, CAMERA1, CAMERA2, OUTPUT_DIR, &pipelineResult);
    if (!ok)
    {
        std::fprintf(stderr, "Pipeline failed: %s\n", pipelineResult.errorMsg.c_str());
        return 2;
    }

    AspPointCloudMetricsThresholds thresholds;
    AspPointCloudMetricsResult metrics;
    std::string metricsError;
    if (!AspPointCloudMetrics::compare(pipelineResult.tifPath, ASP_PC, thresholds, metrics, &metricsError))
    {
        std::fprintf(stderr, "Metrics failed: %s\n", metricsError.c_str());
        return 3;
    }

    const QString jsonPath = QDir(QString::fromUtf8(OUTPUT_DIR)).filePath(
        QStringLiteral("metrics_%1.json").arg(profile));
    if (!writeMetricsJson(jsonPath, metrics))
    {
        std::fprintf(stderr, "Failed to write metrics JSON: %s\n", jsonPath.toUtf8().constData());
        return 4;
    }

    std::printf("\n%s", AspPointCloudMetrics::toTextReport(metrics).c_str());
    std::printf("Metrics JSON: %s\n", jsonPath.toUtf8().constData());
    return metrics.passed ? 0 : 1;
}
