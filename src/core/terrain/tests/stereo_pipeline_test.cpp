// Stereo dense cloud pipeline test: runs StereoDenseCloudPipeline,
// outputs TIF + PLY, compares with ASP run-PC.tif reference.

#include "StereoDenseCloudPipeline.h"
#include "PointCloudTifIO.h"
#include "AspPointCloudMetrics.h"
#include "MvsTypes.h"

#include <QCoreApplication>
#include <QDir>

#include <cstdio>
#include <string>

using namespace xjw;
using namespace xjw::mvs;

static const char *IMAGE1 =
    "/home/guderian/code/plascan/data/stereo_test_20260426/"
    "20260413T174329163_NAS_PAN_L2b.tif";
static const char *IMAGE2 =
    "/home/guderian/code/plascan/data/stereo_test_20260426/"
    "20260413T174419164_NAS_PAN_L2b.tif";
static const char *CAMERA1 =
    "/home/guderian/code/plascan/data/stereo_test_20260426/"
    "ba-tsai_20260413T174329163_NAS_PAN_L2b.tsai";
static const char *CAMERA2 =
    "/home/guderian/code/plascan/data/stereo_test_20260426/"
    "ba-tsai_20260413T174419164_NAS_PAN_L2b.tsai";
static const char *ASP_PC =
    "/home/guderian/code/plascan/data/stereo_test_20260426/run-PC.tif";
static const char *OUTPUT_DIR =
    "/home/guderian/code/plascan/data/test_stereo/stereo_pipeline_output";

static void printSep(const char *title)
{
    printf("\n========================================\n");
    printf("  %s\n", title);
    printf("========================================\n");
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    printSep("StereoDenseCloudPipeline Test");

    // === 1. Run pipeline ===
    printSep("Running StereoDenseCloudPipeline");

    StereoDenseCloudPipeline pipeline;
    StereoPipelineConfig cfg;
    cfg.patchMatch.numIterations = 32;
    cfg.patchMatch.patchHalf = 7;
    cfg.patchMatch.confidenceThresh = 0.0001f;
    cfg.patchMatch.downsampleFactor = 1;
    cfg.triangulation.maxTriangulationError = 0.01f;
    cfg.subpixel.mode = 1;
    pipeline.setConfig(cfg);

    QObject::connect(&pipeline, &StereoDenseCloudPipeline::progressChanged,
                     [](const QString &stage, float ratio)
    {
        printf("  [Pipeline] %s: %.1f%%\n", stage.toUtf8().constData(), ratio * 100.0f);
    });

    StereoPipelineResult pipeResult;
    bool ok = pipeline.run(IMAGE1, IMAGE2, CAMERA1, CAMERA2, OUTPUT_DIR, &pipeResult);

    if (!ok)
    {
        fprintf(stderr, "\nPipeline FAILED: %s\n", pipeResult.errorMsg.c_str());
        return 1;
    }

    printf("\n  Pipeline succeeded:\n");
    printf("    Valid points: %d / %d (%.1f%%)\n",
           pipeResult.validPoints, pipeResult.totalPoints, pipeResult.coveragePercent);
    printf("    Median tri error: %.6f\n", pipeResult.medianTriError);
    printf("    TIF: %s\n", pipeResult.tifPath.c_str());
    printf("    PLY: %s\n", pipeResult.plyPath.c_str());

    // === 2. Verify output TIF ===
    printSep("Verify Output TIF");
    {
        TriangulationResult outTri;
        std::string readErr;
        if (!PointCloudTifIO::readTif(pipeResult.tifPath, outTri, &readErr))
        {
            fprintf(stderr, "  Failed to read output TIF: %s\n", readErr.c_str());
            return 1;
        }
        printf("  Output TIF: %dx%d, valid=%d\n",
               outTri.pointCloud.cols, outTri.pointCloud.rows, outTri.validPoints);
        printf("  POINT_OFFSET: (%.6f, %.6f, %.6f)\n",
               outTri.pointOffset[0], outTri.pointOffset[1], outTri.pointOffset[2]);
    }

    // === 3. Compare with ASP reference ===
    printSep("Comparison with ASP run-PC.tif");
    {
        AspPointCloudMetricsThresholds thresholds;
        AspPointCloudMetricsResult metrics;
        std::string metricsError;
        if (!AspPointCloudMetrics::compare(pipeResult.tifPath, ASP_PC, thresholds, metrics, &metricsError))
        {
            fprintf(stderr, "  ASP comparison failed: %s\n", metricsError.c_str());
            return 1;
        }

        printf("\n%s", AspPointCloudMetrics::toTextReport(metrics).c_str());
    }

    printSep("Test Complete");
    printf("\nOutput files in: %s\n", OUTPUT_DIR);
    return 0;
}
