// Standalone stereo DEM test: loads images + cameras, runs CUDA PatchMatch,
// generates DEM via both paths (point cloud and direct depth-to-DEM),
// compares with ASP reference.

#include "Camera.h"
#include "DemDomIO.h"
#include "DemDomTypes.h"
#include "DemGenerator.h"
#include "DenseCloudBuilder.h"
#include "DepthFrameUtils.h"
#include "DepthMapFusion.h"
#include "DepthMapGenerator.h"
#include "MvsTypes.h"

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QTimer>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <gdal_priv.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace xjw;
using namespace xjw::mvs;

// --- Test data paths ---
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
    "/home/guderian/code/plascan/data/test_stereo/stereo_test_output";

static void printSeparator(const char *title)
{
    printf("\n========================================\n");
    printf("  %s\n", title);
    printf("========================================\n");
}

static float computeCoverage(const cv::Mat &mask)
{
    if (mask.empty())
        return 0.0f;
    int valid = cv::countNonZero(mask);
    return 100.0f * static_cast<float>(valid) / static_cast<float>(mask.total());
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    bool skipMvs = false;
    for (int i = 1; i < argc; ++i)
        if (QString(argv[i]) == "--skip-mvs")
            skipMvs = true;

    printSeparator("PlaScan Stereo DEM Test");

    // --- 1. Load cameras ---
    printf("\n[1] Loading cameras...\n");
    Camera cam1, cam2;
    if (!cam1.loadFromFile(CAMERA1))
    {
        fprintf(stderr, "Failed to load camera: %s\n", CAMERA1);
        return 1;
    }
    if (!cam2.loadFromFile(CAMERA2))
    {
        fprintf(stderr, "Failed to load camera: %s\n", CAMERA2);
        return 1;
    }
    printf("  Camera 1: focal=(%.2f, %.2f) principal=(%.2f, %.2f) pitch=%.6f\n",
           cam1.focalX(), cam1.focalY(), cam1.principalX(), cam1.principalY(), cam1.pixelPitch());
    printf("  Camera 2: focal=(%.2f, %.2f) principal=(%.2f, %.2f) pitch=%.6f\n",
           cam2.focalX(), cam2.focalY(), cam2.principalX(), cam2.principalY(), cam2.pixelPitch());
    auto c1 = cam1.cameraCenter();
    auto c2 = cam2.cameraCenter();
    printf("  Camera 1 center: (%.4f, %.4f, %.4f)\n", c1[0], c1[1], c1[2]);
    printf("  Camera 2 center: (%.4f, %.4f, %.4f)\n", c2[0], c2[1], c2[2]);
    double baseline = std::sqrt((c2[0]-c1[0])*(c2[0]-c1[0]) +
                                (c2[1]-c1[1])*(c2[1]-c1[1]) +
                                (c2[2]-c1[2])*(c2[2]-c1[2]));
    printf("  Baseline: %.4f m\n", baseline);

    // --- 2. Load images ---
    printf("\n[2] Loading images...\n");
    cv::Mat img1 = cv::imread(IMAGE1, cv::IMREAD_UNCHANGED);
    cv::Mat img2 = cv::imread(IMAGE2, cv::IMREAD_UNCHANGED);
    if (img1.empty() || img2.empty())
    {
        fprintf(stderr, "Failed to load images\n");
        return 1;
    }
    printf("  Image 1: %dx%d type=%d\n", img1.cols, img1.rows, img1.type());
    printf("  Image 2: %dx%d type=%d\n", img2.cols, img2.rows, img2.type());

    // --- 3. Setup MVS views ---
    printf("\n[3] Setting up MVS views...\n");
    std::vector<CameraView> views(2);
    views[0].imagePath = IMAGE1;
    views[0].imageWidth = img1.cols;
    views[0].imageHeight = img1.rows;
    views[0].camera = cam1;
    views[1].imagePath = IMAGE2;
    views[1].imageWidth = img2.cols;
    views[1].imageHeight = img2.rows;
    views[1].camera = cam2;

    // --- 4. Configure depth generation (two-view stereo) ---
    printf("\n[4] Configuring depth generation...\n");
    DepthGenConfig genCfg;
    genCfg.numSourceViews = 1;
    genCfg.patchMatch.numSourceViews = 1;
    genCfg.patchMatch.downsampleFactor = 1;
    genCfg.patchMatch.numIterations = 32;
    genCfg.patchMatch.confidenceThresh = 0.0001f;
    genCfg.patchMatch.geomConsistency = false;
    genCfg.patchMatch.useCuda = true;
    genCfg.patchMatch.patchHalf = 7;
    genCfg.fusion.confidenceThresh = 0.0001f;
    genCfg.fusion.minConsistentViews = 1;
    genCfg.fusion.relDepthThresh = 0.40f;
    genCfg.fusion.pixelThresh = 10.0f;
    genCfg.runDepthEstimation = true;
    genCfg.runFusion = true;
    genCfg.saveIntermediateDepthMaps = true;

    // Create output directory
    QDir().mkpath(OUTPUT_DIR);
    genCfg.intermediateDir = OUTPUT_DIR;

    printf("  PatchMatch: iter=%d, patchHalf=%d, downsample=%d, cuda=%d\n",
           genCfg.patchMatch.numIterations, genCfg.patchMatch.patchHalf,
           genCfg.patchMatch.downsampleFactor, genCfg.patchMatch.useCuda);
    printf("  Fusion: minViews=%d, relDepth=%.2f, pixel=%.1f\n",
           genCfg.fusion.minConsistentViews, genCfg.fusion.relDepthThresh,
           genCfg.fusion.pixelThresh);

    // --- 5. Run MVS depth estimation + fusion ---
    std::vector<DensePoint> densePoints;
    std::vector<cv::Mat> filteredDepths;

    if (skipMvs)
    {
        printSeparator("Loading Existing Depth Maps (--skip-mvs)");
        for (int i = 0; i < 2; ++i)
        {
            const QString pngPath = QString("%1/depth_%2.png").arg(OUTPUT_DIR).arg(i);
            const QString path = xjw::core::project::rawDepthStoragePath(pngPath);
            cv::Mat dm;
            const xjw::common::OperationResult loadResult =
                xjw::core::project::loadDepthMatStorage(path, &dm);
            if (!loadResult.ok)
            {
                fprintf(stderr, "%s\n", loadResult.errorMessage.toUtf8().constData());
                return 1;
            }
            filteredDepths.push_back(dm);
            int validPx = cv::countNonZero(dm > 0);
            printf("  Loaded depth_%d: %dx%d, valid=%d (%.1f%%)\n",
                   i, dm.cols, dm.rows, validPx, 100.0 * validPx / (dm.rows * dm.cols));
        }
    }
    else
    {
    printSeparator("Running CUDA PatchMatch + Fusion");

    auto *gen = new DepthMapGenerator(&app);
    gen->setViews(views);
    gen->setConfig(genCfg);
    gen->setOutputDir(OUTPUT_DIR);

    // Empty sparse cloud (no SfM triangulation for two-view test)
    SparseCloud sparse;
    gen->setSparseCloud(sparse);

    bool mvsSuccess = false;

    QEventLoop loop;
    QObject::connect(gen, &DepthMapGenerator::progressChanged,
                     [](const QString &stage, float ratio) {
        printf("  [MVS] %s: %.1f%%\n", stage.toUtf8().constData(), ratio * 100.0f);
    });
    QObject::connect(gen, &DepthMapGenerator::errorOccurred,
                     [](const QString &msg) {
        fprintf(stderr, "  [MVS ERROR] %s\n", msg.toUtf8().constData());
    });
    QObject::connect(gen, &DepthMapGenerator::pointCloudReady,
                     [&densePoints](std::vector<DensePoint> cloud) {
        densePoints = std::move(cloud);
        printf("  [MVS] Point cloud ready: %zu points\n", densePoints.size());
    });
    QObject::connect(gen, &DepthMapGenerator::finished,
                     [&mvsSuccess, &loop](bool success) {
        mvsSuccess = success;
        loop.quit();
    });

    // Start with a timer to avoid blocking before event loop
    QTimer::singleShot(0, gen, &DepthMapGenerator::start);
    loop.exec();

    if (!mvsSuccess)
    {
        fprintf(stderr, "\nMVS pipeline failed!\n");
        delete gen;
        return 1;
    }

    printf("\n  MVS completed successfully.\n");
    printf("  Dense points from fusion: %zu\n", densePoints.size());

    // Get filtered depth maps for direct DEM generation
    filteredDepths = gen->filteredDepths();
    printf("  Filtered depth maps: %zu\n", filteredDepths.size());
    delete gen;
    } // end !skipMvs

    // --- 6. Analyze point cloud distribution ---
    if (!densePoints.empty()) {
    printSeparator("Point Cloud Analysis");
    {
        float minX = 1e30f, maxX = -1e30f;
        float minY = 1e30f, maxY = -1e30f;
        float minZ = 1e30f, maxZ = -1e30f;
        for (const auto &p : densePoints)
        {
            minX = std::min(minX, p.x); maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y); maxY = std::max(maxY, p.y);
            minZ = std::min(minZ, p.z); maxZ = std::max(maxZ, p.z);
        }
        printf("  Points: %zu\n", densePoints.size());
        printf("  X range: [%.4f, %.4f] span=%.4f\n", minX, maxX, maxX - minX);
        printf("  Y range: [%.4f, %.4f] span=%.4f\n", minY, maxY, maxY - minY);
        printf("  Z range: [%.4f, %.4f] span=%.4f\n", minZ, maxZ, maxZ - minZ);

        // Compute percentiles to understand distribution
        std::vector<float> xs, ys, zs;
        xs.reserve(densePoints.size());
        ys.reserve(densePoints.size());
        zs.reserve(densePoints.size());
        for (const auto &p : densePoints)
        {
            xs.push_back(p.x);
            ys.push_back(p.y);
            zs.push_back(p.z);
        }
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());
        std::sort(zs.begin(), zs.end());
        size_t n = densePoints.size();
        size_t p1 = n / 100, p5 = n * 5 / 100, p95 = n * 95 / 100, p99 = n * 99 / 100;
        printf("  X percentiles: 1%%=%.4f 5%%=%.4f 95%%=%.4f 99%%=%.4f\n",
               xs[p1], xs[p5], xs[p95], xs[p99]);
        printf("  Y percentiles: 1%%=%.4f 5%%=%.4f 95%%=%.4f 99%%=%.4f\n",
               ys[p1], ys[p5], ys[p95], ys[p99]);
        printf("  Z percentiles: 1%%=%.4f 5%%=%.4f 95%%=%.4f 99%%=%.4f\n",
               zs[p1], zs[p5], zs[p95], zs[p99]);
    }
    } // end point cloud analysis

    // --- 7. Generate DEM from point cloud with increased grid size ---
    if (!densePoints.empty()) {
    printSeparator("DEM from Point Cloud");
    {
        pointcloud::PointCloud pc;
        pc.reserve(densePoints.size());
        for (const auto &p : densePoints)
        {
            pc.addPoint({p.x, p.y, p.z}, {p.r, p.g, p.b, 255});
        }
        printf("  PointCloud size: %zu\n", pc.size());

        DemGenerationOptions opts;
        opts.gridResolution = 0.0; // auto
        opts.maxGridSize = 8192;
        opts.holeFillIterations = 20;
        opts.holeFillMinNeighbors = 3;
        opts.holeFillSearchRadius = 5;
        opts.useSubPixelBilinearSplat = true;
        opts.generateDenseCloud = false;
        opts.generateMesh = false;

        DemGridData demGrid;
        pointcloud::PointCloud denseOut;
        QString errMsg;
        if (!DemGenerator::generateFromPointCloud(pc, opts, &demGrid, &denseOut, &errMsg))
        {
            fprintf(stderr, "  DEM from point cloud failed: %s\n", errMsg.toUtf8().constData());
        }
        else
        {
            float coverage = computeCoverage(demGrid.validMask);
            printf("  DEM size: %dx%d\n", demGrid.width, demGrid.height);
            printf("  DEM coverage: %.2f%%\n", coverage);

            QString demPath = QString("%1/dem_from_pointcloud.tif").arg(OUTPUT_DIR);
            QString previewPath = QString("%1/dem_from_pointcloud_preview.png").arg(OUTPUT_DIR);
            DemDomIO::writeDemRaster(demGrid, demPath, DemRasterFormat::Float32Tiff);
            DemDomIO::writeDemPreviewPng(demGrid, previewPath);
            printf("  Saved: %s\n", demPath.toUtf8().constData());
        }
    }
    } // end DEM from point cloud

    // --- 8. Generate DEM directly from depth maps (new path) ---
    printSeparator("DEM from Depth Maps (Direct)");
    {
        if (filteredDepths.empty())
        {
            printf("  No filtered depth maps available, skipping.\n");
        }
        else
        {
            std::vector<Camera> cameras = {cam1, cam2};
            // Only use as many cameras as we have depth maps
            if (filteredDepths.size() < cameras.size())
                cameras.resize(filteredDepths.size());

            for (size_t i = 0; i < filteredDepths.size(); ++i)
            {
                const auto &d = filteredDepths[i];
                int valid = cv::countNonZero(d > 0);
                float cov = 100.0f * static_cast<float>(valid) / static_cast<float>(d.total());
                printf("  Depth map %zu: %dx%d, valid=%.2f%%\n",
                       i, d.cols, d.rows, cov);
            }

            DemGenerationOptions opts;
            opts.gridResolution = 0.0;
            opts.maxGridSize = 8192;
            opts.holeFillIterations = 20;
            opts.holeFillMinNeighbors = 3;
            opts.holeFillSearchRadius = 5;

            DemGridData demGrid;
            QString errMsg;
            if (!DemGenerator::generateFromDepthMaps(filteredDepths, cameras, opts, &demGrid, &errMsg))
            {
                fprintf(stderr, "  DEM from depth maps failed: %s\n", errMsg.toUtf8().constData());
            }
            else
            {
                float coverage = computeCoverage(demGrid.validMask);
                printf("  DEM size: %dx%d\n", demGrid.width, demGrid.height);
                printf("  DEM coverage: %.2f%%\n", coverage);

                QString demPath = QString("%1/dem_from_depth.tif").arg(OUTPUT_DIR);
                QString previewPath = QString("%1/dem_from_depth_preview.png").arg(OUTPUT_DIR);
                DemDomIO::writeDemRaster(demGrid, demPath, DemRasterFormat::Float32Tiff);
                DemDomIO::writeDemPreviewPng(demGrid, previewPath);
                printf("  Saved: %s\n", demPath.toUtf8().constData());
            }
        }
    }

    // --- 9. Compare with ASP reference point cloud ---
    printSeparator("Comparison with ASP run-PC.tif");
    {
        // Read ASP 4-band point cloud (X, Y, Z, intersection_error)
        GDALAllRegister();
        GDALDataset *aspDs = static_cast<GDALDataset *>(
            GDALOpen(ASP_PC, GA_ReadOnly));
        if (!aspDs)
        {
            printf("  Could not open ASP PC: %s\n", ASP_PC);
        }
        else
        {
            int aspW = aspDs->GetRasterXSize();
            int aspH = aspDs->GetRasterYSize();
            int aspB = aspDs->GetRasterCount();
            printf("  ASP PC: %dx%d, %d bands\n", aspW, aspH, aspB);

            if (aspB >= 3)
            {
                cv::Mat aspX(aspH, aspW, CV_32FC1);
                cv::Mat aspY(aspH, aspW, CV_32FC1);
                cv::Mat aspZ(aspH, aspW, CV_32FC1);
                aspDs->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, aspW, aspH,
                    aspX.data, aspW, aspH, GDT_Float32, 0, 0);
                aspDs->GetRasterBand(2)->RasterIO(GF_Read, 0, 0, aspW, aspH,
                    aspY.data, aspW, aspH, GDT_Float32, 0, 0);
                aspDs->GetRasterBand(3)->RasterIO(GF_Read, 0, 0, aspW, aspH,
                    aspZ.data, aspW, aspH, GDT_Float32, 0, 0);

                // ASP valid mask: any non-zero coordinate
                int aspValid = 0;
                double aspRSum = 0;
                std::vector<float> aspRadii;
                for (int r = 0; r < aspH; ++r)
                    for (int c = 0; c < aspW; ++c)
                    {
                        float x = aspX.at<float>(r, c);
                        float y = aspY.at<float>(r, c);
                        float z = aspZ.at<float>(r, c);
                        if (x != 0 || y != 0 || z != 0)
                        {
                            ++aspValid;
                            float rad = std::sqrt(x*x + y*y + z*z);
                            aspRSum += rad;
                            aspRadii.push_back(rad);
                        }
                    }
                std::sort(aspRadii.begin(), aspRadii.end());
                printf("  ASP valid points: %d (%.1f%%)\n", aspValid,
                       100.0 * aspValid / (aspW * aspH));
                if (!aspRadii.empty())
                {
                    size_t n = aspRadii.size();
                    printf("  ASP radius: min=%.6f P5=%.6f median=%.6f P95=%.6f max=%.6f\n",
                           aspRadii[0], aspRadii[n*5/100], aspRadii[n/2],
                           aspRadii[n*95/100], aspRadii[n-1]);
                }

                // Now read PlaScan DEM (4-band XYZ + mask)
                QString plascanPath = QString("%1/dem_from_depth.tif").arg(OUTPUT_DIR);
                GDALDataset *plDs = static_cast<GDALDataset *>(
                    GDALOpen(plascanPath.toStdString().c_str(), GA_ReadOnly));
                if (plDs && plDs->GetRasterCount() >= 3)
                {
                    int plW = plDs->GetRasterXSize();
                    int plH = plDs->GetRasterYSize();
                    printf("\n  PlaScan PC: %dx%d, %d bands\n", plW, plH, plDs->GetRasterCount());

                    cv::Mat plX(plH, plW, CV_32FC1);
                    cv::Mat plY(plH, plW, CV_32FC1);
                    cv::Mat plZ(plH, plW, CV_32FC1);
                    plDs->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, plW, plH,
                        plX.data, plW, plH, GDT_Float32, 0, 0);
                    plDs->GetRasterBand(2)->RasterIO(GF_Read, 0, 0, plW, plH,
                        plY.data, plW, plH, GDT_Float32, 0, 0);
                    plDs->GetRasterBand(3)->RasterIO(GF_Read, 0, 0, plW, plH,
                        plZ.data, plW, plH, GDT_Float32, 0, 0);

                    float nodata = -3.40282e+38f;
                    int plValid = 0;
                    std::vector<float> plRadii;
                    for (int r = 0; r < plH; ++r)
                        for (int c = 0; c < plW; ++c)
                        {
                            float x = plX.at<float>(r, c);
                            float y = plY.at<float>(r, c);
                            float z = plZ.at<float>(r, c);
                            if (x != nodata && y != nodata && z != nodata &&
                                (x != 0 || y != 0 || z != 0))
                            {
                                ++plValid;
                                plRadii.push_back(std::sqrt(x*x + y*y + z*z));
                            }
                        }
                    std::sort(plRadii.begin(), plRadii.end());
                    printf("  PlaScan valid points: %d (%.1f%%)\n", plValid,
                           100.0 * plValid / (plW * plH));
                    if (!plRadii.empty())
                    {
                        size_t n = plRadii.size();
                        printf("  PlaScan radius: min=%.6f P5=%.6f median=%.6f P95=%.6f max=%.6f\n",
                               plRadii[0], plRadii[n*5/100], plRadii[n/2],
                               plRadii[n*95/100], plRadii[n-1]);
                    }

                    // Compare distributions
                    if (!aspRadii.empty() && !plRadii.empty())
                    {
                        printf("\n  --- Distribution Comparison ---\n");
                        printf("  ASP  median radius: %.6f\n", aspRadii[aspRadii.size()/2]);
                        printf("  PL   median radius: %.6f\n", plRadii[plRadii.size()/2]);
                        printf("  ASP  IQR: %.6f\n",
                               aspRadii[aspRadii.size()*75/100] - aspRadii[aspRadii.size()*25/100]);
                        printf("  PL   IQR: %.6f\n",
                               plRadii[plRadii.size()*75/100] - plRadii[plRadii.size()*25/100]);
                    }

                    GDALClose(plDs);
                }
                else
                {
                    printf("  PlaScan DEM not found or not 4-band: %s\n",
                           plascanPath.toUtf8().constData());
                }
            }
            GDALClose(aspDs);
        }
    }

    printSeparator("Test Complete");
    printf("\nOutput files in: %s\n", OUTPUT_DIR);

    return 0;
}
