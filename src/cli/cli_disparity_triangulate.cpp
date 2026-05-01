// =============================================================================
// 文件: cli_disparity_triangulate.cpp
// 功能: 视差三角化 CLI (基于 CLI11) — 视差图 + 相机 → 密集点云 .ply
// 用法:
//   triangulate_cli -d disparity.tif --rect rect.xml \
//       --camL camL.txt --camR camR.txt -o cloud.ply
// =============================================================================
#include "cli_common.h"
#include "DisparityTriangulator.h"
#include "Camera.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <string>

int main(int argc, char *argv[])
{
    CLI::App app{"PlaScan 视差三角化工具 — 视差图 → 密集点云 .ply"};

    std::string dispPath, rectPath, camL, camR, outPath;
    app.add_option("-d,--disparity", dispPath, "视差图路径 (.tif)")->required();
    app.add_option("--rect", rectPath, "校正参数文件 (.xml)")->required();
    app.add_option("--camL", camL,    "左相机文件路径")->required();
    app.add_option("--camR", camR,    "右相机文件路径")->required();
    app.add_option("-o,--output", outPath, "输出点云路径 (.ply)")->required();

    float maxError = 0.01f;
    int   threads  = 4;
    app.add_option("--max-error", maxError, "最大三角化误差 (m)");
    app.add_option("--threads",   threads,  "线程数");

    bool verbose = false;
    app.add_flag("-V,--verbose", verbose, "详细诊断日志");

    CLI11_PARSE(app, argc, argv);

    // 加载视差图
    cv::Mat disparity = cv::imread(dispPath, cv::IMREAD_UNCHANGED);
    if (disparity.empty())
        cli::fatal("无法加载视差图: " + dispPath, cli::EXIT_IO_ERR);
    if (disparity.type() != CV_32FC1)
        disparity.convertTo(disparity, CV_32FC1);

    // 加载校正参数
    cv::FileStorage fs(rectPath, cv::FileStorage::READ);
    if (!fs.isOpened())
        cli::fatal("无法加载校正参数: " + rectPath, cli::EXIT_IO_ERR);
    cv::Mat H1inv, H2inv;
    fs["H1inv"] >> H1inv;
    fs["H2inv"] >> H2inv;
    fs.release();

    // 加载相机
    xjw::Camera camLObj, camRObj;
    if (!camLObj.loadFromFile(camL))
        cli::fatal("无法加载左相机: " + camL, cli::EXIT_IO_ERR);
    if (!camRObj.loadFromFile(camR))
        cli::fatal("无法加载右相机: " + camR, cli::EXIT_IO_ERR);

    fprintf(stdout, "三角化: %s -> %s\n", dispPath.c_str(), outPath.c_str());

    // 有效掩码: 视差 > 0
    cv::Mat validMask(disparity.size(), CV_8UC1);
    for (int y = 0; y < disparity.rows; ++y)
        for (int x = 0; x < disparity.cols; ++x)
            validMask.at<uchar>(y, x) = (disparity.at<float>(y, x) > 0) ? 255 : 0;

    // 三角化
    xjw::mvs::TriangulationConfig triCfg;
    triCfg.maxTriangulationError = maxError;
    triCfg.numThreads            = threads;

    auto result = xjw::mvs::DisparityTriangulator::triangulate(
        disparity, validMask, H1inv, H2inv, camLObj, camRObj, triCfg);

    if (verbose)
    {
        fprintf(stdout, "  有效点: %d / %d\n", result.validPoints, result.totalPixels);
        fprintf(stdout, "  中位误差: %.4f m\n", result.medianError);
    }

    // 写入 PLY
    FILE *ply = fopen(outPath.c_str(), "w");
    if (!ply)
        cli::fatal("无法写入: " + outPath, cli::EXIT_IO_ERR);

    // 计数有效点
    int count = 0;
    for (int y = 0; y < result.pointCloud.rows; ++y)
        for (int x = 0; x < result.pointCloud.cols; ++x)
            if (result.validMask.at<uchar>(y, x)) ++count;

    fprintf(ply, "ply\nformat ascii 1.0\n"
            "element vertex %d\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float error\n"
            "end_header\n", count);

    for (int y = 0; y < result.pointCloud.rows; ++y)
    {
        for (int x = 0; x < result.pointCloud.cols; ++x)
        {
            if (!result.validMask.at<uchar>(y, x)) continue;
            auto pt = result.pointCloud.at<cv::Vec3d>(y, x);
            double err = result.errorMap.empty() ? 0.0
                        : result.errorMap.at<float>(y, x);
            fprintf(ply, "%.6f %.6f %.6f %.6f\n",
                    pt[0] - result.pointOffset[0],
                    pt[1] - result.pointOffset[1],
                    pt[2] - result.pointOffset[2], err);
        }
    }
    fclose(ply);

    fprintf(stdout, "点云已保存: %s (%d 点)\n", outPath.c_str(), count);
    return cli::EXIT_OK;
}
