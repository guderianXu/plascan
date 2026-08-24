// =============================================================================
// 文件: cli_dense_match.cpp
// 功能: 密集匹配 CLI (基于 CLI11)
// 用法:
//   dense_match_cli -L imgL.tif -R imgR.tif -o disp.tif
//                   [--device auto|cpu|cuda|opencl] [选项]
// =============================================================================
#include "cli_common.h"
#include "DenseMatchBackend.h"
#include "DenseMatchService.h"
#include "DenseMatchConfig.h"
#include "DenseMatchTypes.h"

#include <opencv2/core.hpp>
#include <exception>
#include <new>
#include <string>

int main(int argc, char* argv[])
{
    CLI::App app{"PlaScan 密集匹配工具 — 立体密集匹配 (MGM/SGM/BM)"};
    cli::configureApp(app);

    std::string imgL, imgR, outPath;
    app.add_option("-L,--left", imgL, "左影像路径")->required();
    app.add_option("-R,--right", imgR, "右影像路径")->required();
    app.add_option("-o,--output", outPath, "输出视差图路径 (.tif)")->required();

    std::string algoStr = "mgm";
    app.add_option("-a,--algorithm", algoStr, "匹配算法: bm, sgm, mgm, opencv_sgbm");

    std::string costStr = "census";
    app.add_option("-f,--cost-func", costStr, "代价函数: ad, sd, ncc, census, ternary");

    std::string subpixelStr = "parabola";
    app.add_option("--subpixel", subpixelStr, "子像素: none, parabola");

    int minDisp = 0, maxDisp = 256;
    app.add_option("--min-disp", minDisp, "最小视差 (px)");
    app.add_option("--max-disp", maxDisp, "最大视差 (px)");

    int kernelW = 15, kernelH = 15;
    app.add_option("--kernel-w", kernelW, "相关核宽度 (px)");
    app.add_option("--kernel-h", kernelH, "相关核高度 (px)");

    int p1 = 8, p2 = 32, directions = 8, pyramid = 2;
    app.add_option("--p1", p1, "SGM 小惩罚");
    app.add_option("--p2", p2, "SGM 大惩罚");
    app.add_option("--directions", directions, "SGM 路径方向数 (4|8)");
    app.add_option("--pyramid", pyramid, "金字塔层数");

    std::string deviceStr = "auto";
    CLI::Option* deviceOption = app.add_option("--device", deviceStr, "计算设备: auto, cpu, cuda, opencl")
                                    ->check(CLI::IsMember({"auto", "cpu", "cuda", "opencl"}));

    bool legacyCuda = false;
    bool legacyNoCuda = false;
    app.add_flag("--cuda", legacyCuda, "兼容选项：等价于 --device cuda");
    app.add_flag("--no-cuda", legacyNoCuda, "兼容选项：等价于 --device cpu");

    int gpu = 0, openClDevice = 0, threads = 4;
    app.add_option("--gpu", gpu, "CUDA 设备 ID");
    app.add_option("--opencl-device", openClDevice, "OpenCL GPU 设备 ID");
    app.add_option("--threads", threads, "CPU 线程数");

    float lrThresh = 1.0f;
    int medianFilter = 3;
    app.add_option("--lr-threshold", lrThresh, "L-R 一致性阈值 (px)");
    app.add_option("--median-filter", medianFilter, "中值滤波核 (0=禁用)");

    bool verbose = false;
    app.add_flag("-V,--verbose", verbose, "详细诊断日志");

    CLI11_PARSE(app, argc, argv);

    if (legacyCuda && legacyNoCuda)
    {
        cli::fatal("--cuda 与 --no-cuda 不能同时使用");
    }
    if (deviceOption->count() > 0 && (legacyCuda || legacyNoCuda))
    {
        cli::fatal("--device 不能与兼容选项 --cuda/--no-cuda 同时使用");
    }

    if (imgL.empty() || imgR.empty() || outPath.empty())
        cli::fatal("必须指定 -L, -R, -o");

    // 解析枚举
    auto parseAlgo = [](const std::string& s) -> xjw::dense_match::StereoAlgorithm
    {
        if (s == "bm")
            return xjw::dense_match::StereoAlgorithm::BlockMatch;
        if (s == "sgm")
            return xjw::dense_match::StereoAlgorithm::SemiGlobalMatch;
        if (s == "mgm")
            return xjw::dense_match::StereoAlgorithm::MoreGlobalMatch;
        if (s == "opencv_sgbm")
            return xjw::dense_match::StereoAlgorithm::OpenCV_SGBM;
        cli::fatal("未知算法: " + s);
        return xjw::dense_match::StereoAlgorithm::MoreGlobalMatch;
    };

    auto parseCost = [](const std::string& s) -> xjw::dense_match::CostFunction
    {
        if (s == "ad")
            return xjw::dense_match::CostFunction::AbsoluteDifference;
        if (s == "sd")
            return xjw::dense_match::CostFunction::SquaredDifference;
        if (s == "ncc")
            return xjw::dense_match::CostFunction::NormalizedCrossCorr;
        if (s == "census")
            return xjw::dense_match::CostFunction::CensusTransform;
        if (s == "ternary")
            return xjw::dense_match::CostFunction::TernaryCensusTransform;
        cli::fatal("未知代价函数: " + s);
        return xjw::dense_match::CostFunction::CensusTransform;
    };

    auto parseSubpixel = [](const std::string& s) -> xjw::dense_match::SubpixelMode
    {
        if (s == "none")
            return xjw::dense_match::SubpixelMode::None;
        if (s == "parabola")
            return xjw::dense_match::SubpixelMode::Parabola;
        cli::fatal("未知子像素模式: " + s);
        return xjw::dense_match::SubpixelMode::Parabola;
    };

    xjw::dense_match::DenseMatchComputeBackend computeBackend =
        xjw::dense_match::parseDenseMatchComputeBackend(deviceStr);
    if (legacyCuda)
    {
        computeBackend = xjw::dense_match::DenseMatchComputeBackend::Cuda;
    }
    else if (legacyNoCuda)
    {
        computeBackend = xjw::dense_match::DenseMatchComputeBackend::Cpu;
    }

    // 构建配置
    xjw::dense_match::DenseMatchConfig cfg;
    cfg.algorithm = parseAlgo(algoStr);
    cfg.costFunc = parseCost(costStr);
    cfg.subpixel = parseSubpixel(subpixelStr);
    cfg.minDisparity = minDisp;
    cfg.maxDisparity = maxDisp;
    cfg.corrKernelW = kernelW;
    cfg.corrKernelH = kernelH;
    cfg.p1 = p1;
    cfg.p2 = p2;
    cfg.sgmDirections = directions;
    cfg.pyramidLevels = pyramid;
    cfg.computeBackend = computeBackend;
    cfg.useCuda = computeBackend != xjw::dense_match::DenseMatchComputeBackend::Cpu;
    cfg.cudaDevice = gpu;
    cfg.openClDevice = openClDevice;
    cfg.numThreads = threads;
    cfg.lrCheckThreshold = lrThresh;
    cfg.medianFilterSize = medianFilter;
    cfg.enableLRCheck = lrThresh > 0.0f;
    cfg.leftImagePath = imgL;
    cfg.rightImagePath = imgR;
    cfg.outputDisparityPath = outPath;

    fprintf(stdout, "密集匹配: %s <-> %s\n", imgL.c_str(), imgR.c_str());

    if (verbose)
    {
        fprintf(stdout,
                "  算法=%s 代价=%s 子像素=%s 视差=[%d,%d] 核=%dx%d\n",
                algoStr.c_str(),
                costStr.c_str(),
                subpixelStr.c_str(),
                minDisp,
                maxDisp,
                kernelW,
                kernelH);
        fprintf(stdout,
                "  设备=%s CUDA设备=%d OpenCL设备=%d CPU线程=%d\n",
                xjw::dense_match::denseMatchComputeBackendName(computeBackend),
                gpu,
                openClDevice,
                threads);
    }

    xjw::dense_match::DenseMatchService service(cfg);
    xjw::dense_match::DisparityResult result;
    try
    {
        result = service.process();
    }
    catch (const std::bad_alloc& error)
    {
        cli::fatal("密集匹配主机内存分配失败: 视差=[" + std::to_string(minDisp) + ',' + std::to_string(maxDisp) +
                       "): " + error.what(),
                   cli::EXIT_ALGO_ERR);
    }
    catch (const cv::Exception& error)
    {
        cli::fatal("密集匹配 GPU/OpenCV 执行失败: 视差=[" + std::to_string(minDisp) + ',' + std::to_string(maxDisp) +
                       "): " + error.what(),
                   cli::EXIT_ALGO_ERR);
    }
    catch (const std::exception& error)
    {
        cli::fatal("密集匹配执行失败: 视差=[" + std::to_string(minDisp) + ',' + std::to_string(maxDisp) +
                       "): " + error.what(),
                   cli::EXIT_ALGO_ERR);
    }

    if (result.disparity.empty())
    {
        const std::string detail = service.lastError().empty() ? "视差图为空" : service.lastError();
        cli::fatal("密集匹配失败: " + detail, cli::EXIT_ALGO_ERR);
    }

    if (!xjw::dense_match::DenseMatchService::saveDisparity(result, outPath))
    {
        cli::fatal("无法保存视差图: " + outPath, cli::EXIT_IO_ERR);
    }

    fprintf(stdout, "视差图已保存: %s\n", outPath.c_str());
    return cli::EXIT_OK;
}
