#include "ReconstructionCliOptions.h"

// 工作流程 CLI 参数只负责命令行到核心配置的适配。

#include "PointCloudWorkflowConfig.h"

#include <QString>

#include <algorithm>
#include <thread>

namespace xjw::cli
{

    void ReconstructionCliOptions::addTo(CLI::App& app)
    {
        app.add_option("list_file", listPathArg, "影像/相机 .lis 清单")->required();
        app.add_option("-o,--output-dir", outputDirArg, "输出目录");
        app.add_option("--chunk-id", chunkIdArg, "使用指定 UUID 的 Chunk");
        app.add_option("--chunk-name", chunkNameArg, "使用指定名称的 Chunk");
        app.add_option("--device", device, "SfM 计算设备: auto, cpu, cuda, opencl, metal")
            ->check(CLI::IsMember({"auto", "cpu", "cuda", "opencl", "metal"}));
        app.add_option("--sfm-matching-algorithm",
                       sfmMatchingAlgorithmId,
                       "SfM 影像匹配算法: auto_sift, plamatch_hct, sift_lightglue, loma_r")
            ->check(CLI::IsMember({"auto_sift", "plamatch_hct", "sift_lightglue", "loma_r"}));
        app.add_option(
            "--sfm-lightglue-engine", sfmLightGlueEnginePath, "TensorRT LightGlue .engine；留空时从模型目录查找");
        app.add_option(
            "--sfm-loma-r-package", sfmLoMaRTensorRtPackagePath, "LoMa-R TensorRT 包清单；留空时从模型目录查找");
        app.add_flag("--sfm-guided-rematching", sfmGuidedRematching, "初始 SfM 后启用引导重匹配");
        app.add_flag("--lock-input-camera-poses", lockInputCameraPoses, "已知位姿 SfM/BA 中保持输入相机外参固定");
        app.add_option("--quality", quality, "SfM 质量等级: 0..3");
        app.add_option("--threads", threads, "CPU 线程数；0 表示按当前硬件自动选择");
        app.add_option("--cuda-parallel-pairs",
                       cudaParallelPairs,
                       "GPU 匹配并行像对数；0 按 SIFT/LightGlue/LoMa-R 独立显存模型自动选择");
        app.add_option("--feature-max-image-dim",
                       featureMaxImageDim,
                       "深度特征输入最长边；0 使用自适应质量预设，负值从不限制开始");
        app.add_option("--mvs-quality", mvsQuality, "MVS 质量: highest, high, medium, low, lowest")
            ->check(CLI::IsMember({"highest", "high", "medium", "low", "lowest"}));
        app.add_option("--mvs-backend", mvsBackend, "MVS 后端: auto, cpu, cuda, opencl")
            ->check(CLI::IsMember({"auto", "cpu", "cuda", "opencl"}));
        app.add_option("--point-cloud-backend",
                       pointCloudBackend,
                       "PlaPoint 后端: auto, cpu, cuda, opencl；"
                       "auto 对小规模滤波/法向估计使用 CPU，较大任务优先 CUDA、OpenCL；"
                       "HeightGrid/Poisson 使用各自策略")
            ->check(CLI::IsMember({"auto", "cpu", "cuda", "opencl"}));
        app.add_option("--mvs-scene-profile",
                       mvsSceneProfile,
                       "MVS 场景类型: auto, general/custom, orbital_object, aerial_terrain")
            ->check(CLI::IsMember({"auto", "general", "custom", "orbital_object", "aerial_terrain"}));
        app.add_option("--mvs-depth-filter", mvsDepthFilter, "MVS 深度过滤: auto, mild, moderate, aggressive")
            ->check(CLI::IsMember({"auto", "mild", "moderate", "aggressive"}));
        app.add_option("--mvs-mask-dir", mvsMaskDirArg, "包含 <影像名>_mask.png 工程排除蒙版的目录");
        app.add_flag("--mvs-save-levels", mvsSaveLevels, "保存 Level 2/3 原始深度结果用于诊断");
        app.add_flag("--mvs-native-depth-grid",
                     mvsNativeDepthGrid,
                     "实验选项：为未校正的 general/custom 场景保留最终 PatchMatch 网格");
        app.add_flag("--mvs-two-source-growth", mvsTwoSourceGrowth, "从三源核心启用保守的双源深度扩展");
        app.add_option("--mvs-two-source-growth-distance", mvsTwoSourceGrowthDistance, "最大测地扩展距离（像素）")
            ->check(CLI::Range(1, 8));
        app.add_option("--mvs-two-source-growth-spread", mvsTwoSourceGrowthSpread, "最大相对逆深度离散度")
            ->check(CLI::Range(0.001, 0.05));
        app.add_option(
               "--mvs-two-source-growth-normal-angle", mvsTwoSourceGrowthNormalAngle, "最大局部表面法线夹角（度）")
            ->check(CLI::Range(5.0, 45.0));
        app.add_option(
               "--mvs-two-source-growth-maximum-area", mvsTwoSourceGrowthMaximumArea, "弱支持连通分量的最大面积")
            ->check(CLI::Range(1, 512));
        mvsResScaleOption =
            app.add_option("--mvs-res-scale", mvsResScale, "MVS 深度分辨率比例；显式覆盖 --mvs-quality 的分辨率");
        app.add_option("--mvs-iterations", mvsIterations, "MVS PatchMatch 迭代次数");
        app.add_option("--mvs-confidence", mvsConfidence, "MVS PatchMatch 置信度阈值");
        app.add_option("--mvs-fusion-confidence", mvsFusionConfidence, "MVS 融合置信度阈值");
        app.add_option("--mvs-gpu-frame-workers", mvsGpuFrameWorkers, "MVS CUDA/OpenCL 帧工作数；0 表示自动选择");
        app.add_option("--mvs-cpu-frame-workers", mvsCpuFrameWorkers, "MVS CPU 帧工作数；0 表示自动选择");
        app.add_option("--mvs-max-frames", mvsMaxFrames, "回归/调试时仅处理前 N 个已注册帧；0 表示全部");
        app.add_option(
            "--mvs-fusion-max-image-dim", mvsFusionMaxImageDim, "深度图融合使用的影像最长边；0 保持完整分辨率");
#ifndef PLASCAN_THREE_D_ONLY
        app.add_option("--dem-resolution", demResolution, "DEM/DOM 分辨率；0 由地形工作流选择");
#endif
        app.add_option("--mesh-resolution", meshResolution, "网格重建分辨率");
        app.add_flag("--stop-after-sfm", stopAfterSfm, "仅运行 SfM 并写出报告，随后在 MVS 前停止");
        app.add_flag("--skip-mvs", skipMvs, "SfM 后跳过 MVS 及后续网格/地形阶段");
        app.add_flag("--mvs-depth-only", mvsDepthOnly, "仅运行 MVS 深度图估计，随后跳过融合、网格和地形阶段");
        app.add_flag("--skip-mesh", skipMesh, "MVS 稠密点云生成后跳过网格重建");
#ifndef PLASCAN_THREE_D_ONLY
        app.add_flag("--skip-model", skipModel, "跳过网格重建");
        app.add_flag("--skip-terrain", skipTerrain, "跳过 DEM/DOM 生成");
#endif
        app.add_flag("--export-obj", exportObj, "在支持时同时导出 OBJ/MTL/纹理");
        app.add_flag("--skip-texture", skipTexture, "跳过 OBJ/MTL 纹理导出，仅保留 PLY 网格");
        app.add_flag("--force", forceOutput, "允许复用或覆盖非空输出目录");
    }

    void ReconstructionCliOptions::normalize()
    {
        const int hardware_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        threads = threads > 0 ? std::clamp(threads, 1, hardware_threads) : std::max(1, hardware_threads - 2);
        if (skipTexture)
        {
            exportObj = false;
        }
        if (skipMesh)
        {
            skipModel = true;
        }
        if (mvsResScaleOption && mvsResScaleOption->count() == 0)
        {
            const auto profile = xjw::core::project::depthQualityProfileFromId(QString::fromStdString(mvsQuality));
            mvsResScale = 1.0 / static_cast<double>(xjw::core::project::depthQualityDownsample(profile));
        }
        mvsResScale = std::clamp(mvsResScale, 0.05, 1.0);
        mvsIterations = std::max(1, mvsIterations);
        mvsConfidence = std::clamp(mvsConfidence, 0.0, 1.0);
        mvsFusionConfidence = std::clamp(mvsFusionConfidence, 0.0, 1.0);
        mvsGpuFrameWorkers = std::max(0, mvsGpuFrameWorkers);
        mvsCpuFrameWorkers = std::max(0, mvsCpuFrameWorkers);
        mvsMaxFrames = std::max(0, mvsMaxFrames);
        mvsFusionMaxImageDim = std::max(0, mvsFusionMaxImageDim);
    }

} // namespace xjw::cli
