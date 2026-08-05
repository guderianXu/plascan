#include "ReconstructionCliOptions.h"

// 工作流程 CLI 参数只负责命令行到核心配置的适配。

#include "PointCloudWorkflowConfig.h"

#include <QString>

#include <algorithm>
#include <thread>

namespace xjw::cli
{

void ReconstructionCliOptions::addTo(CLI::App &app)
{
    app.add_option("list_file", listPathArg, "image/camera .lis file")->required();
    app.add_option("-o,--output-dir", outputDirArg, "output directory");
    app.add_option("--chunk-id", chunkIdArg, "use the Chunk with this UUID");
    app.add_option("--chunk-name", chunkNameArg, "use the Chunk with this name");
    app.add_option("--device", device, "auto, cpu, cuda")->check(CLI::IsMember({"auto", "cpu", "cuda"}));
    app.add_option("--sfm-matching-algorithm", sfmMatchingAlgorithmId,
                   "SFM image matching algorithm id: sift_lightglue or loma_r")
        ->check(CLI::IsMember({"sift_lightglue", "loma_r"}));
    app.add_option("--sfm-lightglue-engine", sfmLightGlueEnginePath,
                   "TensorRT LightGlue .engine; empty enables model-directory lookup");
    app.add_option("--sfm-loma-r-package", sfmLoMaRTensorRtPackagePath,
                   "LoMa-R TensorRT package manifest; empty enables model-directory lookup");
    app.add_flag("--sfm-guided-rematching", sfmGuidedRematching,
                 "enable guided rematching after initial SfM");
    app.add_flag("--lock-input-camera-poses", lockInputCameraPoses,
                 "keep input camera extrinsics fixed during known-pose SfM/BA");
    app.add_option("--quality", quality, "SFM quality level 0..3");
    app.add_option("--threads", threads,
                   "CPU thread count; 0 selects current hardware automatically");
    app.add_option("--cuda-parallel-pairs", cudaParallelPairs, "LightGlue CUDA parallel pair count");
    app.add_option("--feature-max-image-dim", featureMaxImageDim,
                   "deep feature max image side; 0 uses auto/adaptive quality preset, negative starts unbounded");
    app.add_option("--mvs-quality", mvsQuality, "MVS quality: highest, high, medium, low, lowest")
        ->check(CLI::IsMember({"highest", "high", "medium", "low", "lowest"}));
    app.add_option("--mvs-scene-profile", mvsSceneProfile,
                   "MVS scene profile: auto, orbital_object, aerial_terrain")
        ->check(CLI::IsMember({"auto", "orbital_object", "aerial_terrain"}));
    app.add_option("--mvs-depth-filter", mvsDepthFilter,
                   "MVS depth filter: auto, mild, moderate, aggressive")
        ->check(CLI::IsMember({"auto", "mild", "moderate", "aggressive"}));
    app.add_option("--mvs-mask-dir", mvsMaskDirArg,
                   "directory containing <image stem>_mask.png project exclusion masks");
    app.add_flag("--mvs-save-levels", mvsSaveLevels,
                 "save Level 2/3 raw depth results for diagnostics");
    app.add_flag("--mvs-two-source-growth", mvsTwoSourceGrowth,
                 "enable conservative two-source depth growth from three-source cores");
    app.add_option("--mvs-two-source-growth-distance", mvsTwoSourceGrowthDistance,
                   "maximum geodesic growth distance in pixels")->check(CLI::Range(1, 8));
    app.add_option("--mvs-two-source-growth-spread", mvsTwoSourceGrowthSpread,
                   "maximum relative inverse-depth spread")->check(CLI::Range(0.001, 0.05));
    app.add_option("--mvs-two-source-growth-normal-angle", mvsTwoSourceGrowthNormalAngle,
                   "maximum local surface-normal angle in degrees")->check(CLI::Range(5.0, 45.0));
    app.add_option("--mvs-two-source-growth-maximum-area", mvsTwoSourceGrowthMaximumArea,
                   "maximum connected weak-support component area")->check(CLI::Range(1, 512));
    mvsResScaleOption = app.add_option(
        "--mvs-res-scale", mvsResScale,
        "MVS depth resolution scale; explicitly overrides --mvs-quality resolution");
    app.add_option("--mvs-iterations", mvsIterations, "MVS PatchMatch iterations");
    app.add_option("--mvs-confidence", mvsConfidence, "MVS PatchMatch confidence threshold");
    app.add_option("--mvs-fusion-confidence", mvsFusionConfidence, "MVS fusion confidence threshold");
    app.add_option("--mvs-gpu-frame-workers", mvsGpuFrameWorkers,
                   "MVS CUDA frame workers; 0 chooses automatically");
    app.add_option("--mvs-cpu-frame-workers", mvsCpuFrameWorkers,
                   "MVS CPU frame workers; 0 chooses automatically");
    app.add_option("--mvs-max-frames", mvsMaxFrames,
                   "limit MVS to first N registered frames for regression/debug runs; 0 uses all");
    app.add_option("--mvs-fusion-max-image-dim", mvsFusionMaxImageDim,
                   "max image side used during depth-map fusion; 0 keeps full resolution");
#ifndef PLASCAN_THREE_D_ONLY
    app.add_option("--dem-resolution", demResolution, "DEM/DOM resolution; 0 lets TerrainPipeline choose");
#endif
    app.add_option("--mesh-resolution", meshResolution, "mesh reconstruction grid resolution");
    app.add_flag("--stop-after-sfm", stopAfterSfm, "run SFM only, write report, then stop before MVS");
    app.add_flag("--skip-mvs", skipMvs, "skip MVS and downstream mesh/terrain stages after SFM");
    app.add_flag("--mvs-depth-only", mvsDepthOnly,
                 "run MVS depth-map estimation only, then skip fusion, mesh, and terrain");
    app.add_flag("--skip-mesh", skipMesh, "skip mesh reconstruction after MVS dense cloud generation");
#ifndef PLASCAN_THREE_D_ONLY
    app.add_flag("--skip-model", skipModel, "skip mesh reconstruction");
    app.add_flag("--skip-terrain", skipTerrain, "skip DEM/DOM generation");
#endif
    app.add_flag("--export-obj", exportObj, "also export OBJ/MTL/texture where supported");
    app.add_flag("--skip-texture", skipTexture, "skip OBJ/MTL texture export and keep only PLY mesh");
    app.add_flag("--force", forceOutput, "allow reusing or overwriting a non-empty output directory");
}

void ReconstructionCliOptions::normalize()
{
    const int hardware_threads =
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    threads = threads > 0
        ? std::clamp(threads, 1, hardware_threads)
        : hardware_threads;
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
