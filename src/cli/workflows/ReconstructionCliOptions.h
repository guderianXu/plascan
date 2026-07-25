#pragma once

// GUI 工作流程的无界面参数模型。

#include "cli_common.h"

#include <string>

namespace xjw::cli
{

struct ReconstructionCliOptions
{
    std::string listPathArg;
#ifdef PLASCAN_THREE_D_ONLY
    std::string outputDirArg = "three_d_reconstruction_output";
#else
    std::string outputDirArg = "full_pipeline_output";
#endif
    std::string device = "auto";
    std::string sfmFeatureAlgorithm = "disk";
    std::string sfmMatchAlgorithm = "lightglue";
    bool sfmGuidedRematching = false;
    bool lockInputCameraPoses = false;
    int quality = 3;
    int threads = 8;
    int cudaParallelPairs = 1;
    int featureMaxImageDim = 0;
    std::string mvsQuality = "high";
    std::string mvsSceneProfile = "auto";
    std::string mvsDepthFilter = "auto";
    std::string mvsMaskDirArg;
    bool mvsSaveLevels = false;
    bool mvsTwoSourceGrowth = false;
    int mvsTwoSourceGrowthDistance = 3;
    double mvsTwoSourceGrowthSpread = 0.01;
    double mvsTwoSourceGrowthNormalAngle = 15.0;
    int mvsTwoSourceGrowthMaximumArea = 64;
    double mvsResScale = 0.5;
    int mvsIterations = 6;
    double mvsConfidence = 0.20;
    double mvsFusionConfidence = 0.50;
    int mvsGpuFrameWorkers = 0;
    int mvsCpuFrameWorkers = 0;
    int mvsMaxFrames = 0;
    int mvsFusionMaxImageDim = 2048;
#ifndef PLASCAN_THREE_D_ONLY
    double demResolution = 0.0;
#endif
    int meshResolution = 224;
    bool skipModel = false;
    bool skipMesh = false;
    bool stopAfterSfm = false;
    bool skipMvs = false;
    bool mvsDepthOnly = false;
#ifdef PLASCAN_THREE_D_ONLY
    bool skipTerrain = true;
#else
    bool skipTerrain = false;
#endif
    bool exportObj = true;
    bool skipTexture = false;
    bool forceOutput = false;
    CLI::Option *mvsResScaleOption = nullptr;

    void addTo(CLI::App &app);
    void normalize();
};

} // namespace xjw::cli
