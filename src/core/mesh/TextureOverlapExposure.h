#pragma once

#include "TextureMapper.h"

#include <opencv2/core/types.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace xjw::mesh::texture_v4
{

struct PipelineData;

struct ExposureObservation
{
    int viewIndex = -1;
    std::uint64_t pointId = 0;
    double linearLuminance = 0.0;
};

struct ExposureSolveOptions
{
    int minimumPairSamples = 24;
    int maximumPairSamples = 512;
    double maximumPairLogMad = 0.08;
    float minimumGain = 0.90f;
    float maximumGain = 1.10f;
};

struct ExposureSolveResult
{
    std::vector<float> gains;
    std::uint64_t observationCount = 0;
    std::uint64_t candidatePairCount = 0;
    std::uint64_t acceptedPairCount = 0;
    std::uint64_t rejectedInsufficientPairCount = 0;
    std::uint64_t rejectedHighMadPairCount = 0;
    int connectedComponentCount = 0;
    int correctedViewCount = 0;
    double maximumAcceptedLogMad = 0.0;
    bool graphConnected = false;
    bool applied = false;
    std::string status = "disabled";
};

ExposureSolveResult solveRobustOverlapExposure(
    int viewCount,
    std::vector<ExposureObservation> observations,
    const ExposureSolveOptions &options = {});

cv::Vec3f srgb8ToLinear(const cv::Vec3f &srgbColor);

cv::Vec3f applyLinearSrgbExposureGain(const cv::Vec3f &srgbColor,
                                      float gain);

bool estimateOverlapExposureGains(const TextureMappingConfig &config,
                                  PipelineData *data,
                                  TextureMappingResult *result,
                                  std::string *errorMsg);

} // namespace xjw::mesh::texture_v4
