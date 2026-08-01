#pragma once

#include "OrthoProjector.h"

#include <opencv2/core.hpp>

#include <vector>

namespace xjw::ortho_internal
{

struct LoadedFrame
{
    OrthoImageInput input;
    cv::Mat imageBgr;
    cv::Mat exclusionMask;
    double gain = 1.0;
    double sharpnessWeight = 1.0;
    bool contributed = false;
};

struct ColorCandidate
{
    int frameIndex = -1;
    cv::Vec3f color{0.0f, 0.0f, 0.0f};
    double weight = 0.0;
};

bool loadFrames(const std::vector<OrthoImageInput> &inputs,
                const OrthoGenerationOptions &options,
                std::vector<LoadedFrame> *frames,
                const std::atomic_bool *cancelFlag,
                QString *errorMsg);

bool sampleFrame(const LoadedFrame &frame,
                 const double world[3],
                 ColorCandidate *candidate);

void filterGhostCandidates(std::vector<ColorCandidate> *candidates);

qint64 fillSmallInteriorHoles(cv::Mat *imageBgr,
                              const cv::Mat &surfaceMask,
                              const cv::Mat &coverageMask,
                              const OrthoGenerationOptions &options,
                              cv::Mat *filledMask,
                              const std::atomic_bool *cancelFlag);

} // namespace xjw::ortho_internal
