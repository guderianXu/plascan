#pragma once

#include "BundleAdjustOptions.h"
#include "BundleAdjustProblem.h"
#include "BundleAdjustResult.h"

#include <array>
#include <vector>

namespace xjw::detail::plamatrix_ba
{

struct ActiveProblem
{
    std::vector<char> activeTrack;
    std::vector<char> activeLaserRange;
    std::vector<int> cameraBlock;
    std::vector<int> intrinsicBlockByCamera;
    std::vector<int> calibrationGroupByCamera;
    std::vector<int> trackPrimaryBlock;
    std::vector<int> trackBlock;
    std::vector<int> laserBlock;
    int primaryBlockCount = 0;
    int cameraBlockCount = 0;
    int intrinsicBlockCount = 0;
    int promotedTrackBlockCount = 0;
    int trackBlockCount = 0;
    int laserBlockCount = 0;
    int activeLaserRangeCount = 0;
    int activeTrackCount = 0;
    int observationCount = 0;
    int rejectedInitialTracks = 0;
};

inline constexpr int kPrimaryBlockSize = 9;
inline constexpr int kEliminatedBlockSize = 3;

struct IntrinsicGroupState
{
    std::array<double, 9> parameters{};
    std::array<double, 9> prior{};
    std::array<double, 9> inverseSigma{};
    std::array<double, 9> lower{};
    std::array<double, 9> upper{};
    BAIntrinsicParameterMask enabled{};
    double focalReference = 1.0;
    double aspectReference = 1.0;
};

bool hasSharedIntrinsics(const BAOptions& options);

std::vector<IntrinsicGroupState> initializeIntrinsicGroups(
    const std::vector<FramePinholeCamera>& cameras,
    const BAOptions& options,
    const ActiveProblem& active);

BAIntrinsicParameterMask activeIntrinsicParameters(
    const BAOptions& options,
    const BAIntrinsicParameterMask& enabled,
    int iteration);

int intrinsicStageCount(const BAOptions& options,
                        const BAIntrinsicParameterMask& enabled);

void applyIntrinsicStep(const ActiveProblem& active,
                        const std::vector<double>& primary_step,
                        std::vector<IntrinsicGroupState>* groups);

void publishIntrinsics(const std::vector<FramePinholeCamera>& input_cameras,
                       const BAOptions& options,
                       const ActiveProblem& active,
                       const std::vector<IntrinsicGroupState>& groups,
                       BAResult* result);

ActiveProblem prepareActiveProblem(const std::vector<FramePinholeCamera>& cameras,
                                   const std::vector<BATrack>& tracks,
                                   const BAOptions& options);

} // namespace xjw::detail::plamatrix_ba
