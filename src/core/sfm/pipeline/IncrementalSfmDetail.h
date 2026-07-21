#pragma once

#include "IncrementalSfm.h"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace xjw::incremental_sfm_detail
{

struct KnownPoseTriangulationPolicy
{
    TriangulatorOptions triangulatorOptions;
    double filterMinTriAngle = 2.0;
    bool adapted = false;
    int validCandidates = 0;
    int acceptedWithDefault = 0;
    int acceptedWithAdapted = 0;
    double chosenMinTriAngle = 2.0;
};

inline constexpr int kKnownPoseMinLongInputTracksForQualityGate = 10;
inline constexpr int kKnownPoseMinPointsForTrackRatioGate = 20;
inline constexpr double kKnownPoseMaxTwoViewTrackRatio = 0.95;

struct SimilarityTransform3d
{
    bool valid = false;
    double scale = 1.0;
    std::array<double, 9> rotation{{1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0}};
    std::array<double, 3> translation{{0.0, 0.0, 0.0}};
    int inlierCount = 0;
    double rmse = 0.0;
};

bool shouldEvaluateMultipleInitialPairModels(const IncrementalSfmOptions &options,
                                             int totalImages,
                                             std::size_t candidateCount);
double scoreInitialPairTrial(const IncrementalSfmResult &result, int totalImages);
int effectivePnpMinTrackLength(const IncrementalSfmOptions &options,
                               std::size_t registeredImageCount);
bool pointUsableForPnp(const SfmReconstruction &reconstruction,
                       Point3DId pointId,
                       int minTrackLength);
double distance3d(const std::array<double, 3> &a, const std::array<double, 3> &b);
double percentile(std::vector<double> values, double ratio);
std::array<double, 9> interpolateCameraRotation(const std::array<double, 9> &rotationA,
                                                const std::array<double, 9> &rotationB,
                                                double ratio);
KnownPoseTriangulationPolicy resolveKnownPoseTriangulationPolicy(
    const std::shared_ptr<SfmReconstruction> &reconstruction,
    const CorrespondenceGraph &correspondenceGraph,
    const std::vector<ImageId> &imageIds,
    const IncrementalSfmOptions &options);
bool knownPoseMatchPassesGeometry(const SfmReconstruction &reconstruction,
                                  ImageId imageId,
                                  ImageId otherImageId,
                                  const FeatureMatch &match,
                                  const TriangulatorOptions &options);
std::array<double, 3> transformPoint(const SimilarityTransform3d &transform,
                                     const std::array<double, 3> &point);
std::array<double, 9> multiplyRotation(const std::array<double, 9> &left,
                                       const std::array<double, 9> &right);
double pointDistance(const std::array<double, 3> &a, const std::array<double, 3> &b);
double centerExtent(const std::vector<std::array<double, 3>> &points);
SimilarityTransform3d estimateRobustCameraCenterSimilarity(
    const std::vector<std::array<double, 3>> &source,
    const std::vector<std::array<double, 3>> &target);

} // namespace xjw::incremental_sfm_detail
