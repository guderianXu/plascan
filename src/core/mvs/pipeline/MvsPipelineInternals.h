#pragma once
// Private implementation contracts; not an exported include root.
#include "../MvsPipelineService.h"

#include "concurrency/SafeWorkerGroup.h"
#include "DepthComputeScheduler.h"
#include "DenseCloudBuilder.h"
#include "DepthConsistencyCache.h"
#include "DepthConsistencyEvidencePolicy.h"
#include "DepthCrossViewHoleRepair.h"
#include "DepthGeometryConsistency.h"
#include "DepthLayerReliability.h"
#include "DepthFrameUtils.h"
#include "GpuDeviceLease.h"
#include "DepthMemoryPolicy.h"
#include "DepthPyramidPolicy.h"
#include "DepthProvenance.h"
#include "EpipolarRectifier.h"
#include "CameraBaseline.h"
#include "MvsImagePreprocessor.h"
#include "MvsImageMetadataProbe.h"
#include "MvsQualityReport.h"
#include "MvsSourcePlanner.h"
#include "MvsVisibilityGraphBuilder.h"
#include "MvsViewSelection.h"
#include "RecoveredDepthScene.h"
#include "PatchMatchPhotometricCost.h"
#include "Logger.h"
#include "io/PathIO.h"
#include "string_utils/StringTransform.h"
#include <QDir>
#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QStringList>
#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <functional>
#include <exception>
#include <memory>
#include <limits>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw::mvs::pipeline_detail
{
    using common::string_utils::asciiLowerCopy;

    constexpr int kFullRasterBoundaryEdgeRadiusPixels = 1;

    constexpr int kFullRasterLocalSameLayerRadiusPixels = 1;

    constexpr int kFullRasterSparseResidualRadiusPixels = 1;

    constexpr int kFullRasterConsistencySearchRadiusPixels = 1;

    constexpr float kFullRasterConsistencyRoundTripPixels = 3.0f;

    constexpr int kFullRasterMinimumSmallHoleArea = 64;

    constexpr float kSmallHoleAreaFraction = 0.002f;

    struct PreRepairDepthLayerReliability
    {
        DepthLayerReliabilityResult result;
        QJsonObject diagnostics;
    };

    constexpr float kSkipContentMaskCoverage = 0.985f;

    constexpr std::size_t kMaxInlineDenseFilterPoints = 500000;

    constexpr std::size_t kMaxProjectedDepthQuantileSamples = 8192;

    constexpr uint64_t kBytesPerGiB = 1024ull * 1024ull * 1024ull;

    using Clock = std::chrono::steady_clock;

    struct FrameTiming
    {
        double sourceMs = 0.0;
        double rangeMs = 0.0;
        double hintMs = 0.0;
        double rectifyMs = 0.0;
        double patchmatchMs = 0.0;
        double filterMs = 0.0;
        double totalMs = 0.0;
    };

    struct SystemMemorySnapshot
    {
        uint64_t totalPhysicalBytes = 0;
        uint64_t availablePhysicalBytes = 0;
        bool valid = false;
    };

    std::string mvsSourcePairKey(const std::string& imageA, const std::string& imageB);
    std::string mvsSourcePairKey(const QString& keyA, const QString& keyB);

    struct MvsSourcePairQualityLookup
    {
        std::unordered_map<std::string, MvsSourcePairQuality> qualitiesByPairKey;

        const MvsSourcePairQuality* findNormalized(const QString& imageAKey, const QString& imageBKey) const
        {
            const std::string key = mvsSourcePairKey(imageAKey, imageBKey);
            if (key.empty())
            {
                return nullptr;
            }

            const auto it = qualitiesByPairKey.find(key);
            return it == qualitiesByPairKey.end() ? nullptr : &it->second;
        }
    };

    struct SourceQualitySummary
    {
        int sourceViewCount = 0;
        int verifiedSourceViewCount = 0;
        int backfillSourceViewCount = 0;
        int sequenceFallbackSourceViewCount = 0;
        double meanQuality = 0.0;
        double minQuality = 0.0;
    };

    struct DepthConfidenceSummary
    {
        int validPixelCount = 0;
        double meanConfidence = 0.0;
    };

    struct DepthConsistencySourceInput
    {
        cv::Mat depth;
        FramePinholeCamera camera;
        cv::Mat confidence;
        float reliabilityWeight = 1.0f;
        int sourceOrdinal = -1;
        int searchRadiusPixels = kFullRasterConsistencySearchRadiusPixels;
        bool evaluateSubpixelFootprint = false;
    };

    struct DepthConsistencyVoteTotals
    {
        std::uint64_t consistent = 0;
        std::uint64_t occluded = 0;
        std::uint64_t contradicted = 0;
        std::uint64_t unverifiable = 0;
    };
    DepthPixelDomainScale pixelDomainScaleForResult(const DepthFrameResult& result, const cv::Size& grid_size);

    int effectiveMinimumSmallHoleArea(const DepthFrameResult& result, const cv::Size& grid_size);

    int effectiveSparseResidualRadius(const DepthFrameResult& result, const cv::Size& grid_size);

    PreRepairDepthLayerReliability
    analyzePreRepairDepthLayerReliability(const DepthFrameResult& frame,
                                          const cv::Mat& depth,
                                          const cv::Mat* reference_gray,
                                          const cv::Mat& consistent_votes,
                                          const cv::Mat& occluded_votes,
                                          const cv::Mat& contradicted_votes,
                                          const cv::Mat& geometry_source_mask,
                                          const cv::Mat& source_inverse_depth_sum,
                                          const cv::Mat& source_inverse_depth_squared_sum,
                                          const AdaptiveGeometryEvidenceAccumulatorMaps* adaptive_accumulator);

    QJsonObject configuredEffectiveInteger(int configured, int effective);

    QJsonObject configuredEffectiveFloat(float configured, float effective);

    QJsonObject makePixelDomainDiagnostics(const cv::Size& raster_size,
                                           const cv::Size& grid_size,
                                           const FusionConfig& fusion_config,
                                           bool requested_native_final_grid,
                                           bool effective_native_final_grid);

    void calibrateFinalDepthConfidenceMap(cv::Mat& confidence_map, const cv::Mat* depth_provenance = nullptr);

    QJsonArray doubleArrayToJson(const double* values, int count);

    FramePinholeCamera mvsPinholeCamera(const FramePinholeCamera& camera);

    cv::Mat restoreNativePyramidArtifact(const cv::Mat& artifact, const cv::Size& working_size);

    QJsonObject cameraModelToJson(const FramePinholeCamera& camera);

    QJsonObject depthPoseRefinementCandidateToJson(const DepthPoseRefinementCandidate& candidate,
                                                   const DepthPoseRefinementStageResult& stage);

    QJsonArray depthPyramidLevelsToJson(const std::vector<DepthLevelSummary>& summaries);

    QString sceneProfileId(MvsSceneProfile profile);

    QString depthFilterModeId(DepthFilterMode mode);

    double elapsedMs(Clock::time_point start, Clock::time_point end);

    QString normalizedMvsPathKey(const std::string& path);

    const std::string& mvsRasterPath(const CameraView& view);

    std::string mvsSourcePairKey(const QString& keyA, const QString& keyB);

    std::string mvsSourcePairKey(const std::string& imageA, const std::string& imageB);

    MvsSourcePairQualityLookup buildMvsSourcePairQualityLookup(const std::vector<MvsSourcePairQuality>& qualities);

    QJsonObject depthPostProcessStatsToJson(const DepthPostProcessStats& stats);

    DepthPostProcessEvidence depthPostProcessEvidence(const DepthFrameResult& frame);

    DepthPostProcessEvidence updateDepthEvidenceConfidence(DepthFrameResult& frame,
                                                           const cv::Mat& depth,
                                                           cv::Mat& confidence,
                                                           DepthPostProcessEvidence evidence,
                                                           bool enabled);

    double bytesToGiB(uint64_t bytes);

    SystemMemorySnapshot querySystemMemorySnapshot();

    uint64_t depthFramePixelStorageBytes(int width, int height);

    SourceQualitySummary summarizeSourceQuality(const QJsonArray& sourcePlan, int fallbackSourceViewCount);

    DepthConfidenceSummary summarizeDepthConfidence(const cv::Mat& depthMap, const cv::Mat* confidenceMap);

    bool usesAdaptiveGeometryEvidence(const DepthGenConfig& config, MvsSceneProfile sceneProfile);

    std::vector<DepthMemoryFrameSize> depthMemoryFrameSizes(const std::vector<CameraView>& views);

    std::vector<MvsImageMemoryFrame> imageMemoryFrames(const std::vector<CameraView>& views);

    int maximumConsistencySourceViews(const DepthGenConfig& config);

    DepthMemoryPolicyDecision evaluateDepthMemoryPolicy(const std::vector<CameraView>& views,
                                                        const DepthGenConfig& config,
                                                        MvsSceneProfile sceneProfile,
                                                        const SystemMemorySnapshot& snapshot);

    uint64_t largestDepthFrameBytes(const std::vector<CameraView>& views);

    uint64_t saturatingMultiplyBytes(uint64_t value, uint64_t factor);

    uint64_t retainedDepthMemoryBudgetBytes(const SystemMemorySnapshot& snapshot,
                                            const DepthGenConfig& config,
                                            uint64_t largestFrameBytes,
                                            uint64_t transientFrameBytes,
                                            size_t concurrentFrameWorkers);

    QString depthMemoryPolicyReason(const DepthMemoryPolicyDecision& decision, const SystemMemorySnapshot& snapshot);

    bool memoryPressureRequiresStreaming(const DepthGenConfig& config,
                                         const SystemMemorySnapshot& snapshot,
                                         const DepthMemoryPolicyDecision& decision);

    void releaseStoredDepthFramePixelStorage(std::vector<DepthFrameResult>& frames);

    void releaseStoredDepthFrameStreamingPixelStorage(std::vector<DepthFrameResult>& frames);

    size_t adaptiveSaveQueueCapacity(const SystemMemorySnapshot& snapshot,
                                     const DepthGenConfig& config,
                                     uint64_t largestFrameBytes,
                                     uint64_t transientFrameBytes,
                                     size_t concurrentFrameWorkers);

    uint64_t estimatedSaveQueueProducerBytes(uint64_t largestFrameBytes);

    uint64_t adaptiveSaveQueueResidentByteCapacity(const SystemMemorySnapshot& snapshot,
                                                   const DepthGenConfig& config,
                                                   uint64_t largestFrameBytes,
                                                   size_t maxResidentTasks,
                                                   uint64_t transientFrameBytes,
                                                   size_t concurrentFrameWorkers);

    int preloadImagesWorkerCount(int viewCount, int requestedThreads);

    bool writeFastDepthMatStorage(const std::string& path, const cv::Mat& matrix, std::string* errorMsg);

    bool saveDepthPreviewPng(const std::string& path, const cv::Mat& depthMap, std::string* errorMsg);

    QString manifestPathForOutput(const DepthGenConfig& config, const std::string& outputDir);

    int resolvedTotalCpuThreadBudget(const DepthGenConfig& config);

    cv::Size patchMatchWorkSize(const cv::Mat& image, const PatchMatchConfig& config);

    CrossViewHoleRepairOptions orbitalCrossViewHoleRepairOptions(const DepthGenConfig& config);

    std::vector<int>
    consistencySourceIndicesForFrame(const std::vector<DepthFrameResult>& frames, int refIdx, int viewCount);

    std::vector<int> orbitalHoleRepairSourceIndices(const std::vector<DepthFrameResult>& frames,
                                                    const std::vector<int>& consistencySources,
                                                    int refIdx,
                                                    int viewCount,
                                                    int requestedSourceCount);

    bool isCudaMemoryFailure(const std::string& message);

    PatchMatchConfig patchMatchConfigForRecordedWorker(PatchMatchConfig config, std::string_view workerId);

    bool estimatePatchMatchWithAdaptiveCuda(const char* stageLabel,
                                            int refIdx,
                                            const cv::Mat& refGray,
                                            const std::vector<cv::Mat>& srcGrays,
                                            const FramePinholeCamera& refCam,
                                            const std::vector<FramePinholeCamera>& srcCams,
                                            float zNear,
                                            float zFar,
                                            const PatchMatchConfig& config,
                                            cv::Mat& depthOut,
                                            cv::Mat* confOut,
                                            std::string* errorMsg,
                                            const cv::Mat* hintDepth,
                                            const cv::Mat* hintRadius,
                                            const cv::Mat* referenceValidMask,
                                            const std::vector<cv::Mat>* sourceValidMasks,
                                            const PatchMatchAuxiliaryInput* auxiliaryInput = nullptr,
                                            PatchMatchAuxiliaryOutput* auxiliaryOutput = nullptr);

    float sourceGeometryReliabilityWeight(const DepthFrameResult& reference_frame, int source_view_index);

    int cameraBaselineSector(const FramePinholeCamera& reference_camera, const FramePinholeCamera& source_camera);

    void accumulateDepthConsistency(const cv::Mat& referenceDepth,
                                    const FramePinholeCamera& referenceCamera,
                                    const std::vector<DepthConsistencySourceInput>& sources,
                                    float relativeThreshold,
                                    float maximumRoundTripErrorPixels,
                                    int rowWorkers,
                                    const std::atomic<bool>& cancelled,
                                    cv::Mat& consistentVotes,
                                    cv::Mat& occludedVotes,
                                    cv::Mat& contradictedVotes,
                                    cv::Mat& unverifiableVotes,
                                    cv::Mat& geometrySourceMask,
                                    cv::Mat& sourceInverseDepthSum,
                                    cv::Mat& sourceInverseDepthSquaredSum,
                                    AdaptiveGeometryEvidenceAccumulatorMaps* adaptiveEvidence);

    cv::Mat makeDepthConsistencyMask(const cv::Mat& referenceDepth,
                                     int sourceViewCount,
                                     int minimumSourceConfirmations,
                                     const cv::Mat& consistentVotes,
                                     const cv::Mat& occludedVotes,
                                     const cv::Mat& contradictedVotes,
                                     int rowWorkerCount,
                                     const std::atomic<bool>* cancelled);

    DepthConsistencyVoteTotals summarizeDepthConsistencyVotes(const cv::Mat& consistentVotes,
                                                              const cv::Mat& occludedVotes,
                                                              const cv::Mat& contradictedVotes,
                                                              const cv::Mat& unverifiableVotes,
                                                              int rowWorkerCount);

    void updateDepthCompletenessAfterPostprocess(DepthFrameResult& result,
                                                 const cv::Mat& depth,
                                                 const DepthPostProcessStats& stats);

    DepthAnchoredHoleInterpolationStats
    repairPostprocessedInternalDepthHoles(DepthFrameResult& result,
                                          cv::Mat& depth,
                                          cv::Mat& confidence,
                                          MvsSceneProfile sceneProfile,
                                          cv::Mat* anchoredInterpolationMask = nullptr);

    void updateDepthFrameQualityAfterConsistency(DepthFrameResult& result,
                                                 const cv::Mat& depth,
                                                 const cv::Mat& confidence,
                                                 MvsSceneProfile scene_profile,
                                                 DepthFilterMode filter_mode,
                                                 bool consistency_stage_expected,
                                                 const AdaptiveGeometryEvidenceSummary& adaptive_summary,
                                                 const DiscreteGeometryCoreSummary& discrete_summary);

    double det3(const double* R);

    template <typename Fn> void parallelForRows(int rowCount, int workerCount, Fn&& fn)
    {
        if (rowCount <= 0)
        {
            return;
        }
        const int workers = std::clamp(std::max(1, workerCount), 1, rowCount);
        if (workers == 1)
        {
            for (int row = 0; row < rowCount; ++row)
            {
                fn(row);
            }
            return;
        }

        std::atomic<int> nextRow{0};
        xjw::common::concurrency::runWorkerGroup(static_cast<std::size_t>(workers),
                                                 [&](std::stop_token stopToken)
                                                 {
                                                     while (!stopToken.stop_requested())
                                                     {
                                                         const int row = nextRow.fetch_add(1);
                                                         if (row >= rowCount)
                                                         {
                                                             break;
                                                         }
                                                         fn(row);
                                                     }
                                                 });
    }
} // namespace xjw::mvs::pipeline_detail
