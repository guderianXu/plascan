#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    void MvsPipelineService::applyLearnedDepthCandidatesAfterConsistency()
    {
        if (!_config.enableLearnedMvsCandidates)
        {
            return;
        }
        const QDir candidate_directory(QString::fromStdString(_config.learnedMvsCandidateDirectory));
        if (_config.learnedMvsCandidateDirectory.empty() || !candidate_directory.exists())
        {
            LOG_WARN(
                QStringLiteral("[MVS][学习候选] 候选目录不存在，已跳过：%1").arg(candidate_directory.absolutePath()));
            return;
        }

        LearnedDepthCandidateGateOptions options;
        options.minimumCandidateConfidence = std::clamp(_config.learnedMvsMinimumConfidence, 0.0f, 1.0f);
        options.minimumGeometryObservationCount = std::max(2, _config.learnedMvsMinimumGeometryObservations);
        options.maximumInverseDepthRelativeSpread = std::max(0.0f, _config.learnedMvsMaximumInverseDepthSpread);
        options.maximumRelativeDepthDifference = std::max(0.0f, _config.learnedMvsMaximumRelativeDepthDifference);
        options.replacementConfidenceMargin = std::max(0.0f, _config.learnedMvsReplacementConfidenceMargin);

        for (int frame_index = 0; frame_index < static_cast<int>(_depthFrames.size()); ++frame_index)
        {
            DepthFrameResult& frame = _depthFrames[frame_index];
            QJsonObject diagnostics;
            diagnostics.insert(QStringLiteral("enabled"), true);
            const QString depth_path =
                candidate_directory.filePath(QStringLiteral("learned_depth_%1.bin").arg(frame_index));
            const QString confidence_path =
                candidate_directory.filePath(QStringLiteral("learned_depth_%1_conf.bin").arg(frame_index));
            diagnostics.insert(QStringLiteral("depth_path"), depth_path);
            diagnostics.insert(QStringLiteral("confidence_path"), confidence_path);
            if (!frame.success || !frame.depthMap || frame.depthMap->empty() || !frame.geometrySupportCount ||
                !frame.inverseDepthMean || !frame.inverseDepthRelativeSpread)
            {
                diagnostics.insert(QStringLiteral("status"), QStringLiteral("missing_independent_geometry_evidence"));
                frame.learnedCandidateDiagnostics = diagnostics;
                continue;
            }
            if (!QFileInfo::exists(depth_path) || !QFileInfo::exists(confidence_path))
            {
                diagnostics.insert(QStringLiteral("status"), QStringLiteral("candidate_artifact_missing"));
                frame.learnedCandidateDiagnostics = diagnostics;
                continue;
            }

            cv::Mat candidate_depth;
            cv::Mat candidate_confidence;
            const xjw::common::OperationResult depth_result =
                xjw::core::project::loadDepthMatStorage(depth_path, &candidate_depth);
            const xjw::common::OperationResult confidence_result =
                xjw::core::project::loadDepthMatStorage(confidence_path, &candidate_confidence);
            if (!depth_result.ok || !confidence_result.ok)
            {
                diagnostics.insert(QStringLiteral("status"), QStringLiteral("candidate_load_failed"));
                diagnostics.insert(QStringLiteral("error"),
                                   !depth_result.ok ? depth_result.errorMessage : confidence_result.errorMessage);
                frame.learnedCandidateDiagnostics = diagnostics;
                continue;
            }
            if (!frame.confidence || frame.confidence->empty())
            {
                frame.confidence = QSharedPointer<cv::Mat>::create(frame.depthMap->size(), CV_32FC1, cv::Scalar(0.0f));
            }
            cv::Mat accepted_mask;
            const LearnedDepthCandidateGateStats stats = gateLearnedDepthCandidate(*frame.depthMap,
                                                                                   *frame.confidence,
                                                                                   candidate_depth,
                                                                                   candidate_confidence,
                                                                                   *frame.geometrySupportCount,
                                                                                   *frame.inverseDepthMean,
                                                                                   *frame.inverseDepthRelativeSpread,
                                                                                   &accepted_mask,
                                                                                   options);
            diagnostics.insert(QStringLiteral("status"),
                               stats.validInputs ? QStringLiteral("geometry_gated")
                                                 : QStringLiteral("invalid_candidate_shape_or_type"));
            diagnostics.insert(QStringLiteral("candidate_pixel_count"), stats.candidatePixelCount);
            diagnostics.insert(QStringLiteral("geometry_supported_pixel_count"), stats.geometrySupportedPixelCount);
            diagnostics.insert(QStringLiteral("accepted_pixel_count"), stats.acceptedPixelCount);
            diagnostics.insert(QStringLiteral("filled_pixel_count"), stats.filledPixelCount);
            diagnostics.insert(QStringLiteral("replaced_pixel_count"), stats.replacedPixelCount);
            diagnostics.insert(QStringLiteral("rejected_confidence_count"), stats.rejectedConfidenceCount);
            diagnostics.insert(QStringLiteral("rejected_geometry_count"), stats.rejectedGeometryCount);
            diagnostics.insert(QStringLiteral("rejected_depth_difference_count"), stats.rejectedDepthDifferenceCount);
            frame.learnedCandidateDiagnostics = diagnostics;
            if (stats.acceptedPixelCount <= 0)
            {
                continue;
            }
            frame.learnedCandidateAcceptedMask = QSharedPointer<cv::Mat>::create(std::move(accepted_mask));
            if (!frame.depthProvenance || frame.depthProvenance->empty())
            {
                frame.depthProvenance = QSharedPointer<cv::Mat>::create(initializeDepthProvenance(*frame.depthMap));
            }
            updateDepthProvenance(*frame.depthProvenance,
                                  *frame.depthMap,
                                  cv::Mat(),
                                  cv::Mat(),
                                  cv::Mat(),
                                  cv::Mat(),
                                  *frame.learnedCandidateAcceptedMask);
            LOG_INFO(QStringLiteral("[MVS][帧 %1][学习候选] candidate=%2 geometry=%3 "
                                    "accepted=%4 fill=%5 replace=%6")
                         .arg(frame_index)
                         .arg(stats.candidatePixelCount)
                         .arg(stats.geometrySupportedPixelCount)
                         .arg(stats.acceptedPixelCount)
                         .arg(stats.filledPixelCount)
                         .arg(stats.replacedPixelCount));
        }
    }

    void MvsPipelineService::runDepthPoseRefinementCandidateStage(bool residentDepthFrames)
    {
        if (!_config.depthPoseRefinement.enabled)
        {
            return;
        }

        DepthPoseRefinementStageResult stage;
        stage.enabled = true;
        stage.candidateOnly = true;
        stage.anchorCameraIndex = _config.depthPoseRefinement.optimizer.anchorCameraIndex;
        if (residentDepthFrames)
        {
            std::vector<DepthPoseRefinementFrame> frames;
            frames.reserve(_depthFrames.size());
            for (int frame_index = 0; frame_index < static_cast<int>(_depthFrames.size()); ++frame_index)
            {
                const DepthFrameResult& depth_frame = _depthFrames[static_cast<std::size_t>(frame_index)];
                DepthPoseRefinementFrame frame;
                frame.cameraIndex = frame_index;
                frame.camera = depth_frame.cameraModel.isValid()
                                   ? depth_frame.cameraModel
                                   : mvsPinholeCamera(_views[static_cast<std::size_t>(frame_index)].camera);
                frame.depthMap = depth_frame.depthMap ? *depth_frame.depthMap : cv::Mat();
                frame.normalMap = depth_frame.normalMap ? *depth_frame.normalMap : cv::Mat();
                frame.confidence = depth_frame.confidence ? *depth_frame.confidence : cv::Mat();
                frame.adaptiveSupportWeight =
                    depth_frame.adaptiveGeometrySupportWeight ? *depth_frame.adaptiveGeometrySupportWeight : cv::Mat();
                frame.adaptiveEffectiveViewCount = depth_frame.adaptiveGeometryEffectiveViewCount
                                                       ? *depth_frame.adaptiveGeometryEffectiveViewCount
                                                       : cv::Mat();
                frame.adaptiveConflictRatio =
                    depth_frame.adaptiveGeometryConflictRatio ? *depth_frame.adaptiveGeometryConflictRatio : cv::Mat();
                frame.sourceCameraIndices = depth_frame.sourceViewIndices;
                frames.push_back(std::move(frame));
            }
            stage = DepthPoseRefinementStage::buildCandidates(frames, _config.depthPoseRefinement);
        }
        else
        {
            stage.candidates.reserve(_depthFrames.size());
            for (int frame_index = 0; frame_index < static_cast<int>(_depthFrames.size()); ++frame_index)
            {
                DepthPoseRefinementCandidate candidate;
                candidate.cameraIndex = frame_index;
                candidate.reason = "streaming_depth_not_resident";
                stage.candidates.push_back(std::move(candidate));
            }
        }

        {
            std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
            for (const DepthPoseRefinementCandidate& candidate : stage.candidates)
            {
                if (candidate.cameraIndex < 0 || candidate.cameraIndex >= static_cast<int>(_depthFrames.size()))
                {
                    continue;
                }
                DepthFrameResult& frame = _depthFrames[static_cast<std::size_t>(candidate.cameraIndex)];
                frame.poseRefinementDiagnostics = depthPoseRefinementCandidateToJson(candidate, stage);
                if (candidate.accepted && candidate.derivedCamera.isValid())
                {
                    frame.derivedCameraModel = candidate.derivedCamera;
                }
                const QJsonObject derived_camera =
                    frame.derivedCameraModel.isValid() ? cameraModelToJson(frame.derivedCameraModel) : QJsonObject{};
                _workspaceManifest.updatePoseRefinement(
                    candidate.cameraIndex, frame.poseRefinementDiagnostics, derived_camera);
            }
            QString manifest_error;
            if (!_workspaceManifestPath.isEmpty() && !persistWorkspaceManifest(&manifest_error))
            {
                LOG_WARN(QStringLiteral("[MVS] 写入位姿细化候选诊断失败: %1").arg(manifest_error));
            }
        }

        LOG_INFO(
            QStringLiteral("[MVS] 深度约束位姿细化候选完成: accepted=%1/%2 mode=candidate_only; "
                           "项目相机与本轮深度均未修改")
                .arg(std::count_if(stage.candidates.begin(),
                                   stage.candidates.end(),
                                   [](const DepthPoseRefinementCandidate& candidate) { return candidate.accepted; }))
                .arg(stage.candidates.size()));
    }
} // namespace xjw::mvs
