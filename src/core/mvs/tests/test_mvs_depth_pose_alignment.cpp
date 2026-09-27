#include "DepthPoseAlignmentRefiner.h"
#include "DepthPoseRefinementStage.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace
{

    constexpr double kPi = 3.14159265358979323846;

    cv::Matx33d rotationZ(double angle)
    {
        return cv::Matx33d(
            std::cos(angle), -std::sin(angle), 0.0, std::sin(angle), std::cos(angle), 0.0, 0.0, 0.0, 1.0);
    }

    placamera::FramePinholeModel
    makeCamera(double center_z, int camera_index = 0, const std::string& frame_name = "pose-refinement-world")
    {
        const placamera::FrameId frame(frame_name);
        const std::string id = "pose-refinement-" + std::to_string(camera_index);
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId(id + "-definition"),
                                                      placamera::FrameIntrinsics{80.0, 80.0, 31.5, 31.5},
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        return placamera::FramePinholeModel::create(
            placamera::CameraInstanceId(id),
            placamera::ImageId(id + "-image"),
            definition,
            {64, 64},
            placamera::Pose::create(frame, {0.0, 0.0, center_z}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    }

    cv::Vec3d cameraCoordinates(const placamera::FramePinholeModel& camera, const cv::Vec3d& world)
    {
        const auto& pose = camera.pose();
        const cv::Vec3d offset(world[0] - pose.center[0], world[1] - pose.center[1], world[2] - pose.center[2]);
        const auto& rotation = pose.cameraToWorldRotation;
        return cv::Vec3d(rotation[0] * offset[0] + rotation[3] * offset[1] + rotation[6] * offset[2],
                         rotation[1] * offset[0] + rotation[4] * offset[1] + rotation[7] * offset[2],
                         rotation[2] * offset[0] + rotation[5] * offset[1] + rotation[8] * offset[2]);
    }

    xjw::mvs::DepthPoseRefinementFrame makePlaneFrame(int camera_index, double center_z, int source_index)
    {
        xjw::mvs::DepthPoseRefinementFrame frame;
        frame.cameraIndex = camera_index;
        frame.camera = std::make_shared<const placamera::FramePinholeModel>(makeCamera(center_z, camera_index));
        frame.depthMap = cv::Mat(64, 64, CV_32FC1, cv::Scalar(2.0f));
        frame.normalMap = cv::Mat(64, 64, CV_32FC3, cv::Scalar(0.0f, 0.0f, 1.0f));
        frame.confidence = cv::Mat(64, 64, CV_32FC1, cv::Scalar(0.95f));
        frame.adaptiveSupportWeight = cv::Mat(64, 64, CV_32FC1, cv::Scalar(0.90f));
        frame.adaptiveEffectiveViewCount = cv::Mat(64, 64, CV_32FC1, cv::Scalar(2.0f));
        frame.adaptiveConflictRatio = cv::Mat(64, 64, CV_32FC1, cv::Scalar(0.02f));
        frame.sourceCameraIndices = {source_index};
        return frame;
    }

    std::vector<xjw::mvs::DepthPoseAlignmentSample> makeSamples()
    {
        std::vector<xjw::mvs::DepthPoseAlignmentSample> samples;
        const cv::Matx33d drift_rotation = rotationZ(1.2 * kPi / 180.0);
        const cv::Vec3d drift_translation(0.006, -0.004, 0.003);
        for (int latitude = -4; latitude <= 4; ++latitude)
        {
            for (int longitude = -6; longitude <= 6; ++longitude)
            {
                const double theta = static_cast<double>(longitude) * 0.10;
                const double phi = static_cast<double>(latitude) * 0.09;
                const cv::Vec3d target(std::cos(phi) * std::cos(theta), std::cos(phi) * std::sin(theta), std::sin(phi));
                xjw::mvs::DepthPoseAlignmentSample sample;
                sample.cameraIndex = 1;
                sample.targetPointWorld = target;
                sample.targetNormalWorld = target;
                sample.sourcePointWorld = drift_rotation * target + drift_translation;
                sample.confidence = 0.9;
                samples.push_back(sample);
            }
        }
        return samples;
    }

    TEST(DepthPoseAlignmentRefinerTest, ReducesRobustPointToPlaneResidual)
    {
        const std::vector<xjw::mvs::DepthPoseAlignmentSample> samples = makeSamples();
        xjw::mvs::DepthPoseAlignmentOptions options;
        options.enabled = true;
        options.maximumTranslation = 0.02;
        options.maximumRotationDegrees = 2.0;
        options.huberDelta = 0.002;

        const xjw::mvs::DepthPoseAlignmentResult result = xjw::mvs::DepthPoseAlignmentRefiner::refine(samples, options);

        ASSERT_EQ(result.corrections.size(), 1);
        const auto& correction = result.corrections.front();
        EXPECT_TRUE(result.acceptedAny);
        EXPECT_TRUE(correction.accepted) << correction.reason;
        EXPECT_LT(correction.residualP90After, correction.residualP90Before * 0.2);
        EXPECT_LE(cv::norm(correction.translation), options.maximumTranslation);
    }

    TEST(DepthPoseAlignmentRefinerTest, KeepsAnchorCameraFixed)
    {
        std::vector<xjw::mvs::DepthPoseAlignmentSample> samples = makeSamples();
        for (auto& sample : samples)
        {
            sample.cameraIndex = 3;
        }
        xjw::mvs::DepthPoseAlignmentOptions options;
        options.enabled = true;
        options.anchorCameraIndex = 3;

        const xjw::mvs::DepthPoseAlignmentResult result = xjw::mvs::DepthPoseAlignmentRefiner::refine(samples, options);

        ASSERT_EQ(result.corrections.size(), 1);
        EXPECT_FALSE(result.corrections.front().accepted);
        EXPECT_EQ(result.corrections.front().reason, "anchor_camera");
        EXPECT_LE(cv::norm(result.corrections.front().rotation - cv::Matx33d::eye()), 1.0e-12);
    }

    TEST(DepthPoseAlignmentRefinerTest, IgnoresOccludedAndInvalidSamples)
    {
        std::vector<xjw::mvs::DepthPoseAlignmentSample> samples = makeSamples();
        for (int index = 0; index < 10; ++index)
        {
            xjw::mvs::DepthPoseAlignmentSample outlier = samples[index];
            outlier.targetPointWorld += cv::Vec3d(10.0, -8.0, 6.0);
            outlier.occluded = true;
            samples.push_back(outlier);
        }
        xjw::mvs::DepthPoseAlignmentOptions options;
        options.enabled = true;

        const auto result = xjw::mvs::DepthPoseAlignmentRefiner::refine(samples, options);

        ASSERT_EQ(result.corrections.size(), 1);
        EXPECT_TRUE(result.corrections.front().accepted);
        EXPECT_EQ(result.corrections.front().correspondenceCount, 117);
    }

    TEST(DepthPoseAlignmentRefinerTest, IsDisabledByDefault)
    {
        const auto result = xjw::mvs::DepthPoseAlignmentRefiner::refine(makeSamples());

        EXPECT_FALSE(result.enabled);
        EXPECT_FALSE(result.acceptedAny);
        EXPECT_TRUE(result.corrections.empty());
    }

    TEST(DepthPoseAlignmentRefinerTest, RejectsCandidateWithoutResidualImprovement)
    {
        std::vector<xjw::mvs::DepthPoseAlignmentSample> samples = makeSamples();
        for (auto& sample : samples)
        {
            sample.sourcePointWorld = sample.targetPointWorld;
        }
        xjw::mvs::DepthPoseAlignmentOptions options;
        options.enabled = true;

        const auto result = xjw::mvs::DepthPoseAlignmentRefiner::refine(samples, options);

        ASSERT_EQ(result.corrections.size(), 1);
        EXPECT_FALSE(result.corrections.front().accepted);
        EXPECT_EQ(result.corrections.front().reason, "insufficient_residual_improvement");
    }

    TEST(DepthPoseRefinementStageTest, IsDisabledByDefault)
    {
        const auto result = xjw::mvs::DepthPoseRefinementStage::buildCandidates({});

        EXPECT_FALSE(result.enabled);
        EXPECT_TRUE(result.candidates.empty());
    }

    TEST(DepthPoseRefinementStageTest, ProducesSafeCandidateWithoutMutatingInputCamera)
    {
        std::vector<xjw::mvs::DepthPoseRefinementFrame> frames;
        frames.push_back(makePlaneFrame(0, 0.0, 1));
        frames.push_back(makePlaneFrame(1, 0.006, 0));
        const std::array<double, 3> original_center = frames[1].camera->pose().center;

        xjw::mvs::DepthPoseRefinementOptions options;
        options.enabled = true;
        options.samplingStridePixels = 8;
        options.optimizer.anchorCameraIndex = 0;
        options.optimizer.minimumCorrespondences = 24;
        options.optimizer.maximumTranslation = 0.02;
        options.optimizer.requiredP90ImprovementRatio = 0.9;

        const auto result = xjw::mvs::DepthPoseRefinementStage::buildCandidates(frames, options);

        ASSERT_EQ(result.candidates.size(), 2);
        const auto& anchor = result.candidates[0];
        const auto& candidate = result.candidates[1];
        EXPECT_FALSE(anchor.accepted);
        EXPECT_EQ(anchor.reason, "anchor_camera");
        ASSERT_TRUE(candidate.accepted) << candidate.reason;
        EXPECT_TRUE(candidate.evidenceComplete);
        EXPECT_GE(candidate.projectionRetentionRatio, 0.98);
        EXPECT_LT(candidate.correction.residualP90After, candidate.correction.residualP90Before * 0.1);
        ASSERT_TRUE(candidate.derivedCamera);
        EXPECT_NEAR(candidate.derivedCamera->pose().center[2], 0.0, 1.0e-5);
        EXPECT_EQ(candidate.derivedCamera->instanceId(), frames[1].camera->instanceId());
        EXPECT_EQ(candidate.derivedCamera->definitionId(), frames[1].camera->definitionId());
        EXPECT_DOUBLE_EQ(frames[1].camera->pose().center[2], original_center[2]);
    }

    TEST(DepthPoseRefinementStageTest, RejectsOccludedCorrespondences)
    {
        std::vector<xjw::mvs::DepthPoseRefinementFrame> frames;
        frames.push_back(makePlaneFrame(0, 0.0, 1));
        frames.push_back(makePlaneFrame(1, 0.006, 0));
        frames[0].depthMap.setTo(1.0f);

        xjw::mvs::DepthPoseRefinementOptions options;
        options.enabled = true;
        options.samplingStridePixels = 8;
        options.optimizer.anchorCameraIndex = 0;

        const auto result = xjw::mvs::DepthPoseRefinementStage::buildCandidates(frames, options);

        ASSERT_EQ(result.candidates.size(), 2);
        EXPECT_FALSE(result.candidates[1].accepted);
        EXPECT_GT(result.candidates[1].occludedCandidateCount, 0);
        EXPECT_EQ(result.candidates[1].reason, "no_usable_correspondences");
    }

    TEST(DepthPoseRefinementStageTest, RequiresCompleteAdaptiveEvidence)
    {
        std::vector<xjw::mvs::DepthPoseRefinementFrame> frames;
        frames.push_back(makePlaneFrame(0, 0.0, 1));
        frames.push_back(makePlaneFrame(1, 0.006, 0));
        frames[1].adaptiveConflictRatio.release();

        xjw::mvs::DepthPoseRefinementOptions options;
        options.enabled = true;
        options.optimizer.anchorCameraIndex = 0;

        const auto result = xjw::mvs::DepthPoseRefinementStage::buildCandidates(frames, options);

        ASSERT_EQ(result.candidates.size(), 2);
        EXPECT_FALSE(result.candidates[1].accepted);
        EXPECT_FALSE(result.candidates[1].evidenceComplete);
        EXPECT_EQ(result.candidates[1].reason, "incomplete_geometry_evidence");
    }

    TEST(DepthPoseRefinementStageTest, RejectsMissingOrMismatchedCameraGeometry)
    {
        std::vector<xjw::mvs::DepthPoseRefinementFrame> frames;
        frames.push_back(makePlaneFrame(0, 0.0, 1));
        frames.push_back(makePlaneFrame(1, 0.006, 0));
        frames[0].camera.reset();
        frames[1].depthMap = cv::Mat(32, 32, CV_32FC1, cv::Scalar(2.0f));

        xjw::mvs::DepthPoseRefinementOptions options;
        options.enabled = true;
        const auto result = xjw::mvs::DepthPoseRefinementStage::buildCandidates(frames, options);

        ASSERT_EQ(result.candidates.size(), 2);
        for (const auto& candidate : result.candidates)
        {
            EXPECT_FALSE(candidate.evidenceComplete);
            EXPECT_FALSE(candidate.derivedCamera);
            EXPECT_EQ(candidate.reason, "incomplete_geometry_evidence");
        }
    }

    TEST(DepthPoseRefinementStageTest, RejectsCrossFrameCorrespondences)
    {
        std::vector<xjw::mvs::DepthPoseRefinementFrame> frames;
        frames.push_back(makePlaneFrame(0, 0.0, 1));
        frames.push_back(makePlaneFrame(1, 0.006, 0));
        frames[1].camera = std::make_shared<const placamera::FramePinholeModel>(makeCamera(0.006, 1, "another-world"));

        xjw::mvs::DepthPoseRefinementOptions options;
        options.enabled = true;
        options.samplingStridePixels = 8;
        options.optimizer.anchorCameraIndex = 0;
        const auto result = xjw::mvs::DepthPoseRefinementStage::buildCandidates(frames, options);

        ASSERT_EQ(result.candidates.size(), 2);
        EXPECT_TRUE(result.candidates[1].evidenceComplete);
        EXPECT_EQ(result.candidates[1].generatedCorrespondenceCount, 0);
        EXPECT_EQ(result.candidates[1].reason, "no_usable_correspondences");
    }

    TEST(DepthPoseRefinementStageTest, DerivesNormalsFromDepthWhenBackendOmitsThem)
    {
        std::vector<xjw::mvs::DepthPoseRefinementFrame> frames;
        frames.push_back(makePlaneFrame(0, 0.0, 1));
        frames.push_back(makePlaneFrame(1, 0.006, 0));
        frames[0].normalMap.release();
        frames[1].normalMap.release();

        xjw::mvs::DepthPoseRefinementOptions options;
        options.enabled = true;
        options.samplingStridePixels = 8;
        options.optimizer.anchorCameraIndex = 0;
        options.optimizer.minimumCorrespondences = 24;
        options.optimizer.maximumTranslation = 0.02;
        options.optimizer.requiredP90ImprovementRatio = 0.9;

        const auto result = xjw::mvs::DepthPoseRefinementStage::buildCandidates(frames, options);

        ASSERT_EQ(result.candidates.size(), 2);
        EXPECT_TRUE(result.candidates[0].evidenceComplete);
        const auto& candidate = result.candidates[1];
        EXPECT_TRUE(candidate.evidenceComplete);
        EXPECT_GT(candidate.generatedCorrespondenceCount, 0);
        EXPECT_TRUE(candidate.accepted) << candidate.reason;
    }

    TEST(DepthPoseRefinementStageTest, ProjectionCoverageGateCanVetoOptimizerCandidate)
    {
        std::vector<xjw::mvs::DepthPoseRefinementFrame> frames;
        frames.push_back(makePlaneFrame(0, 0.0, 1));
        frames.push_back(makePlaneFrame(1, 0.006, 0));

        xjw::mvs::DepthPoseRefinementOptions options;
        options.enabled = true;
        options.samplingStridePixels = 8;
        options.minimumProjectionRetentionRatio = 1.01;
        options.optimizer.anchorCameraIndex = 0;
        options.optimizer.minimumCorrespondences = 24;
        options.optimizer.requiredP90ImprovementRatio = 0.9;

        const auto result = xjw::mvs::DepthPoseRefinementStage::buildCandidates(frames, options);

        ASSERT_EQ(result.candidates.size(), 2);
        EXPECT_FALSE(result.candidates[1].accepted);
        EXPECT_EQ(result.candidates[1].reason, "projection_coverage_regressed");
    }

    TEST(DepthPoseRefinementStageTest, DerivedCameraPreservesCorrectedCameraCoordinates)
    {
        const auto source = makeCamera(0.0);
        const auto camera = source.withPose(
            source.instanceId(),
            placamera::Pose::create(source.groundFrame(), {0.3, -0.2, 0.5}, source.pose().cameraToWorldRotation));
        xjw::mvs::DepthPoseAlignmentCorrection correction;
        correction.accepted = true;
        correction.pivotWorld = cv::Vec3d(0.1, 0.2, -0.3);
        correction.rotation = rotationZ(0.12);
        correction.translation = cv::Vec3d(0.02, -0.01, 0.03);
        const placamera::FramePinholeModel derived =
            xjw::mvs::DepthPoseRefinementStage::deriveCameraCandidate(camera, correction);

        const cv::Vec3d original_world(0.7, 0.1, 2.0);
        const cv::Vec3d original_camera = cameraCoordinates(camera, original_world);
        const cv::Vec3d corrected_world =
            xjw::mvs::DepthPoseAlignmentRefiner::applyCorrection(correction, original_world);
        const cv::Vec3d derived_camera = cameraCoordinates(derived, corrected_world);

        for (int axis = 0; axis < 3; ++axis)
        {
            EXPECT_NEAR(derived_camera[axis], original_camera[axis], 1.0e-10);
        }
    }

} // namespace
