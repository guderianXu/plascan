#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow::workflow_detail
{
    using namespace workflow_detail;

    bool validateUsableDepthSceneProfileBatch(const QVector<DepthFrameArtifact>& artifacts,
                                              QString* canonical_scene_profile,
                                              int* invalid_ref_index,
                                              QString* invalid_scene_profile)
    {
        QString canonical;
        for (const DepthFrameArtifact& artifact : artifacts)
        {
            if (artifact.role == xjw::mvs::DepthFrameRole::Excluded)
            {
                continue;
            }
            if (!xjw::mvs::extendCanonicalDepthSceneProfileBatch(artifact.sceneProfile, &canonical))
            {
                if (invalid_ref_index != nullptr)
                {
                    *invalid_ref_index = artifact.refIndex;
                }
                if (invalid_scene_profile != nullptr)
                {
                    *invalid_scene_profile = artifact.sceneProfile;
                }
                return false;
            }
        }
        if (canonical_scene_profile != nullptr)
        {
            *canonical_scene_profile = canonical;
        }
        return true;
    }

    float visibilityOccupancyMedianVoxelStep(const xjw::mesh::DepthTsdfLayout& layout, int occupancy_resolution)
    {
        std::array<float, 3> occupancy_steps{};
        for (int axis = 0; axis < 3; ++axis)
        {
            occupancy_steps[axis] = (layout.boundsMax[axis] - layout.boundsMin[axis]) /
                                    static_cast<float>(std::max(2, occupancy_resolution - 1));
        }
        std::sort(occupancy_steps.begin(), occupancy_steps.end());
        return std::max(occupancy_steps[1], 1.0e-6f);
    }

    xjw::mesh::DepthConstrainedSurfaceRefineOptions
    makeVisibilityOccupancyDepthRefineOptions(const QJsonObject& settings,
                                              const xjw::mesh::DepthTsdfLayout& layout,
                                              int occupancy_resolution,
                                              bool orbital_workspace)
    {
        xjw::mesh::DepthConstrainedSurfaceRefineOptions options;
        const bool measured_depth_only = interpolationIsDisabled(settings);
        options.depthRefine.measuredDepthSamplesOnly = measured_depth_only;
        options.depthRefine.primaryFramesOnly = measured_depth_only;
        options.maximumAcceptedBlend = measured_depth_only ? 0.25f : 1.0f;
        options.maximumCumulativeProjectedP95DisplacementPixels =
            measured_depth_only
                ? std::clamp(static_cast<float>(
                                 settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMaximumProjectedP95Pixels"))
                                     .toDouble(0.75)),
                             0.25f,
                             4.0f)
                : 0.0f;
        const float occupancy_step = visibilityOccupancyMedianVoxelStep(layout, occupancy_resolution);
        options.passes = qBound(1,
                                settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthRefinementPasses"))
                                    .toInt(orbital_workspace ? 4 : 1),
                                4);
        options.depthRefine.maximumEvidenceDistance =
            occupancy_step *
            std::clamp(
                static_cast<float>(
                    settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMaximumEvidenceVoxels")).toDouble(3.0)),
                0.5f,
                12.0f);
        options.depthRefine.maximumDisplacement =
            occupancy_step *
            std::clamp(static_cast<float>(
                           settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMaximumDisplacementVoxels"))
                               .toDouble(orbital_workspace ? 0.60 : 0.20)),
                       0.02f,
                       1.0f);
        options.depthRefine.minimumViewCount =
            qBound(2, settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMinimumViews")).toInt(2), 8);
        options.depthRefine.minimumNativeViewCount =
            qBound(0,
                   settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMinimumNativeViews")).toInt(1),
                   options.depthRefine.minimumViewCount);
        options.depthRefine.minimumDepthConfidence = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMinimumConfidence")).toDouble(0.20)),
            0.0f,
            1.0f);
        options.depthRefine.repairedObservationWeight =
            std::clamp(static_cast<float>(
                           settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthRepairedWeight")).toDouble(0.25)),
                       0.0f,
                       1.0f);
        options.depthRefine.enableInverseDepthSpreadWeighting =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthSpreadWeighting")).toBool(true);
        options.depthRefine.inverseDepthSpreadWeightKnee = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthSpreadWeightKnee")).toDouble(0.005)),
            0.0f,
            0.099f);
        options.depthRefine.inverseDepthSpreadWeightZero = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthSpreadWeightZero")).toDouble(0.015)),
            options.depthRefine.inverseDepthSpreadWeightKnee + 1.0e-6f,
            0.10f);
        options.depthRefine.minimumInverseDepthSpreadWeight = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMinimumSpreadWeight")).toDouble(0.05)),
            0.0f,
            1.0f);
        options.depthRefine.maximumViewMedianAbsoluteDeviation =
            occupancy_step *
            std::clamp(
                static_cast<float>(
                    settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMaximumMadVoxels")).toDouble(1.5)),
                0.25f,
                6.0f);
        options.depthRefine.enableCrossViewBiasCompensation =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthBiasCompensation")).toBool(true);
        options.depthRefine.minimumCrossViewBiasPairSamples = qBound(
            8, settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthBiasMinimumPairSamples")).toInt(64), 4096);
        options.depthRefine.maximumCrossViewBias =
            occupancy_step *
            std::clamp(
                static_cast<float>(
                    settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMaximumBiasVoxels")).toDouble(0.75)),
                0.05f,
                3.0f);
        options.depthRefine.minimumAnchorWeight = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMinimumAnchorWeight")).toDouble(0.05)),
            0.0f,
            1.0f);
        options.depthRefine.regularizationMaximumNormalAngleDegrees = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMaximumNormalAngleDegrees")).toDouble(55.0)),
            5.0f,
            89.0f);
        options.depthRefine.enableGlobalRobustSolver =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthGlobalRobustSolver")).toBool(true);
        options.depthRefine.globalSolverIrlsIterations = qBound(
            1, settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthGlobalSolverIrlsIterations")).toInt(4), 10);
        options.depthRefine.globalSolverMaximumPcgIterations = qBound(
            1,
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthGlobalSolverMaximumPcgIterations")).toInt(120),
            500);
        options.depthRefine.globalSolverConvergenceTolerance = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthGlobalSolverTolerance")).toDouble(1.0e-5)),
            1.0e-9f,
            1.0e-2f);
        options.depthRefine.globalSolverRobustScaleMultiplier = std::clamp(
            static_cast<float>(
                settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthGlobalSolverRobustScale")).toDouble(0.5)),
            0.01f,
            4.0f);
        options.depthRefine.globalSolverLaplacianWeight = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthGlobalSolverLaplacian"))
                                   .toDouble(orbital_workspace ? 2.50 : 0.60)),
            0.0f,
            20.0f);
        options.depthRefine.globalSolverHullPriorWeight = std::clamp(
            static_cast<float>(settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthGlobalSolverCarrierPrior"))
                                   .toDouble(orbital_workspace ? 0.04 : 0.05)),
            1.0e-6f,
            1.0f);
        options.minimumAreaRatio =
            std::clamp(settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMinimumAreaRatio"))
                           .toDouble(orbital_workspace ? 0.80 : 0.97),
                       0.80,
                       1.0);
        options.maximumAreaRatio = std::clamp(
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMaximumAreaRatio")).toDouble(1.03), 1.0, 1.20);
        options.minimumVolumeRatio =
            std::clamp(settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMinimumVolumeRatio"))
                           .toDouble(orbital_workspace ? 0.85 : 0.98),
                       0.80,
                       1.0);
        options.maximumVolumeRatio = std::clamp(
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMaximumVolumeRatio")).toDouble(1.02), 1.0, 1.20);
        options.minimumFaceNormalDot = std::clamp(
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMinimumFaceNormalDot")).toDouble(0.25),
            -1.0,
            1.0);
        options.minimumFaceAreaRatio = std::clamp(
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthMinimumFaceAreaRatio")).toDouble(0.10),
            0.01,
            1.0);
        options.removeMedianNormalBias =
            settings.value(QStringLiteral("tsdfVisibilityOccupancyDepthRemoveMedianNormalBias")).toBool(true);
        return options;
    }

    int maximumReliableOrbitalResolution(const QVector<DepthTsdfFrame>& frames, int requestedResolution)
    {
        std::vector<int> image_dimensions;
        image_dimensions.reserve(static_cast<std::size_t>(frames.size()));
        for (const DepthTsdfFrame& frame : frames)
        {
            if (!frame.depth.empty())
            {
                image_dimensions.push_back(std::min(frame.depth.cols, frame.depth.rows));
            }
        }
        if (image_dimensions.empty())
        {
            return requestedResolution;
        }
        const auto middle = image_dimensions.begin() + image_dimensions.size() / 2;
        std::nth_element(image_dimensions.begin(), middle, image_dimensions.end());
        const int median_dimension = *middle;
        const int sampled_resolution = std::max(192, median_dimension * 2 / 5);
        const int aligned_resolution = std::max(192, sampled_resolution / 64 * 64);
        return std::min(requestedResolution, std::clamp(aligned_resolution, 192, 384));
    }
} // namespace xjw::mesh::workflow::workflow_detail
