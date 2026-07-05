#include "AerialTriangulationWorkflow.h"

#include <gtest/gtest.h>

namespace
{

xjw::gui::AerialTriangulationWorkflowOptions makeBaseOptions()
{
    xjw::gui::AerialTriangulationWorkflowOptions options;
    options.images = {
        QStringLiteral("E:/data/img_001.jpg"),
        QStringLiteral("E:/data/img_002.jpg"),
        QStringLiteral("E:/data/img_003.jpg")
    };
    options.cameraPaths = {
        QStringLiteral("E:/data/cameras/img_001.tsai"),
        QStringLiteral("E:/data/cameras/img_002.tsai"),
        QStringLiteral("E:/data/cameras/img_003.tsai")
    };
    options.projectPath = QStringLiteral("E:/project/test.plascan");
    options.outputDir = QStringLiteral("E:/project/assets/aerial_triangulation");
    options.projectMeta.insert(QStringLiteral("name"), QStringLiteral("unit"));
    return options;
}

} // namespace

TEST(AerialTriangulationWorkflowCoreTest, HighestQualityMapsToFullResolutionSfmAndTieLimits)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("highest");
    options.keypointLimit = 40000;
    options.tiepointLimit = 4000;
    options.featureAlgorithm = QStringLiteral("sift");
    options.matchAlgorithm = QStringLiteral("lightglue");
    options.guidedImageMatching = true;

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.sfmOptions.images, options.images);
    EXPECT_EQ(resolved.sfmOptions.cameraPaths, options.cameraPaths);
    EXPECT_EQ(resolved.sfmOptions.plascanPath, options.projectPath);
    EXPECT_EQ(resolved.sfmOptions.projectMeta, options.projectMeta);
    EXPECT_EQ(resolved.sfmOptions.outputDir,
              QStringLiteral("E:/project/assets/aerial_triangulation/sfm_sparse"));
    EXPECT_EQ(resolved.sfmOptions.quality, 3);
    EXPECT_EQ(resolved.sfmOptions.featureMaxImageDim, -1);
    EXPECT_EQ(resolved.sfmOptions.skeletonFeatureMaxKeypoints, 40000);
    EXPECT_EQ(resolved.sfmOptions.maxTiePointsPerImage, 4000);
    EXPECT_EQ(resolved.sfmOptions.maxTiePointsPerGridCell, 500);
    EXPECT_EQ(resolved.sfmOptions.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(resolved.sfmOptions.matchAlgorithm, QStringLiteral("lightglue"));
    EXPECT_TRUE(resolved.sfmOptions.enableGuidedRematching);
    EXPECT_TRUE(resolved.sfmOptions.enableTwoStageMatching);
    EXPECT_TRUE(resolved.resolvedSettings.value(QStringLiteral("adaptive_known_pose_soft_prior")).toBool());
}

TEST(AerialTriangulationWorkflowCoreTest, LowQualityUsesReducedImageScaleAndConservativeBudgets)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("low");
    options.keypointLimit = 40000;
    options.tiepointLimit = 4000;

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.sfmOptions.quality, 0);
    EXPECT_EQ(resolved.sfmOptions.featureMaxImageDim, 2048);
    EXPECT_EQ(resolved.sfmOptions.skeletonFeatureMaxKeypoints, 10000);
    EXPECT_EQ(resolved.sfmOptions.maxTiePointsPerImage, 1000);
    EXPECT_EQ(resolved.sfmOptions.maxTiePointsPerGridCell, 125);
    EXPECT_FALSE(resolved.sfmOptions.enableGuidedRematching);
}

TEST(AerialTriangulationWorkflowCoreTest, ReferenceSequencePreselectionMapsToSequenceWindowPairs)
{
    auto options = makeBaseOptions();
    options.genericPreselection = false;
    options.referencePreselection = true;
    options.referenceMode = QStringLiteral("sequence");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_TRUE(resolved.sfmOptions.autoRestrictKnownCameraPairs);
    EXPECT_FALSE(resolved.sfmOptions.useKnownCameraOverlapPairs);
    EXPECT_EQ(resolved.sfmOptions.knownCameraPairWindow, 4);
    EXPECT_EQ(resolved.sfmOptions.knownCameraSpatialNeighborCount, 0);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("pair_planning_mode")).toString(),
              QStringLiteral("sequence"));
}

TEST(AerialTriangulationWorkflowCoreTest, MatchPipelineOverridesFeatureAndMatcherNames)
{
    auto options = makeBaseOptions();
    options.featureAlgorithm = QStringLiteral("disk");
    options.matchAlgorithm = QStringLiteral("lightglue");
    options.matchPipeline = QStringLiteral("sift-bf-l2");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.sfmOptions.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(resolved.sfmOptions.matchAlgorithm, QStringLiteral("bf_l2"));
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("match_pipeline")).toString(),
              QStringLiteral("sift-bf-l2"));
}
