#include "GeometryVerifyStage.h"
#include "MatchPhotosTask.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QTemporaryDir>

TEST(MatchPhotosTaskTest, PlanOnlyUsesUnifiedAlgorithmWithoutWritingIntermediateFiles)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString image0 = QDir(tempDir.path()).filePath(QStringLiteral("a.png"));
    const QString image1 = QDir(tempDir.path()).filePath(QStringLiteral("b.png"));
    xjw::matchphotos::MatchPhotosContext context;
    context.workingDirectory = tempDir.path();
    context.matchDirectory = QDir(tempDir.path()).filePath(QStringLiteral("image_matches"));
    context.pairInput.images = {image0, image1};
    context.pairInput.manualPairKeys = {
        xjw::matchphotos::makePairKey(image0, image1)};

    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = true;
    options.algorithmId = QStringLiteral("sift_lightglue");
    options.pairPolicy.mode = xjw::matchphotos::PairSelectionMode::ManualOnly;

    const auto result = xjw::matchphotos::MatchPhotosTask(options).run(context);

    EXPECT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_TRUE(result.algorithmPlan.valid);
    EXPECT_EQ(result.algorithmPlan.algorithmId, QStringLiteral("sift_lightglue"));
    EXPECT_EQ(result.pairSelection.candidates.size(), 1u);
    EXPECT_TRUE(result.features.empty());
    EXPECT_TRUE(result.matches.empty());
    EXPECT_TRUE(result.imageMatchFiles.empty());
    EXPECT_FALSE(QDir(context.matchDirectory).exists());
}

TEST(MatchPhotosTaskTest, CpuSelectionFailsBeforeReadingImages)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.device = xjw::matchphotos::ComputeDevice::Cpu;
    options.planOnly = true;

    xjw::matchphotos::MatchPhotosContext context;
    context.pairInput.images = {QStringLiteral("a.png"), QStringLiteral("b.png")};

    const auto result = xjw::matchphotos::MatchPhotosTask(options).run(context);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("CUDA")));
    ASSERT_FALSE(result.stages.empty());
    EXPECT_EQ(result.stages.front().stageId, QStringLiteral("algorithm_selection"));
    EXPECT_EQ(result.stages.front().status,
              xjw::matchphotos::MatchPhotosStageStatus::Failed);
}

TEST(MatchPhotosGeometryGateTest, RequiresStrongSupportForSmallInlierSets)
{
    EXPECT_FALSE(xjw::matchphotos::passesGeometryQualityGate(100, 20, 20));
    EXPECT_TRUE(xjw::matchphotos::passesGeometryQualityGate(25, 20, 20));
    EXPECT_TRUE(xjw::matchphotos::passesGeometryQualityGate(200, 64, 20));
    EXPECT_FALSE(xjw::matchphotos::passesGeometryQualityGate(100, 19, 20));
}
