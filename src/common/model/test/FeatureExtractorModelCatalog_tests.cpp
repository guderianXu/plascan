#include "model/FeatureExtractorModelCatalog.h"

#include <gtest/gtest.h>

namespace model = xjw::common::model;

TEST(FeatureExtractorModelCatalogTest, ListsCudaThenCpuThenGenericCandidates)
{
    const QStringList candidates = model::featureExtractorModelCandidates(QStringLiteral(" DISK "), true);

    ASSERT_EQ(candidates.size(), 10);
    EXPECT_EQ(candidates.first(), QStringLiteral("disk_extractor_cuda_8192.torchscript"));
    EXPECT_EQ(candidates.at(4), QStringLiteral("disk_extractor_cpu_8192.torchscript"));
    EXPECT_EQ(candidates.last(), QStringLiteral("disk_extractor.pt"));
}

TEST(FeatureExtractorModelCatalogTest, CpuCandidatesExcludeCudaModels)
{
    const QStringList candidates = model::featureExtractorModelCandidates(QStringLiteral("ALIKED"), false);

    ASSERT_EQ(candidates.size(), 4);
    EXPECT_EQ(candidates.first(), QStringLiteral("aliked_extractor_cpu_480.torchscript"));
    EXPECT_FALSE(candidates.join(QLatin1Char('|')).contains(QStringLiteral("_cuda")));
    EXPECT_TRUE(model::featureExtractorModelCandidates(QStringLiteral("sift"), false).isEmpty());
}

TEST(FeatureExtractorModelCatalogTest, RecognizesOnlyManagedExtractorModelNames)
{
    EXPECT_TRUE(model::isManagedFeatureExtractorModelPath(
        QStringLiteral("C:/models/SUPERPOINT_EXTRACTOR_CPU.PT")));
    EXPECT_TRUE(model::isManagedFeatureExtractorModelPath(QStringLiteral("disk_extractor.pt")));
    EXPECT_TRUE(model::isManagedFeatureExtractorModelPath(QStringLiteral("aliked_extractor_cuda_480.torchscript")));
    EXPECT_FALSE(model::isManagedFeatureExtractorModelPath(QStringLiteral("custom_model.pt")));
}
