#include "MatchPhotosTask.h"
#include "FeatureFileIO.h"
#include "FeatureStage.h"
#include "GeometryVerifyStage.h"
#include "MatchPhotosAlgorithmSelector.h"
#include "MatchingStage.h"
#include "TrackBuildStage.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QTemporaryDir>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{

QStringList makeImages(int count)
{
    QStringList images;
    const QDir dir(QDir::tempPath());
    for (int i = 0; i < count; ++i)
    {
        images.append(dir.filePath(QStringLiteral("plascan_matchphotos_task_%1.png").arg(i)));
    }
    return images;
}

QString writeSyntheticSiftImage(const QString &dirPath, const QString &name, int shift)
{
    cv::Mat image(160, 160, CV_8UC1, cv::Scalar(20));
    cv::rectangle(image, cv::Rect(25 + shift, 25, 50, 50), cv::Scalar(230), cv::FILLED);
    cv::circle(image, cv::Point(110 + shift, 105), 28, cv::Scalar(180), cv::FILLED);
    cv::line(image, cv::Point(15, 145 - shift), cv::Point(145, 15 + shift), cv::Scalar(255), 3);

    const QString path = QDir(dirPath).filePath(name);
    EXPECT_TRUE(cv::imwrite(path.toStdString(), image));
    return path;
}

QString readProjectSourceFile(const QString &relativePath)
{
    QStringList roots;
#ifdef PLASCAN_SOURCE_DIR
    roots.append(QStringLiteral(PLASCAN_SOURCE_DIR));
#endif
    roots.append(QDir::currentPath());
    roots.append(QCoreApplication::applicationDirPath());

    for (const QString &root : roots)
    {
        QDir dir(root);
        for (int depth = 0; depth < 8; ++depth)
        {
            const QString path = dir.filePath(relativePath);
            if (QFileInfo::exists(path))
            {
                QFile file(path);
                if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
                {
                    return QString();
                }
                QTextStream in(&file);
                return in.readAll();
            }
            if (!dir.cdUp())
            {
                break;
            }
        }
    }
    return QString();
}

} // namespace

TEST(MatchPhotosTaskTest, RunsPairSelectionAndReportsStageSkeleton)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.pairPolicy.exhaustiveMaxImages = 3;

    xjw::matchphotos::MatchPhotosContext context;
    context.pairInput.images = makeImages(5);

    const xjw::matchphotos::MatchPhotosTask task(options);
    const xjw::matchphotos::MatchPhotosResult result = task.run(context);

    EXPECT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_EQ(result.algorithmPlan.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(result.algorithmPlan.matcherAlgorithm, QStringLiteral("lightglue"));
    EXPECT_TRUE(result.algorithmPlan.rotationRobust);
    EXPECT_TRUE(result.pairSelection.restrictPairs);
    ASSERT_EQ(result.stages.size(), 7);
    EXPECT_EQ(result.stages.front().status, xjw::matchphotos::MatchPhotosStageStatus::Completed);
    EXPECT_EQ(result.stages.front().stageId, QStringLiteral("algorithm_selection"));
    EXPECT_EQ(result.stages.at(1).stageId, QStringLiteral("pair_selection"));
    EXPECT_EQ(result.stages.at(2).status, xjw::matchphotos::MatchPhotosStageStatus::Skipped);
}

TEST(MatchPhotosTaskTest, FeatureStageWritesSiftFilesForSyntheticImages)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString image0 = writeSyntheticSiftImage(tempDir.path(), QStringLiteral("image0.png"), 0);
    const QString image1 = writeSyntheticSiftImage(tempDir.path(), QStringLiteral("image1.png"), 5);

    xjw::matchphotos::MatchPhotosOptions options;
    options.planOnly = false;
    options.device = xjw::matchphotos::ComputeDevice::Cpu;
    options.maxImageDim = 160;
    options.maxKeypoints = 128;
    options.reuseExistingFeatures = false;

    xjw::matchphotos::MatchPhotosContext context;
    context.featureDirectory = QDir(tempDir.path()).filePath(QStringLiteral("ip"));
    context.pairInput.images = QStringList{image0, image1};

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);
    const xjw::matchphotos::FeatureStage stage;
    std::vector<xjw::matchphotos::MatchPhotosFeatureRecord> records;
    const xjw::matchphotos::MatchPhotosStageReport report =
        stage.run(context, options, plan, &records);

    EXPECT_EQ(report.status, xjw::matchphotos::MatchPhotosStageStatus::Completed)
        << report.message.toStdString();
    ASSERT_EQ(records.size(), 2u);
    for (const xjw::matchphotos::MatchPhotosFeatureRecord &record : records)
    {
        EXPECT_TRUE(QFile::exists(record.featurePath));
        EXPECT_EQ(FeatureFileIO::peekAlgorithm(record.featurePath), "sift");
        EXPECT_GT(FeatureFileIO::peekCount(record.featurePath), 0);
    }
}

TEST(MatchPhotosTaskTest, FeatureStageUsesTraditionalFeatureConfig)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/FeatureStage.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("TraditionalFeatureConfig config")));
    EXPECT_FALSE(source.contains(QStringLiteral("SuperPointConfig config")));
}

TEST(MatchPhotosTaskTest, MatchingStageUsesMemoryAwareLightGlueBudget)
{
    const QString source =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/MatchingStage.cpp"));
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(source.contains(QStringLiteral("resolveLightGlueKeypointBudget")));
    EXPECT_TRUE(source.contains(QStringLiteral("budgetFeatureDataForLightGlue")));
    EXPECT_TRUE(source.contains(QStringLiteral("remapLightGlueMatchResultToOriginal")));
    EXPECT_TRUE(source.contains(QStringLiteral("lightglue_keypoint_budget")));
}

TEST(MatchPhotosTaskTest, GeometryAndTrackStagesUseExistingCoreImplementations)
{
    const QString geometrySource =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/GeometryVerifyStage.cpp"));
    const QString trackSource =
        readProjectSourceFile(QStringLiteral("src/core/matchphototask/stages/TrackBuildStage.cpp"));
    ASSERT_FALSE(geometrySource.isEmpty());
    ASSERT_FALSE(trackSource.isEmpty());

    EXPECT_TRUE(geometrySource.contains(QStringLiteral("MatchGeometryFilter::filter")));
    EXPECT_FALSE(geometrySource.contains(QStringLiteral("几何验证阶段尚未接入")));
    EXPECT_TRUE(trackSource.contains(QStringLiteral("MultiViewTrackBuilder")));
    EXPECT_FALSE(trackSource.contains(QStringLiteral("轨迹构建阶段尚未接入")));
}
