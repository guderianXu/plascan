#include "TiePointTrackManager.h"

#include "FeatureFileIO.h"
#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"

// Avoid Qt keyword macros rewriting LibTorch's slots() member name.
#ifdef slots
#undef slots
#define PLASCAN_MATCHPHOTOS_TEST_RESTORE_QT_SLOTS
#endif
#ifdef signals
#undef signals
#define PLASCAN_MATCHPHOTOS_TEST_RESTORE_QT_SIGNALS
#endif
#ifdef emit
#undef emit
#define PLASCAN_MATCHPHOTOS_TEST_RESTORE_QT_EMIT
#endif

#include "FeatureOutput.h"

#ifdef PLASCAN_MATCHPHOTOS_TEST_RESTORE_QT_SLOTS
#define slots Q_SLOTS
#undef PLASCAN_MATCHPHOTOS_TEST_RESTORE_QT_SLOTS
#endif
#ifdef PLASCAN_MATCHPHOTOS_TEST_RESTORE_QT_SIGNALS
#define signals Q_SIGNALS
#undef PLASCAN_MATCHPHOTOS_TEST_RESTORE_QT_SIGNALS
#endif
#ifdef PLASCAN_MATCHPHOTOS_TEST_RESTORE_QT_EMIT
#define emit Q_EMIT
#undef PLASCAN_MATCHPHOTOS_TEST_RESTORE_QT_EMIT
#endif

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

namespace
{

QString writeFeatureFile(const QString &dirPath,
                         const QString &fileName,
                         const QString &imageName,
                         const std::vector<cv::Point2f> &points)
{
    FeatureOutput output;
    output.imageWidth = 200;
    output.imageHeight = 160;
    output.descriptors = torch::ones({static_cast<int64_t>(points.size()), 128}, torch::kFloat32);
    output.scores.reserve(points.size());
    output.keypoints.reserve(points.size());
    for (const cv::Point2f &point : points)
    {
        cv::KeyPoint keypoint;
        keypoint.pt = point;
        keypoint.response = 1.0f;
        keypoint.size = 1.0f;
        output.keypoints.push_back(keypoint);
        output.scores.push_back(1.0f);
    }

    const QString path = QDir(dirPath).filePath(fileName);
    EXPECT_TRUE(FeatureFileIO::write(path, imageName, output, "sift")) << qPrintable(path);
    return path;
}

xjw::matchphotos::MatchPhotosMatchRecord makeMatchRecord(const QString &image0Path,
                                                         const QString &image1Path,
                                                         const QString &feature0Path,
                                                         const QString &feature1Path,
                                                         bool passedGeometry,
                                                         const std::vector<std::array<int, 2>> &inlierIndexPairs)
{
    xjw::matchphotos::MatchPhotosMatchRecord record;
    record.image0Path = image0Path;
    record.image1Path = image1Path;
    record.passedGeometry = passedGeometry;
    record.inlierIndexPairs = inlierIndexPairs;
    record.settings[QStringLiteral("feature0_path")] = feature0Path;
    record.settings[QStringLiteral("feature1_path")] = feature1Path;
    return record;
}

} // namespace

TEST(TiePointTrackManagerTest, BuildsFinalMultiviewTracksFromVerifiedPairMatches)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString image0 = QDir(tempDir.path()).filePath(QStringLiteral("image0.png"));
    const QString image1 = QDir(tempDir.path()).filePath(QStringLiteral("image1.png"));
    const QString image2 = QDir(tempDir.path()).filePath(QStringLiteral("image2.png"));

    const QString feature0 = writeFeatureFile(tempDir.path(),
                                              QStringLiteral("image0.sift"),
                                              QStringLiteral("image0.png"),
                                              {cv::Point2f(10.0f, 10.0f), cv::Point2f(90.0f, 90.0f)});
    const QString feature1 = writeFeatureFile(tempDir.path(),
                                              QStringLiteral("image1.sift"),
                                              QStringLiteral("image1.png"),
                                              {cv::Point2f(12.0f, 11.0f), cv::Point2f(91.0f, 91.0f)});
    const QString feature2 = writeFeatureFile(tempDir.path(),
                                              QStringLiteral("image2.sift"),
                                              QStringLiteral("image2.png"),
                                              {cv::Point2f(14.0f, 12.0f), cv::Point2f(92.0f, 92.0f)});

    xjw::matchphotos::MatchPhotosContext context;
    context.workingDirectory = tempDir.path();
    context.pairInput.images = QStringList{image0, image1, image2};

    xjw::matchphotos::MatchPhotosOptions options;
    options.enableGeometryVerification = true;
    options.excludeStationaryTiePoints = false;

    const std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records{
        makeMatchRecord(image0, image1, feature0, feature1, true, {{0, 0}}),
        makeMatchRecord(image1, image2, feature1, feature2, true, {{0, 0}}),
        makeMatchRecord(image0, image2, feature0, feature2, false, {{1, 1}}),
    };

    const xjw::matchphotos::TiePointTrackManager manager;
    const xjw::matchphotos::TiePointTrackBuildResult result =
        manager.build(context, options, records);

    EXPECT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_EQ(result.consumedPairCount, 2);
    EXPECT_EQ(result.skippedPairCount, 1);
    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().length(), 3u);
    EXPECT_EQ(result.trackSummary.value(QStringLiteral("tracks")).toInt(), 1);
    EXPECT_EQ(result.acceptedComponents, 1);
    ASSERT_FALSE(result.tiePointPath.isEmpty());
    ASSERT_TRUE(QFileInfo::exists(result.tiePointPath)) << qPrintable(result.tiePointPath);

    QFile file(result.tiePointPath);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    const QJsonObject stored = QJsonDocument::fromJson(file.readAll()).object();
    EXPECT_EQ(stored.value(QStringLiteral("format")).toString(), QStringLiteral("plascan_tie_points"));
    EXPECT_EQ(stored.value(QStringLiteral("track_count")).toInt(), 1);
    EXPECT_EQ(stored.value(QStringLiteral("summary")).toObject().value(QStringLiteral("tracks")).toInt(), 1);
    const QJsonArray tracks = stored.value(QStringLiteral("tracks")).toArray();
    ASSERT_EQ(tracks.size(), 1);
    EXPECT_EQ(tracks.at(0).toObject().value(QStringLiteral("observations")).toArray().size(), 3);
}

TEST(TiePointTrackManagerTest, FailsWhenNoFinalTiePointTracksAreBuilt)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString image0 = QDir(tempDir.path()).filePath(QStringLiteral("image0.png"));
    const QString image1 = QDir(tempDir.path()).filePath(QStringLiteral("image1.png"));

    const QString feature0 = writeFeatureFile(tempDir.path(),
                                              QStringLiteral("image0.sift"),
                                              QStringLiteral("image0.png"),
                                              {cv::Point2f(10.0f, 10.0f)});
    const QString feature1 = writeFeatureFile(tempDir.path(),
                                              QStringLiteral("image1.sift"),
                                              QStringLiteral("image1.png"),
                                              {cv::Point2f(10.2f, 10.1f)});

    xjw::matchphotos::MatchPhotosContext context;
    context.workingDirectory = tempDir.path();
    context.pairInput.images = QStringList{image0, image1};

    xjw::matchphotos::MatchPhotosOptions options;
    options.enableGeometryVerification = true;
    options.excludeStationaryTiePoints = true;
    options.stationaryTiePointMaxPixelMotion = 1.0f;

    const std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records{
        makeMatchRecord(image0, image1, feature0, feature1, true, {{0, 0}}),
    };

    const xjw::matchphotos::TiePointTrackManager manager;
    const xjw::matchphotos::TiePointTrackBuildResult result =
        manager.build(context, options, records);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("连接点")));
    EXPECT_TRUE(result.tracks.empty());
    EXPECT_TRUE(result.tiePointPath.isEmpty());
}
