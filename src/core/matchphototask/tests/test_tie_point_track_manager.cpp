#include "TiePointTrackManager.h"

#include "ImageMatchFile.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <array>
#include <memory>
#include <vector>

namespace
{

    xjw::image_matching::KeypointObservation observation(std::uint32_t featureId, float x, float y)
    {
        xjw::image_matching::KeypointObservation value;
        value.featureId = featureId;
        value.x = x;
        value.y = y;
        value.scale = 1.0f;
        value.response = 1.0f;
        return value;
    }

    xjw::matchphotos::MatchPhotosMatchRecord makeMatchRecord(const QString& image0Path,
                                                             const QString& image1Path,
                                                             bool passedGeometry,
                                                             const std::vector<std::array<float, 4>>& points)
    {
        auto pair = std::make_shared<xjw::image_matching::PairMatchData>();
        pair->image0 = xjw::image_matching::ImageMatchFile::identityForImage(image0Path, 200, 160);
        pair->image1 = xjw::image_matching::ImageMatchFile::identityForImage(image1Path, 200, 160);
        pair->algorithmId = QStringLiteral("sift_lightglue");
        pair->algorithmVersion = 1;
        pair->geometryPassed = passedGeometry;
        for (std::uint32_t index = 0; index < points.size(); ++index)
        {
            const auto& point = points[index];
            xjw::image_matching::PairCorrespondence edge;
            edge.observation0 = observation(index, point[0], point[1]);
            edge.observation1 = observation(index, point[2], point[3]);
            edge.confidence = 0.9f;
            edge.flags = passedGeometry ? xjw::image_matching::MatchRecordFlag::GeometryInlier
                                        : xjw::image_matching::MatchRecordFlag::None;
            pair->correspondences.push_back(edge);
        }

        xjw::matchphotos::MatchPhotosMatchRecord record;
        record.image0Path = image0Path;
        record.image1Path = image1Path;
        record.matchCount = static_cast<int>(points.size());
        record.geometricInlierCount = passedGeometry ? record.matchCount : 0;
        record.passedGeometry = passedGeometry;
        record.pairData = std::move(pair);
        return record;
    }

} // namespace

TEST(TiePointTrackManagerTest, BuildsFinalMultiviewTracksFromInMemoryPairData)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString image0 = QDir(tempDir.path()).filePath(QStringLiteral("image0.png"));
    const QString image1 = QDir(tempDir.path()).filePath(QStringLiteral("image1.png"));
    const QString image2 = QDir(tempDir.path()).filePath(QStringLiteral("image2.png"));

    xjw::matchphotos::MatchPhotosContext context;
    context.workingDirectory = tempDir.path();
    context.pairInput.images = {image0, image1, image2};

    xjw::matchphotos::MatchPhotosOptions options;
    options.enableGeometryVerification = true;
    options.excludeStationaryTiePoints = false;

    std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records;
    records.push_back(makeMatchRecord(image0, image1, true, {{10.0f, 10.0f, 12.0f, 11.0f}}));
    records.push_back(makeMatchRecord(image1, image2, true, {{12.0f, 11.0f, 14.0f, 12.0f}}));
    records.push_back(makeMatchRecord(image0, image2, false, {{90.0f, 90.0f, 92.0f, 92.0f}}));

    const xjw::matchphotos::TiePointTrackBuildResult result =
        xjw::matchphotos::TiePointTrackManager().build(context, options, &records);

    EXPECT_TRUE(result.success) << qPrintable(result.errorMessage);
    EXPECT_EQ(result.consumedPairCount, 2);
    EXPECT_EQ(result.skippedPairCount, 1);
    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().length(), 3u);
    EXPECT_TRUE(xjw::image_matching::hasFlag(records.front().pairData->correspondences.front().flags,
                                             xjw::image_matching::MatchRecordFlag::InTiePointTrack));
    ASSERT_TRUE(QFileInfo::exists(result.tiePointPath));

    QFile file(result.tiePointPath);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    const QJsonObject stored = QJsonDocument::fromJson(file.readAll()).object();
    EXPECT_EQ(stored.value(QStringLiteral("format")).toString(), QStringLiteral("plascan_tie_points"));
    EXPECT_EQ(stored.value(QStringLiteral("format_version")).toInt(), 3);
    EXPECT_EQ(stored.value(QStringLiteral("observation_fields")).toArray(),
              QJsonArray({QStringLiteral("image_id"),
                          QStringLiteral("feature_idx"),
                          QStringLiteral("x"),
                          QStringLiteral("y"),
                          QStringLiteral("scale")}));
    EXPECT_EQ(stored.value(QStringLiteral("track_count")).toInt(), 1);
    const QJsonObject summary = stored.value(QStringLiteral("summary")).toObject();
    EXPECT_EQ(summary.value(QStringLiteral("strategy")).toString(), QStringLiteral("align_photos_reference"));
    EXPECT_EQ(summary.value(QStringLiteral("generated_tracks")).toInt(), 1);
    EXPECT_EQ(summary.value(QStringLiteral("pruned_by_spatial_selection")).toInt(), 0);
    const QJsonObject storedTrack = stored.value(QStringLiteral("tracks")).toArray().first().toObject();
    const QJsonArray storedObservations = storedTrack.value(QStringLiteral("observations")).toArray();
    EXPECT_EQ(storedObservations.size(), 3);
    for (const QJsonValue& value : storedObservations)
    {
        ASSERT_TRUE(value.isArray());
        const QJsonArray observation = value.toArray();
        ASSERT_EQ(observation.size(), 5);
        EXPECT_DOUBLE_EQ(observation.at(4).toDouble(), 1.0);
    }
    const QJsonArray directEdges = storedTrack.value(QStringLiteral("direct_edges")).toArray();
    ASSERT_EQ(directEdges.size(), 2);
    EXPECT_EQ(directEdges.at(0).toArray(), QJsonArray({0, 1}));
    EXPECT_EQ(directEdges.at(1).toArray(), QJsonArray({1, 2}));
}

TEST(TiePointTrackManagerTest, StationaryTracksAreRejectedWithoutFeatureFiles)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString image0 = QDir(tempDir.path()).filePath(QStringLiteral("image0.png"));
    const QString image1 = QDir(tempDir.path()).filePath(QStringLiteral("image1.png"));
    xjw::matchphotos::MatchPhotosContext context;
    context.workingDirectory = tempDir.path();
    context.pairInput.images = {image0, image1};

    xjw::matchphotos::MatchPhotosOptions options;
    options.enableGeometryVerification = true;
    options.excludeStationaryTiePoints = true;
    options.stationaryTiePointMaxPixelMotion = 1.0f;
    std::vector<xjw::matchphotos::MatchPhotosMatchRecord> records;
    records.push_back(makeMatchRecord(image0, image1, true, {{10.0f, 10.0f, 10.2f, 10.1f}}));

    const auto result = xjw::matchphotos::TiePointTrackManager().build(context, options, &records);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("连接点")));
    EXPECT_TRUE(result.tracks.empty());
    EXPECT_TRUE(result.tiePointPath.isEmpty());
}
