#include "TiePointVisualization.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <gtest/gtest.h>

namespace
{

using xjw::gui::tie_points::ImageCountMetadata;
using xjw::gui::tie_points::PrunePreviewQuery;
using xjw::gui::tie_points::QualityCriterion;
using xjw::gui::tie_points::QualityMetadata;
using xjw::gui::tie_points::ScalarRange;
using xjw::gui::tie_points::elevationColor;
using xjw::gui::tie_points::imageCountColor;
using xjw::gui::tie_points::inferSidecarPath;
using xjw::gui::tie_points::loadImageCountMetadata;
using xjw::gui::tie_points::loadQualityMetadata;
using xjw::gui::tie_points::pointSizeForMode;
using xjw::gui::tie_points::queryPruneCandidates;
using xjw::gui::tie_points::scalarRampColor;

TEST(TiePointVisualizationTest, ElevationRunsFromBlueLowToRedHigh)
{
    const ScalarRange range{-5.0, -1.0};
    const QColor low = elevationColor(-5.0, range);
    const QColor middle = elevationColor(-3.0, range);
    const QColor high = elevationColor(-1.0, range);

    EXPECT_GT(low.blue(), low.red());
    EXPECT_GT(middle.green(), middle.red());
    EXPECT_GT(high.red(), high.blue());
}

TEST(TiePointVisualizationTest, ImageCountRunsFromRedLowToBlueHigh)
{
    const ScalarRange range{2.0, 10.0};
    const QColor low = imageCountColor(2, range);
    const QColor high = imageCountColor(10, range);

    EXPECT_GT(low.red(), low.blue());
    EXPECT_GT(high.blue(), high.red());
}

TEST(TiePointVisualizationTest, ConstantRangeUsesStableMiddleColor)
{
    const ScalarRange range{4.0, 4.0};
    EXPECT_EQ(elevationColor(4.0, range), elevationColor(9.0, range));
    EXPECT_EQ(imageCountColor(4, range), imageCountColor(9, range));
}

TEST(TiePointVisualizationTest, ScalarRampStaysSaturatedAndHighContrast)
{
    for (int step = 0; step <= 20; ++step)
    {
        const QColor color = scalarRampColor(static_cast<double>(step) / 20.0);
        EXPECT_GE(color.hsvSaturation(), 190);
        EXPECT_LE(color.value(), 242);
    }
}

TEST(TiePointVisualizationTest, ScalarModesUseLargerPointsThanRgbMode)
{
    using xjw::gui::tie_points::ColorMode;

    EXPECT_FLOAT_EQ(pointSizeForMode(ColorMode::Color), 1.8f);
    EXPECT_GE(pointSizeForMode(ColorMode::Elevation), 3.0f);
    EXPECT_GE(pointSizeForMode(ColorMode::ImageCount), 3.0f);
}

TEST(TiePointVisualizationTest, LoadsImageCountsInPointOrder)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString sidecarPath = directory.filePath(QStringLiteral("sparse_cloud_points.json"));
    QFile sidecar(sidecarPath);
    ASSERT_TRUE(sidecar.open(QIODevice::WriteOnly));
    const QJsonArray points{
        QJsonObject{{QStringLiteral("track_len"), 2}},
        QJsonObject{{QStringLiteral("track_len"), 7}},
        QJsonObject{{QStringLiteral("track_len"), 4}}
    };
    sidecar.write(QJsonDocument(QJsonObject{{QStringLiteral("points"), points}})
                      .toJson(QJsonDocument::Compact));
    sidecar.close();

    const ImageCountMetadata metadata = loadImageCountMetadata(sidecarPath);
    ASSERT_TRUE(metadata.errorMessage.isEmpty());
    EXPECT_TRUE(metadata.isValidFor(3));
    EXPECT_EQ(metadata.counts, QVector<int>({2, 7, 4}));
}

TEST(TiePointVisualizationTest, InfersMetashapeStyleSparseSidecarNames)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString sidecarPath = directory.filePath(QStringLiteral("sfm_sparse_points.json"));
    QFile sidecar(sidecarPath);
    ASSERT_TRUE(sidecar.open(QIODevice::WriteOnly));
    sidecar.write("{}");
    sidecar.close();

    EXPECT_EQ(inferSidecarPath(directory.filePath(QStringLiteral("sfm_sparse.ply"))),
              sidecarPath);
}

TEST(TiePointVisualizationTest, LoadsAllQualityFieldsInOriginalPointOrder)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("quality.json"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    const QJsonArray points{
        QJsonObject{{QStringLiteral("rms_reproj_px"), 0.4},
                    {QStringLiteral("reconstruction_uncertainty"), 5.0},
                    {QStringLiteral("track_len"), 8},
                    {QStringLiteral("projection_accuracy"), 1.0},
                    {QStringLiteral("min_tri_angle_deg"), 6.0}},
        QJsonObject{{QStringLiteral("rms_reproj_px"), 2.5},
                    {QStringLiteral("reconstruction_uncertainty"), 15.0},
                    {QStringLiteral("track_len"), 2},
                    {QStringLiteral("projection_accuracy"), 3.0},
                    {QStringLiteral("min_tri_angle_deg"), 1.5}},
        QJsonObject{{QStringLiteral("rms_reproj_px"), 1.2},
                    {QStringLiteral("reconstruction_uncertainty"), 7.0},
                    {QStringLiteral("track_len"), 4},
                    {QStringLiteral("projection_accuracy"), 1.5},
                    {QStringLiteral("min_tri_angle_deg"), 3.0}}};
    file.write(QJsonDocument(QJsonObject{{QStringLiteral("points"), points}})
                   .toJson(QJsonDocument::Compact));
    file.close();

    const QualityMetadata metadata = loadQualityMetadata(path);

    ASSERT_TRUE(metadata.errorMessage.isEmpty());
    ASSERT_TRUE(metadata.isValidFor(3));
    EXPECT_EQ(metadata.reprojectionErrors, QVector<double>({0.4, 2.5, 1.2}));
    EXPECT_EQ(metadata.reconstructionUncertainties,
              QVector<double>({5.0, 15.0, 7.0}));
    EXPECT_EQ(metadata.imageCounts, QVector<int>({8, 2, 4}));
    EXPECT_EQ(metadata.projectionAccuracies,
              QVector<double>({1.0, 3.0, 1.5}));
    EXPECT_EQ(metadata.minimumTriangulationAngles,
              QVector<double>({6.0, 1.5, 3.0}));
    EXPECT_DOUBLE_EQ(metadata.reprojectionErrorRange.minimum, 0.4);
    EXPECT_DOUBLE_EQ(metadata.reprojectionErrorRange.maximum, 2.5);
    EXPECT_DOUBLE_EQ(metadata.imageCountRange.minimum, 2.0);
    EXPECT_DOUBLE_EQ(metadata.imageCountRange.maximum, 8.0);
    EXPECT_DOUBLE_EQ(metadata.reconstructionUncertaintyRange.maximum, 15.0);
    EXPECT_DOUBLE_EQ(metadata.projectionAccuracyRange.maximum, 3.0);
}

TEST(TiePointVisualizationTest, KeepsAvailableCriteriaWhenOneFieldIsMissing)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("partial.json"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    const QJsonArray points{
        QJsonObject{{QStringLiteral("rms_reproj_px"), 0.4},
                    {QStringLiteral("track_len"), 8}},
        QJsonObject{{QStringLiteral("rms_reproj_px"), 2.5},
                    {QStringLiteral("track_len"), 2}}};
    file.write(QJsonDocument(QJsonObject{{QStringLiteral("points"), points}})
                   .toJson(QJsonDocument::Compact));
    file.close();

    const QualityMetadata metadata = loadQualityMetadata(path);

    EXPECT_TRUE(metadata.hasCriterion(QualityCriterion::ReprojectionError, 2));
    EXPECT_TRUE(metadata.hasCriterion(QualityCriterion::ImageCount, 2));
    EXPECT_FALSE(metadata.hasCriterion(
        QualityCriterion::MinimumTriangulationAngle, 2));
    EXPECT_TRUE(metadata.minimumTriangulationAngles.isEmpty());
    const auto candidates = queryPruneCandidates(
        metadata, {QualityCriterion::ReprojectionError, 1.0}, 2);
    ASSERT_TRUE(candidates.succeeded());
    EXPECT_EQ(candidates.indices, (std::vector<std::uint32_t>{1U}));
}

TEST(TiePointVisualizationTest, MissingImageCountKeepsReprojectionPreviewAvailable)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("without_track_len.json"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    const QJsonArray points{
        QJsonObject{{QStringLiteral("rms_reproj_px"), 0.4},
                    {QStringLiteral("min_tri_angle_deg"), 6.0}},
        QJsonObject{{QStringLiteral("rms_reproj_px"), 2.5},
                    {QStringLiteral("min_tri_angle_deg"), 1.5}}};
    file.write(QJsonDocument(QJsonObject{{QStringLiteral("points"), points}})
                   .toJson(QJsonDocument::Compact));
    file.close();

    const QualityMetadata metadata = loadQualityMetadata(path);
    EXPECT_TRUE(metadata.hasCriterion(QualityCriterion::ReprojectionError, 2));
    EXPECT_FALSE(metadata.hasCriterion(QualityCriterion::ImageCount, 2));
    EXPECT_TRUE(metadata.hasCriterion(
        QualityCriterion::MinimumTriangulationAngle, 2));
    const auto candidates = queryPruneCandidates(
        metadata, {QualityCriterion::ReprojectionError, 1.0}, 2);
    ASSERT_TRUE(candidates.succeeded());
    EXPECT_EQ(candidates.indices, (std::vector<std::uint32_t>{1U}));
}

TEST(TiePointVisualizationTest, ZeroPlaceholderAnglesAreNotOfferedForCleaning)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("zero_angles.json"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    const QJsonArray points{
        QJsonObject{{QStringLiteral("rms_reproj_px"), 0.4},
                    {QStringLiteral("track_len"), 3},
                    {QStringLiteral("min_tri_angle_deg"), 0.0}},
        QJsonObject{{QStringLiteral("rms_reproj_px"), 0.8},
                    {QStringLiteral("track_len"), 4},
                    {QStringLiteral("min_tri_angle_deg"), 0.0}}};
    file.write(QJsonDocument(QJsonObject{{QStringLiteral("points"), points}})
                   .toJson(QJsonDocument::Compact));
    file.close();

    const QualityMetadata metadata = loadQualityMetadata(path);

    EXPECT_TRUE(metadata.hasCriterion(QualityCriterion::ReprojectionError, 2));
    EXPECT_TRUE(metadata.hasCriterion(QualityCriterion::ImageCount, 2));
    EXPECT_FALSE(metadata.hasCriterion(
        QualityCriterion::MinimumTriangulationAngle, 2));
}

TEST(TiePointVisualizationTest, QueriesPruneCandidatesWithCriterionDirection)
{
    QualityMetadata metadata;
    metadata.sourcePointCount = 4;
    metadata.reprojectionErrors = {0.4, 2.5, 1.2, 3.0};
    metadata.reconstructionUncertainties = {5.0, 15.0, 7.0, 20.0};
    metadata.imageCounts = {8, 2, 4, 1};
    metadata.projectionAccuracies = {1.0, 3.0, 1.5, 4.0};
    metadata.minimumTriangulationAngles = {6.0, 1.5, 3.0, 0.5};

    const auto reprojection = queryPruneCandidates(
        metadata, {QualityCriterion::ReprojectionError, 1.5}, 4);
    const auto image_count = queryPruneCandidates(
        metadata, {QualityCriterion::ImageCount, 3.0}, 4);
    const auto uncertainty = queryPruneCandidates(
        metadata, {QualityCriterion::ReconstructionUncertainty, 10.0}, 4);
    const auto projection_accuracy = queryPruneCandidates(
        metadata, {QualityCriterion::ProjectionAccuracy, 2.0}, 4);
    const auto angle = queryPruneCandidates(
        metadata, {QualityCriterion::MinimumTriangulationAngle, 2.0}, 4);

    ASSERT_TRUE(reprojection.succeeded());
    EXPECT_EQ(reprojection.candidateCount, 2);
    EXPECT_EQ(reprojection.indices, (std::vector<std::uint32_t>{1U, 3U}));
    ASSERT_TRUE(image_count.succeeded());
    EXPECT_EQ(image_count.candidateCount, 2);
    EXPECT_EQ(image_count.indices, (std::vector<std::uint32_t>{1U, 3U}));
    ASSERT_TRUE(uncertainty.succeeded());
    EXPECT_EQ(uncertainty.indices, (std::vector<std::uint32_t>{1U, 3U}));
    ASSERT_TRUE(projection_accuracy.succeeded());
    EXPECT_EQ(projection_accuracy.indices, (std::vector<std::uint32_t>{1U, 3U}));
    ASSERT_TRUE(angle.succeeded());
    EXPECT_EQ(angle.candidateCount, 2);
    EXPECT_EQ(angle.indices, (std::vector<std::uint32_t>{1U, 3U}));
}

TEST(TiePointVisualizationTest, BoundsLargeCandidateResultsWithStableEvenSampling)
{
    constexpr qsizetype point_count = 250'001;
    QualityMetadata metadata;
    metadata.sourcePointCount = point_count;
    metadata.reprojectionErrors.fill(2.0, point_count);

    const auto candidates = queryPruneCandidates(
        metadata,
        {QualityCriterion::ReprojectionError, 1.0},
        point_count,
        nullptr,
        5);

    ASSERT_TRUE(candidates.succeeded());
    EXPECT_EQ(candidates.candidateCount, point_count);
    EXPECT_EQ(candidates.indices,
              (std::vector<std::uint32_t>{0U, 62'500U, 125'000U,
                                          187'500U, 250'000U}));

    const auto count_only = queryPruneCandidates(
        metadata,
        {QualityCriterion::ReprojectionError, 1.0},
        point_count,
        nullptr,
        0);
    ASSERT_TRUE(count_only.succeeded());
    EXPECT_EQ(count_only.candidateCount, point_count);
    EXPECT_TRUE(count_only.indices.empty());
}

TEST(TiePointVisualizationTest, ExcludesAlreadyStagedCandidates)
{
    QualityMetadata metadata;
    metadata.sourcePointCount = 5;
    metadata.reprojectionErrors = {0.5, 2.0, 3.0, 4.0, 5.0};
    const std::vector<std::uint32_t> excluded{2U, 4U};

    const auto candidates = queryPruneCandidates(
        metadata,
        {QualityCriterion::ReprojectionError, 1.0},
        5,
        nullptr,
        std::numeric_limits<std::size_t>::max(),
        &excluded);

    ASSERT_TRUE(candidates.succeeded());
    EXPECT_EQ(candidates.candidateCount, 2);
    EXPECT_EQ(candidates.indices, (std::vector<std::uint32_t>{1U, 3U}));
}

TEST(TiePointVisualizationTest, RejectsMismatchedOrCancelledCandidateQueries)
{
    QualityMetadata metadata;
    metadata.sourcePointCount = 2;
    metadata.reprojectionErrors = {0.4, 2.5};
    EXPECT_FALSE(queryPruneCandidates(
        metadata, {QualityCriterion::ReprojectionError, 1.0}, -9).succeeded());
    EXPECT_FALSE(queryPruneCandidates(
        metadata, {QualityCriterion::ReprojectionError, 1.0}, 3).succeeded());
    EXPECT_FALSE(queryPruneCandidates(
        metadata, {QualityCriterion::ReprojectionError, -1.0}, 2).succeeded());
    EXPECT_FALSE(queryPruneCandidates(
        metadata, {QualityCriterion::ImageCount, 2.5}, 2).succeeded());

    std::atomic_bool cancelled{true};
    EXPECT_FALSE(queryPruneCandidates(
        metadata,
        {QualityCriterion::ReprojectionError, 1.0},
        2,
        &cancelled).succeeded());
}

} // namespace
