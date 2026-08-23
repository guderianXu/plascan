#include "TerrainProductManifest.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QTemporaryDir>

using xjw::TerrainProductManifest;
using xjw::TerrainProductRecord;

namespace
{

TerrainProductRecord makeDemRecord(const QString &id, const QString &demName)
{
    TerrainProductRecord record;
    record.productId = id;
    record.productType = QStringLiteral("dem");
    record.createdAt = QStringLiteral("2026-06-19T00:00:00Z");
    record.demPath = QStringLiteral("E:/terrain/%1").arg(demName);
    record.domPath = QStringLiteral("E:/terrain/dom.png");
    record.errorPath = QStringLiteral("E:/terrain/error.tif");
    record.countPath = QStringLiteral("E:/terrain/count.tif");
    record.confidencePath = QStringLiteral("E:/terrain/confidence.tif");
    record.coveragePath = QStringLiteral("E:/terrain/coverage.tif");
    record.previewPath = QStringLiteral("E:/terrain/depth_map.png");
    record.projection = QStringLiteral("LOCAL_CS[\"PlaScan\"]");
    record.gridResolution = 0.25;
    record.gridWidth = 640;
    record.gridHeight = 480;
    record.aggregation = QStringLiteral("confidence_weighted");
    return record;
}

} // namespace

TEST(TerrainProductManifest, SavesLoadsAndPreservesQualityRasterPaths)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString manifestPath = QDir(tempDir.path()).filePath(QStringLiteral("terrain_products.json"));

    TerrainProductManifest manifest;
    manifest.upsertRecord(makeDemRecord(QStringLiteral("dem-a"), QStringLiteral("dem_002.tif")));

    QString error;
    ASSERT_TRUE(manifest.saveAtomic(manifestPath, &error)) << error.toStdString();

    TerrainProductManifest loaded;
    ASSERT_TRUE(loaded.load(manifestPath, &error)) << error.toStdString();
    ASSERT_EQ(loaded.records().size(), 1);

    const TerrainProductRecord record = loaded.records().front();
    EXPECT_EQ(record.productId, QStringLiteral("dem-a"));
    EXPECT_EQ(record.demPath, QStringLiteral("E:/terrain/dem_002.tif"));
    EXPECT_EQ(record.domPath, QStringLiteral("E:/terrain/dom.png"));
    EXPECT_EQ(record.errorPath, QStringLiteral("E:/terrain/error.tif"));
    EXPECT_EQ(record.countPath, QStringLiteral("E:/terrain/count.tif"));
    EXPECT_EQ(record.confidencePath, QStringLiteral("E:/terrain/confidence.tif"));
    EXPECT_EQ(record.coveragePath, QStringLiteral("E:/terrain/coverage.tif"));
    EXPECT_EQ(record.projection, QStringLiteral("LOCAL_CS[\"PlaScan\"]"));
    EXPECT_DOUBLE_EQ(record.gridResolution, 0.25);
    EXPECT_EQ(record.aggregation, QStringLiteral("confidence_weighted"));
}

TEST(TerrainProductManifest, UpsertsRecordsAndSortsByNaturalPrimaryFileName)
{
    TerrainProductManifest manifest;
    manifest.upsertRecord(makeDemRecord(QStringLiteral("dem-10"), QStringLiteral("dem_A_10.tif")));
    manifest.upsertRecord(makeDemRecord(QStringLiteral("dem-2"), QStringLiteral("DEM_a_2.tif")));

    TerrainProductRecord replacement = makeDemRecord(QStringLiteral("dem-10"), QStringLiteral("dem_A_10.tif"));
    replacement.gridResolution = 0.5;
    manifest.upsertRecord(replacement);

    ASSERT_EQ(manifest.records().size(), 2);
    const auto sorted = manifest.recordsSortedByPrimaryPath();
    ASSERT_EQ(sorted.size(), 2);
    EXPECT_EQ(sorted[0].productId, QStringLiteral("dem-2"));
    EXPECT_EQ(sorted[1].productId, QStringLiteral("dem-10"));
    EXPECT_EQ(sorted[1].gridResolution, 0.5);
}

TEST(TerrainProductManifest, ProducesProjectMetadataFieldsForGuiConsumption)
{
    const TerrainProductRecord record = makeDemRecord(QStringLiteral("dem-a"), QStringLiteral("dem_002.tif"));
    const QJsonObject json = record.toJson();

    EXPECT_EQ(json.value(QStringLiteral("dem_path")).toString(), record.demPath);
    EXPECT_EQ(json.value(QStringLiteral("dom_path")).toString(), record.domPath);
    EXPECT_EQ(json.value(QStringLiteral("error_path")).toString(), record.errorPath);
    EXPECT_EQ(json.value(QStringLiteral("count_path")).toString(), record.countPath);
    EXPECT_EQ(json.value(QStringLiteral("confidence_path")).toString(), record.confidencePath);
    EXPECT_EQ(json.value(QStringLiteral("coverage_path")).toString(), record.coveragePath);
    EXPECT_EQ(json.value(QStringLiteral("projection")).toString(), record.projection);
    EXPECT_DOUBLE_EQ(json.value(QStringLiteral("grid_resolution")).toDouble(), record.gridResolution);
    EXPECT_EQ(json.value(QStringLiteral("aggregation")).toString(), record.aggregation);
}
