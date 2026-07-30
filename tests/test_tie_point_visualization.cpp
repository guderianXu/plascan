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
using xjw::gui::tie_points::ScalarRange;
using xjw::gui::tie_points::elevationColor;
using xjw::gui::tie_points::imageCountColor;
using xjw::gui::tie_points::inferSidecarPath;
using xjw::gui::tie_points::loadImageCountMetadata;
using xjw::gui::tie_points::pointSizeForMode;
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

} // namespace
