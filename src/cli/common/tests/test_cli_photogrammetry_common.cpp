#include "cli_photogrammetry_common.h"

#include <gtest/gtest.h>

namespace
{

using xjw::cli::parsePhotogrammetryListLine;

TEST(CliPhotogrammetryCommonTest, ParsesShellAndCsvRows)
{
    QStringList parts;
    QString error;

    EXPECT_TRUE(parsePhotogrammetryListLine(
        QStringLiteral("\"image one.tif\" camera.tsai"), &parts, &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("image one.tif"),
                                  QStringLiteral("camera.tsai")}));

    EXPECT_TRUE(parsePhotogrammetryListLine(
        QStringLiteral("\"image,one.tif\",\"camera\"\"one.tsai\""), &parts, &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("image,one.tif"),
                                  QStringLiteral("camera\"one.tsai")}));
}

TEST(CliPhotogrammetryCommonTest, PreservesWindowsPathSeparators)
{
    QStringList parts;
    QString error;

    EXPECT_TRUE(parsePhotogrammetryListLine(
        QStringLiteral("E:\\code\\images\\image.tif E:\\code\\cameras\\camera.tsai"),
        &parts,
        &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("E:\\code\\images\\image.tif"),
                                  QStringLiteral("E:\\code\\cameras\\camera.tsai")}));

    EXPECT_TRUE(parsePhotogrammetryListLine(
        QStringLiteral("\"E:\\data set\\image one.tif\" \"\\\\server\\camera set\\camera.tsai\""),
        &parts,
        &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("E:\\data set\\image one.tif"),
                                  QStringLiteral("\\\\server\\camera set\\camera.tsai")}));
}

TEST(CliPhotogrammetryCommonTest, PreservesTrailingCsvCellAndReportsMalformedRows)
{
    QStringList parts;
    QString error;

    EXPECT_TRUE(parsePhotogrammetryListLine(QStringLiteral("image.tif,"), &parts, &error));
    EXPECT_EQ(parts, (QStringList{QStringLiteral("image.tif"), QString()}));

    EXPECT_FALSE(parsePhotogrammetryListLine(QStringLiteral("\"image.tif"), &parts, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("引号未闭合")));

    EXPECT_FALSE(parsePhotogrammetryListLine(QStringLiteral("image.tif\\"), &parts, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("行尾转义")));
}

TEST(CliPhotogrammetryCommonTest, SerializesProjectItemsAndReportPairsSeparately)
{
    xjw::cli::PhotogrammetryInputItem item;
    item.imagePath = QStringLiteral("image.tif");
    item.cameraPath = QStringLiteral("camera.tsai");
    item.hasCameraPath = true;
    const std::vector<xjw::cli::PhotogrammetryInputItem> items{item};

    const QJsonObject projectItem = xjw::cli::inputItemsToJson(items).first().toObject();
    EXPECT_EQ(projectItem.value(QStringLiteral("path")).toString(), item.imagePath);
    EXPECT_EQ(projectItem.value(QStringLiteral("camera_path")).toString(), item.cameraPath);

    const QJsonObject reportPair = xjw::cli::inputPairsToJson(items).first().toObject();
    EXPECT_EQ(reportPair.value(QStringLiteral("image")).toString(), item.imagePath);
    EXPECT_EQ(reportPair.value(QStringLiteral("camera")).toString(), item.cameraPath);
}

} // namespace
