#include "MetashapeCameraReferenceImporter.h"

#include <gtest/gtest.h>

#include <QByteArray>
#include <QSaveFile>
#include <QTemporaryDir>

namespace
{

using xjw::gui::reference_import::MetashapeCameraReferenceImportResult;
using xjw::gui::reference_import::importMetashapeCameraReferenceTxt;

const QByteArray kFullHeader =
    "# file\t WGS84_lat\t WGS84_lon\t WGS84_H\t roll\t pitch\t yaw\t time\t"
    "Std Dev n (m)\tStd Dev e (m)\tStd Dev u (m)\tStd Dev Hz (m)\t\n";

bool writeUtf8File(const QString &path, const QByteArray &content)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(content) != content.size())
    {
        return false;
    }
    return file.commit();
}

QByteArray validCameraRecord(const QByteArray &fileName)
{
    return fileName
        + "\t59.84507804\t31.46683755\t225.157\t1.81\t3.67\t-179.65\t"
          "2019.08.06 09:26:06.109095\t0.0131\t0.0095\t0.018\t0.0162\t \t\n";
}

TEST(MetashapeCameraReferenceImporter, ImportsCameraAndGnssWithBomAndTrailingEmptyColumns)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    const QString camera_path = directory.filePath(QStringLiteral("Cameras_WGS84.txt"));
    const QString gnss_path = directory.filePath(QStringLiteral("GNSS_offset.txt"));
    const QByteArray camera_content = QByteArray("\xEF\xBB\xBF", 3)
        + kFullHeader + validCameraRecord("IMG_001.JPG");
    ASSERT_TRUE(writeUtf8File(camera_path, camera_content));
    ASSERT_TRUE(writeUtf8File(gnss_path,
                              QByteArray("\xEF\xBB\xBF", 3)
                                  + " Z = -0.032 \nY=-0.1815\n X = 0.3682\n"));

    MetashapeCameraReferenceImportResult result;
    QString error;
    ASSERT_TRUE(importMetashapeCameraReferenceTxt(camera_path, gnss_path, &result, &error))
        << error.toStdString();
    EXPECT_TRUE(error.isEmpty());
    ASSERT_EQ(result.records.size(), 1);
    ASSERT_TRUE(result.leverArm.has_value());

    const auto &record = result.records.front();
    EXPECT_EQ(record.fileName, QStringLiteral("IMG_001.JPG"));
    EXPECT_DOUBLE_EQ(record.wgs84LatitudeDegrees, 59.84507804);
    EXPECT_DOUBLE_EQ(record.wgs84LongitudeDegrees, 31.46683755);
    EXPECT_DOUBLE_EQ(record.wgs84EllipsoidalHeightMeters, 225.157);
    EXPECT_DOUBLE_EQ(record.rollDegrees, 1.81);
    EXPECT_DOUBLE_EQ(record.pitchDegrees, 3.67);
    EXPECT_DOUBLE_EQ(record.yawDegrees, -179.65);
    EXPECT_EQ(record.timeText, QStringLiteral("2019.08.06 09:26:06.109095"));
    ASSERT_TRUE(record.stdDevNorthMeters.has_value());
    ASSERT_TRUE(record.stdDevEastMeters.has_value());
    ASSERT_TRUE(record.stdDevUpMeters.has_value());
    ASSERT_TRUE(record.stdDevHorizontalMeters.has_value());
    EXPECT_DOUBLE_EQ(*record.stdDevNorthMeters, 0.0131);
    EXPECT_DOUBLE_EQ(*record.stdDevEastMeters, 0.0095);
    EXPECT_DOUBLE_EQ(*record.stdDevUpMeters, 0.018);
    EXPECT_DOUBLE_EQ(*record.stdDevHorizontalMeters, 0.0162);
    EXPECT_DOUBLE_EQ(result.leverArm->xMeters, 0.3682);
    EXPECT_DOUBLE_EQ(result.leverArm->yMeters, -0.1815);
    EXPECT_DOUBLE_EQ(result.leverArm->zMeters, -0.032);
    EXPECT_TRUE(result.warnings.isEmpty());
}

TEST(MetashapeCameraReferenceImporter, RejectsDuplicateBasenameIgnoringCase)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString camera_path = directory.filePath(QStringLiteral("duplicate.txt"));
    ASSERT_TRUE(writeUtf8File(camera_path,
                              kFullHeader
                                  + validCameraRecord("folder/IMAGE.JPG")
                                  + validCameraRecord("other\\image.jpg")));

    MetashapeCameraReferenceImportResult result;
    QString error;
    EXPECT_FALSE(importMetashapeCameraReferenceTxt(camera_path, {}, &result, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("第 3 行"))) << error.toStdString();
    EXPECT_TRUE(error.contains(QStringLiteral("文件名重复"))) << error.toStdString();
}

TEST(MetashapeCameraReferenceImporter, RejectsMissingRequiredColumn)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString camera_path = directory.filePath(QStringLiteral("missing_column.txt"));
    const QByteArray content =
        "# file\tWGS84_lat\tWGS84_lon\tWGS84_H\troll\tpitch\ttime\n"
        "IMAGE.JPG\t59\t31\t225\t1\t2\t2019.08.06 09:26:06\n";
    ASSERT_TRUE(writeUtf8File(camera_path, content));

    MetashapeCameraReferenceImportResult result;
    QString error;
    EXPECT_FALSE(importMetashapeCameraReferenceTxt(camera_path, {}, &result, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("第 1 行"))) << error.toStdString();
    EXPECT_TRUE(error.contains(QStringLiteral("yaw"))) << error.toStdString();
}

TEST(MetashapeCameraReferenceImporter, RejectsNonFiniteAndInvalidNumericValuesWithLineNumbers)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString camera_path = directory.filePath(QStringLiteral("invalid_numbers.txt"));
    const QByteArray content =
        kFullHeader
        + "NONFINITE.JPG\tnan\t31\t225\t1\t2\t3\ttime\t0.1\t0.2\t0.3\t0.4\n"
          "INVALID.JPG\t59\t31\t225\t1\t2\tnot-a-number\ttime\t0.1\t0.2\t0.3\t0.4\n";
    ASSERT_TRUE(writeUtf8File(camera_path, content));

    MetashapeCameraReferenceImportResult result;
    QString error;
    EXPECT_FALSE(importMetashapeCameraReferenceTxt(camera_path, {}, &result, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("第 2 行"))) << error.toStdString();
    EXPECT_TRUE(error.contains(QStringLiteral("WGS84_lat"))) << error.toStdString();
    EXPECT_TRUE(error.contains(QStringLiteral("第 3 行"))) << error.toStdString();
    EXPECT_TRUE(error.contains(QStringLiteral("yaw"))) << error.toStdString();
}

TEST(MetashapeCameraReferenceImporter, RejectsGnssOffsetWithMissingAxis)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString camera_path = directory.filePath(QStringLiteral("camera.txt"));
    const QString gnss_path = directory.filePath(QStringLiteral("gnss.txt"));
    ASSERT_TRUE(writeUtf8File(camera_path, kFullHeader + validCameraRecord("IMAGE.JPG")));
    ASSERT_TRUE(writeUtf8File(gnss_path, "Y=-0.1815\nX=0.3682\n"));

    MetashapeCameraReferenceImportResult result;
    QString error;
    EXPECT_FALSE(importMetashapeCameraReferenceTxt(camera_path, gnss_path, &result, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("缺少分量"))) << error.toStdString();
    EXPECT_TRUE(error.contains(QStringLiteral("Z"))) << error.toStdString();
}

TEST(MetashapeCameraReferenceImporter, RejectsOutOfRangeCoordinatesAndInvalidSigmas)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString camera_path = directory.filePath(QStringLiteral("invalid_physical_values.txt"));
    const QByteArray content =
        kFullHeader
        + "OUT_OF_RANGE.JPG\t91\t181\t225\t1\t2\t3\ttime\t-0.1\t0.2\t\t0\n";
    ASSERT_TRUE(writeUtf8File(camera_path, content));

    MetashapeCameraReferenceImportResult result;
    QString error;
    EXPECT_FALSE(importMetashapeCameraReferenceTxt(camera_path, {}, &result, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("[-90, 90]"))) << error.toStdString();
    EXPECT_TRUE(error.contains(QStringLiteral("[-180, 180]"))) << error.toStdString();
    EXPECT_TRUE(error.contains(QStringLiteral("必须大于 0"))) << error.toStdString();
    EXPECT_TRUE(error.contains(QStringLiteral("必须同时提供或同时留空")))
        << error.toStdString();
}

TEST(MetashapeCameraReferenceImporter, RejectsInvalidUtf8)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString camera_path = directory.filePath(QStringLiteral("invalid_utf8.txt"));
    QByteArray content = kFullHeader;
    content.append("IMAGE_");
    content.append(char(0xff));
    content.append(
        ".JPG\t59\t31\t225\t1\t2\t3\ttime\t0.1\t0.2\t0.3\t0.4\n");
    ASSERT_TRUE(writeUtf8File(camera_path, content));

    MetashapeCameraReferenceImportResult result;
    QString error;
    EXPECT_FALSE(importMetashapeCameraReferenceTxt(camera_path, {}, &result, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("UTF-8"))) << error.toStdString();
}

} // namespace
