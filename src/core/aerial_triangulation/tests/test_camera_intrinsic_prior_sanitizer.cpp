#include "reconstruction/CameraIntrinsicPriorSanitizer.h"

#include <gtest/gtest.h>

#include <QJsonArray>

namespace
{

QJsonObject makeCamera(double focal)
{
    return QJsonObject{
        {QStringLiteral("fu"), focal},
        {QStringLiteral("fv"), focal},
        {QStringLiteral("cu"), 320.0},
        {QStringLiteral("cv"), 240.0},
        {QStringLiteral("pitch"), 1.0},
        {QStringLiteral("intrinsics_unit"), QStringLiteral("mm")},
        {QStringLiteral("C"), QJsonArray{0.0, 0.0, 0.0}},
        {QStringLiteral("R"), QJsonArray{1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0}},
    };
}

} // namespace

TEST(CameraIntrinsicPriorSanitizerTest, NormalizesOnlyExtremeFocalOutlierInDominantCameraGroup)
{
    QStringList imagePaths;
    QMap<QString, QJsonObject> cameras;
    for (int index = 0; index < 15; ++index)
    {
        const QString path = QStringLiteral("image_%1.png").arg(index);
        imagePaths.append(path);
        cameras.insert(path, makeCamera(1536.0 + (index % 3)));
    }
    imagePaths.append(QStringLiteral("image_bad.png"));
    cameras.insert(QStringLiteral("image_bad.png"), makeCamera(352.0));

    const xjw::aerial_triangulation::CameraIntrinsicPriorSanitizationResult result =
        xjw::aerial_triangulation::sanitizeProjectCameraIntrinsicPriors(imagePaths, &cameras);

    EXPECT_EQ(result.inspectedCameraCount, 16);
    EXPECT_EQ(result.dominantGroupCount, 15);
    EXPECT_EQ(result.normalizedCameraCount, 1);
    EXPECT_EQ(result.normalizedImagePaths, QStringList{QStringLiteral("image_bad.png")});
    EXPECT_NEAR(cameras.value(QStringLiteral("image_bad.png")).value(QStringLiteral("fu")).toDouble(),
                1537.0,
                1.0);
    EXPECT_DOUBLE_EQ(cameras.value(QStringLiteral("image_0.png")).value(QStringLiteral("fu")).toDouble(),
                     1536.0);
}

TEST(CameraIntrinsicPriorSanitizerTest, KeepsModerateFocalDifferencesAndMixedCameraGroups)
{
    QStringList imagePaths;
    QMap<QString, QJsonObject> cameras;
    const QList<double> focals{900.0, 920.0, 1100.0, 1120.0, 1700.0, 1720.0};
    for (int index = 0; index < focals.size(); ++index)
    {
        const QString path = QStringLiteral("camera_%1.png").arg(index);
        imagePaths.append(path);
        cameras.insert(path, makeCamera(focals.at(index)));
    }

    const xjw::aerial_triangulation::CameraIntrinsicPriorSanitizationResult result =
        xjw::aerial_triangulation::sanitizeProjectCameraIntrinsicPriors(imagePaths, &cameras);

    EXPECT_EQ(result.normalizedCameraCount, 0);
    EXPECT_DOUBLE_EQ(cameras.value(QStringLiteral("camera_0.png")).value(QStringLiteral("fu")).toDouble(),
                     900.0);
    EXPECT_DOUBLE_EQ(cameras.value(QStringLiteral("camera_5.png")).value(QStringLiteral("fu")).toDouble(),
                     1720.0);
}
