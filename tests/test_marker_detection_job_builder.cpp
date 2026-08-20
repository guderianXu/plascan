#include "MarkerDetectionJobBuilder.h"

#include "project/ProjectSessionModel.h"
#include "project/ProjectIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

using xjw::control_points::MarkerTargetFamily;
using xjw::common::project::ProjectIO;
using xjw::gui::markers::MarkerDetectionJobBuildOptions;
using xjw::gui::markers::MarkerDetectionJobBuilder;

namespace
{

QString writeImage(const QString &path)
{
    QImage image(64, 48, QImage::Format_Grayscale8);
    image.fill(Qt::white);
    return image.save(path) ? path : QString();
}

} // namespace

TEST(MarkerDetectionJobBuilderTest, ResolvesImagesSignaturesAndMasksFromProjectMetadata)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString project_path = directory.filePath(QStringLiteral("markers.plascan"));

    ProjectData project;
    ASSERT_TRUE(project.createProject(project_path, QStringLiteral("markers")));
    const QDir project_root(ProjectIO::projectRootFromPlascan(project_path));
    const QString image_path =
        writeImage(project_root.filePath(QStringLiteral("photo.png")));
    const QString mask_path =
        writeImage(project_root.filePath(QStringLiteral("photo-mask.png")));
    ASSERT_FALSE(image_path.isEmpty());
    ASSERT_FALSE(mask_path.isEmpty());

    ASSERT_TRUE(project.addImages({image_path}));
    QJsonObject metadata = project.coreFilesMeta();
    QJsonArray images = metadata.value(QStringLiteral("images")).toArray();
    ASSERT_EQ(images.size(), 1);
    QJsonObject image = images[0].toObject();
    image.insert(QStringLiteral("path"), QFileInfo(image_path).fileName());
    image.insert(QStringLiteral("mask_path"), QFileInfo(mask_path).fileName());
    image.insert(QStringLiteral("image_content_signature"), QStringLiteral("sha256:abc"));
    images[0] = image;
    metadata.insert(QStringLiteral("images"), images);
    project.updateMetadata(metadata);

    MarkerDetectionJobBuildOptions options;
    options.baseRevision = 17;
    options.targetFamilies = {MarkerTargetFamily::AprilTag36h11};
    options.maxConcurrentImages = 3;
    const auto result = MarkerDetectionJobBuilder::build(project, options);

    ASSERT_TRUE(result.ok) << qPrintable(result.errors.join(QLatin1Char('\n')));
    ASSERT_EQ(result.job.images.size(), 1);
    EXPECT_EQ(result.job.baseRevision, 17u);
    EXPECT_EQ(result.job.maxConcurrentImages, 3);
    EXPECT_EQ(QDir::cleanPath(result.job.images[0].imagePath), QDir::cleanPath(image_path));
    EXPECT_EQ(QDir::cleanPath(result.job.images[0].maskPath), QDir::cleanPath(mask_path));
    EXPECT_EQ(result.job.images[0].imageContentSignature, QStringLiteral("sha256:abc"));
    EXPECT_EQ(result.job.images[0].imageId, image.value(QStringLiteral("image_uuid")).toString());
}

TEST(MarkerDetectionJobBuilderTest, RejectsMissingImagesAndMissingStableIds)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    ProjectData project;
    ASSERT_TRUE(project.createProject(directory.filePath(QStringLiteral("broken.plascan")),
                                      QStringLiteral("broken")));
    QJsonObject metadata = project.coreFilesMeta();
    metadata.insert(QStringLiteral("images"), QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("missing.png")},
                    {QStringLiteral("image_uuid"), QStringLiteral("")}}
    });
    project.updateMetadata(metadata, false);

    MarkerDetectionJobBuildOptions options;
    options.targetFamilies = {MarkerTargetFamily::NonCodedCircle};
    const auto result = MarkerDetectionJobBuilder::build(project, options);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.job.images.isEmpty());
    EXPECT_FALSE(result.errors.isEmpty());
}
