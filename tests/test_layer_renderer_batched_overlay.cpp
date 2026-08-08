#include "LayerRenderer.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QDir>
#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QImage>
#include <QTemporaryDir>

#include "io/PathIO.h"

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include <vector>

namespace
{
int countFeatureOverlayItems(const QGraphicsScene &scene)
{
    int count = 0;
    for (QGraphicsItem *item : scene.items())
    {
        if (item && item->zValue() >= 900.0)
        {
            ++count;
        }
    }
    return count;
}

std::vector<cv::KeyPoint> makeKeypoints(int count)
{
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        const float x = static_cast<float>(5 + (i % 90));
        const float y = static_cast<float>(5 + ((i / 90) % 90));
        cv::KeyPoint keypoint(x, y, 3.0f);
        keypoint.response = static_cast<float>(count - i);
        keypoints.push_back(keypoint);
    }
    return keypoints;
}
} // namespace

TEST(LayerRendererResponsivenessTest, LargeFeatureSetUsesSingleSceneItem)
{
    QGraphicsScene scene;
    LayerRenderer renderer(&scene);

    QImage image(128, 128, QImage::Format_RGB32);
    image.fill(Qt::black);
    ASSERT_TRUE(renderer.addImageLayer(image, 0));

    renderer.addFeatureItems(makeKeypoints(2000));

    EXPECT_EQ(countFeatureOverlayItems(scene), 1);

    renderer.clearFeatureLayers();
    EXPECT_EQ(countFeatureOverlayItems(scene), 0);
}

TEST(LayerRendererResponsivenessTest, ClearingFeatureLayerKeepsImageLayer)
{
    QGraphicsScene scene;
    LayerRenderer renderer(&scene);

    QImage image(128, 128, QImage::Format_RGB32);
    image.fill(Qt::black);
    ASSERT_TRUE(renderer.addImageLayer(image, 0));

    renderer.addFeatureItems(makeKeypoints(20));
    ASSERT_EQ(scene.items().size(), 2);

    renderer.clearFeatureLayers();

    EXPECT_EQ(countFeatureOverlayItems(scene), 0);
    EXPECT_EQ(scene.items().size(), 1);
}

TEST(LayerRendererMaskOverlayTest, MaskContourUsesCosmeticHaloAndOutline)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString maskPath = QDir(tempDir.path()).filePath(QStringLiteral("mask.png"));
    cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(255));
    mask(cv::Rect(14, 12, 36, 38)).setTo(cv::Scalar(0));
    ASSERT_TRUE(xjw::common::io::writeImage(maskPath, mask));

    QGraphicsScene scene;
    LayerRenderer renderer(&scene);
    ASSERT_TRUE(renderer.addMaskContourLayer(maskPath));

    bool hasCosmeticHalo = false;
    bool hasCosmeticOutline = false;
    for (QGraphicsItem *item : scene.items())
    {
        auto *pathItem = dynamic_cast<QGraphicsPathItem *>(item);
        if (!pathItem)
        {
            continue;
        }

        const QPen pen = pathItem->pen();
        if (pen.color().red() == 0 && pen.color().green() == 0 && pen.color().blue() == 0)
        {
            hasCosmeticHalo = pen.isCosmetic() && pen.widthF() >= 4.0;
        }
        if (pen.color().red() == 255 && pen.color().green() == 255 && pen.color().blue() == 255)
        {
            hasCosmeticOutline = pen.isCosmetic() && pen.widthF() >= 1.5;
        }
    }

    EXPECT_TRUE(hasCosmeticHalo);
    EXPECT_TRUE(hasCosmeticOutline);
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
