#include "LayerRenderer.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QImage>

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

TEST(LayerRendererResponsivenessTest, ClearingFeatureLayerDoesNotDeleteMatchLayer)
{
    QGraphicsScene scene;
    LayerRenderer renderer(&scene);

    QImage image(128, 128, QImage::Format_RGB32);
    image.fill(Qt::black);
    ASSERT_TRUE(renderer.addImageLayer(image, 0));

    renderer.addFeatureItems(makeKeypoints(20));
    renderer.addMatchLines({QPointF(10.0, 10.0), QPointF(20.0, 20.0)},
                           {QPointF(12.0, 11.0), QPointF(22.0, 21.0)},
                           0.0);

    renderer.clearFeatureLayers();

    EXPECT_EQ(countFeatureOverlayItems(scene), 6);
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
