#include "LayerRenderer.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFileInfo>
#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QImage>
#include <QTemporaryDir>
#include <QThread>

#include "io/PathIO.h"

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include <functional>
#include <vector>

namespace
{
bool waitUntil(const std::function<bool()> &condition, int timeout_ms = 5000)
{
    QElapsedTimer timer;
    timer.start();
    while (!condition() && timer.elapsed() < timeout_ms)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        QThread::msleep(1);
    }
    QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
    return condition();
}

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
    int readyCount = 0;
    QObject::connect(&renderer,
                     &LayerRenderer::maskContourLayerReady,
                     [&readyCount](const QString &, bool)
                     {
                         ++readyCount;
                     });
    ASSERT_TRUE(renderer.addMaskContourLayer(maskPath));
    // 首次请求不在调用线程读取蒙版或创建场景路径项。
    EXPECT_TRUE(scene.items().isEmpty());
    ASSERT_TRUE(waitUntil([&readyCount]() { return readyCount == 1; }));

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

TEST(LayerRendererMaskOverlayTest, ReusesOnlyMatchingPathAndModificationCacheEntry)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString maskPath = QDir(tempDir.path()).filePath(QStringLiteral("mask.png"));
    cv::Mat firstMask(64, 64, CV_8UC1, cv::Scalar(255));
    firstMask(cv::Rect(8, 8, 24, 24)).setTo(cv::Scalar(0));
    ASSERT_TRUE(xjw::common::io::writeImage(maskPath, firstMask));

    QGraphicsScene scene;
    LayerRenderer renderer(&scene);
    QVector<bool> cacheHits;
    QObject::connect(&renderer,
                     &LayerRenderer::maskContourLayerReady,
                     [&cacheHits](const QString &, bool from_cache)
                     {
                         cacheHits.append(from_cache);
                     });

    ASSERT_TRUE(renderer.addMaskContourLayer(maskPath));
    ASSERT_TRUE(waitUntil([&cacheHits]() { return cacheHits.size() == 1; }));
    EXPECT_FALSE(cacheHits.at(0));

    // 同一路径和修改时间直接复用已构建的路径，不再启动后台读取。
    const QString equivalentPath = QDir(tempDir.path()).filePath(QStringLiteral("./mask.png"));
    ASSERT_TRUE(renderer.addMaskContourLayer(equivalentPath));
    ASSERT_EQ(cacheHits.size(), 2);
    EXPECT_TRUE(cacheHits.at(1));

    cv::Mat changedMask(96, 96, CV_8UC1, cv::Scalar(255));
    changedMask(cv::Rect(40, 32, 28, 36)).setTo(cv::Scalar(0));
    ASSERT_TRUE(xjw::common::io::writeImage(maskPath, changedMask));
    ASSERT_TRUE(renderer.addMaskContourLayer(maskPath));
    ASSERT_TRUE(waitUntil([&cacheHits]() { return cacheHits.size() == 3; }));
    EXPECT_FALSE(cacheHits.at(2));
}

TEST(LayerRendererMaskOverlayTest, OlderResultCannotReplaceLatestMask)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString oldMaskPath = QDir(tempDir.path()).filePath(QStringLiteral("old.png"));
    const QString latestMaskPath = QDir(tempDir.path()).filePath(QStringLiteral("latest.png"));
    cv::Mat oldMask(128, 128, CV_8UC1, cv::Scalar(255));
    oldMask(cv::Rect(4, 4, 28, 28)).setTo(cv::Scalar(0));
    cv::Mat latestMask(128, 128, CV_8UC1, cv::Scalar(255));
    latestMask(cv::Rect(72, 68, 32, 36)).setTo(cv::Scalar(0));
    ASSERT_TRUE(xjw::common::io::writeImage(oldMaskPath, oldMask));
    ASSERT_TRUE(xjw::common::io::writeImage(latestMaskPath, latestMask));

    QGraphicsScene scene;
    LayerRenderer renderer(&scene);
    QStringList readyPaths;
    QObject::connect(&renderer,
                     &LayerRenderer::maskContourLayerReady,
                     [&readyPaths](const QString &path, bool)
                     {
                         readyPaths.append(QDir::cleanPath(path));
                     });

    ASSERT_TRUE(renderer.addMaskContourLayer(oldMaskPath));
    ASSERT_TRUE(renderer.addMaskContourLayer(latestMaskPath));
    ASSERT_TRUE(waitUntil([&readyPaths]() { return !readyPaths.isEmpty(); }));
    ASSERT_EQ(readyPaths.size(), 1);
    EXPECT_EQ(readyPaths.constFirst(),
              QDir::cleanPath(QFileInfo(latestMaskPath).canonicalFilePath()));

    for (QGraphicsItem *item : scene.items())
    {
        auto *pathItem = dynamic_cast<QGraphicsPathItem *>(item);
        if (pathItem)
        {
            EXPECT_GT(pathItem->path().boundingRect().left(), 60.0);
        }
    }

    // 即使旧后台任务稍后完成，也不能覆盖最新图层或发出 ready。
    EXPECT_FALSE(waitUntil([&readyPaths]() { return readyPaths.size() > 1; }, 100));
    EXPECT_EQ(readyPaths.size(), 1);
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
