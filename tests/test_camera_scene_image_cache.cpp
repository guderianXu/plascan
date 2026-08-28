#include "views/CameraSceneImageCache.h"

#include <gtest/gtest.h>

namespace
{

using xjw::gui::camera_scene::CameraSceneImageCache;

QImage makeImage(const QColor &color)
{
    QImage image(4, 4, QImage::Format_RGBA8888);
    image.fill(color);
    return image;
}

TEST(CameraSceneImageCacheTest, EvictsLeastRecentlyStoredFullImage)
{
    const QImage first = makeImage(Qt::red);
    const QImage second = makeImage(Qt::green);
    CameraSceneImageCache cache(static_cast<qint64>(first.sizeInBytes()));

    cache.store(QStringLiteral("full|first"), first, true);
    cache.store(QStringLiteral("full|second"), second, true);

    EXPECT_FALSE(cache.contains(QStringLiteral("full|first")));
    EXPECT_TRUE(cache.contains(QStringLiteral("full|second")));
    EXPECT_EQ(cache.image(QStringLiteral("full|second")).pixelColor(0, 0),
              QColor(Qt::green));
}

TEST(CameraSceneImageCacheTest, KeepsThumbnailEntriesOutsideFullImageBudget)
{
    CameraSceneImageCache cache(0);

    cache.store(QStringLiteral("thumb|camera"), makeImage(Qt::blue), false);

    EXPECT_TRUE(cache.contains(QStringLiteral("thumb|camera")));
}

TEST(CameraSceneImageCacheTest, ClearsEntriesAndFailuresByPrefix)
{
    CameraSceneImageCache cache;
    cache.store(QStringLiteral("thumb|one"), makeImage(Qt::red), false);
    cache.store(QStringLiteral("full|one"), makeImage(Qt::blue), true);
    EXPECT_TRUE(cache.markFailure(QStringLiteral("thumb|missing")));
    EXPECT_FALSE(cache.markFailure(QStringLiteral("thumb|missing")));

    cache.clearWithPrefix(QStringLiteral("thumb|"));

    EXPECT_FALSE(cache.contains(QStringLiteral("thumb|one")));
    EXPECT_FALSE(cache.hasFailure(QStringLiteral("thumb|missing")));
    EXPECT_TRUE(cache.contains(QStringLiteral("full|one")));
}

} // namespace
