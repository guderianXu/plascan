#include "widgets/PhotoStripWidget.h"

#include <functional>

#include <gtest/gtest.h>

#include <QApplication>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QJsonArray>
#include <QJsonObject>
#include <QListWidget>
#include <QSignalSpy>
#include <QThread>

namespace
{

    constexpr int SentinelRole = Qt::UserRole + 100;

    QJsonObject imageEntry(const QString& path, const QJsonObject& camera = {}, const QString& maskPath = {})
    {
        QJsonObject entry{{QStringLiteral("path"), path}};
        if (!camera.isEmpty())
        {
            entry.insert(QStringLiteral("camera"), camera);
        }
        if (!maskPath.isEmpty())
        {
            entry.insert(QStringLiteral("mask_path"), maskPath);
        }
        return entry;
    }

    QJsonObject projectMeta(const QJsonArray& images, int resultRevision = 0)
    {
        return {{QStringLiteral("images"), images},
                {QStringLiteral("depth_map_results"),
                 QJsonArray{QJsonObject{{QStringLiteral("revision"), resultRevision}}}}};
    }

    QListWidget* photoList(PhotoStripWidget* widget)
    {
        return widget->findChild<QListWidget*>(QStringLiteral("photoStripList"));
    }

    bool waitFor(const std::function<bool()>& condition, int timeoutMs = 3'000)
    {
        QElapsedTimer timer;
        timer.start();
        while (!condition() && timer.elapsed() < timeoutMs)
        {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
            QThread::msleep(1);
        }
        return condition();
    }

    TEST(PhotoStripWidgetRefreshTest, IgnoresResultOnlyMetadataAndPreservesListState)
    {
        PhotoStripWidget widget;
        widget.setProjectPath(QStringLiteral("/tmp/photo-strip-refresh/project.plascan"));
        QSignalSpy progressSpy(&widget, &PhotoStripWidget::imageLoadingProgressChanged);
        QSignalSpy finishedSpy(&widget, &PhotoStripWidget::imageLoadingFinished);
        const QJsonArray images{imageEntry(QStringLiteral("/tmp/photo-strip-refresh/first.tif")),
                                imageEntry(QStringLiteral("/tmp/photo-strip-refresh/second.tif"))};

        widget.loadFromJson(projectMeta(images));
        QListWidget* list = photoList(&widget);
        ASSERT_NE(list, nullptr);
        ASSERT_EQ(list->count(), 2);
        ASSERT_EQ(finishedSpy.count(), 1);
        QListWidgetItem* first = list->item(0);
        ASSERT_NE(first, nullptr);
        first->setData(SentinelRole, QStringLiteral("preserve-me"));
        widget.setCurrentPhoto(QStringLiteral("/tmp/photo-strip-refresh/first.tif"));
        ASSERT_EQ(list->currentItem(), first);
        ASSERT_TRUE(first->isSelected());

        progressSpy.clear();
        finishedSpy.clear();
        widget.loadFromJson(projectMeta(images, 1));

        EXPECT_TRUE(progressSpy.isEmpty());
        EXPECT_TRUE(finishedSpy.isEmpty());
        EXPECT_EQ(list->count(), 2);
        ASSERT_NE(list->item(0), nullptr);
        EXPECT_EQ(list->item(0)->data(SentinelRole).toString(), QStringLiteral("preserve-me"));
        EXPECT_EQ(list->currentItem(), list->item(0));
        EXPECT_TRUE(list->item(0)->isSelected());
    }

    TEST(PhotoStripWidgetRefreshTest, RefreshesForOrderAndFullImageEntryChanges)
    {
        PhotoStripWidget widget;
        QSignalSpy progressSpy(&widget, &PhotoStripWidget::imageLoadingProgressChanged);
        QSignalSpy finishedSpy(&widget, &PhotoStripWidget::imageLoadingFinished);
        const QString firstPath = QStringLiteral("/tmp/photo-strip-refresh/first.tif");
        const QString secondPath = QStringLiteral("/tmp/photo-strip-refresh/second.tif");
        const QJsonArray images{imageEntry(firstPath), imageEntry(secondPath)};

        widget.loadFromJson(projectMeta(images));
        QListWidget* list = photoList(&widget);
        ASSERT_NE(list, nullptr);
        ASSERT_EQ(list->count(), 2);
        list->item(0)->setData(SentinelRole, QStringLiteral("old-order"));

        progressSpy.clear();
        finishedSpy.clear();
        widget.loadFromJson(projectMeta(QJsonArray{imageEntry(secondPath), imageEntry(firstPath)}));
        ASSERT_EQ(progressSpy.count(), 2);
        ASSERT_EQ(finishedSpy.count(), 1);
        EXPECT_EQ(list->item(0)->data(SentinelRole), QVariant());
        EXPECT_EQ(list->item(0)->data(Qt::UserRole + 1).toString(), secondPath);

        progressSpy.clear();
        finishedSpy.clear();
        widget.loadFromJson(projectMeta(
            QJsonArray{imageEntry(secondPath, QJsonObject{{QStringLiteral("focal"), 12.5}}), imageEntry(firstPath)}));
        ASSERT_EQ(progressSpy.count(), 2);
        ASSERT_EQ(finishedSpy.count(), 1);
        EXPECT_TRUE(list->item(0)->toolTip().contains(QStringLiteral("已对齐")));

        list->item(0)->setData(SentinelRole, QStringLiteral("old-mask"));
        progressSpy.clear();
        finishedSpy.clear();
        widget.loadFromJson(
            projectMeta(QJsonArray{imageEntry(secondPath,
                                              QJsonObject{{QStringLiteral("focal"), 12.5}},
                                              QStringLiteral("/tmp/photo-strip-refresh/second_mask.png")),
                                   imageEntry(firstPath)}));
        ASSERT_EQ(progressSpy.count(), 2);
        ASSERT_EQ(finishedSpy.count(), 1);
        EXPECT_FALSE(list->item(0)->data(SentinelRole).isValid());
    }

    TEST(PhotoStripWidgetRefreshTest, ProjectChangesAndExplicitClearsInvalidateCache)
    {
        PhotoStripWidget widget;
        QSignalSpy progressSpy(&widget, &PhotoStripWidget::imageLoadingProgressChanged);
        QSignalSpy finishedSpy(&widget, &PhotoStripWidget::imageLoadingFinished);
        const QJsonArray images{imageEntry(QStringLiteral("/tmp/photo-strip-refresh/image.tif"))};

        widget.setProjectPath(QStringLiteral("/tmp/photo-strip-refresh/first.plascan"));
        widget.loadFromJson(projectMeta(images));
        progressSpy.clear();
        finishedSpy.clear();

        widget.setProjectPath(QStringLiteral("/tmp/photo-strip-refresh/second.plascan"));
        widget.loadFromJson(projectMeta(images));
        EXPECT_EQ(progressSpy.count(), 2);
        EXPECT_EQ(finishedSpy.count(), 1);

        progressSpy.clear();
        finishedSpy.clear();
        widget.clearPhotos();
        widget.loadFromJson(projectMeta(images));
        EXPECT_EQ(progressSpy.count(), 2);
        EXPECT_EQ(finishedSpy.count(), 1);
    }

    TEST(PhotoStripWidgetRefreshTest, DeduplicatesEmptyAndInFlightImageLists)
    {
        PhotoStripWidget widget;
        QSignalSpy progressSpy(&widget, &PhotoStripWidget::imageLoadingProgressChanged);
        QSignalSpy finishedSpy(&widget, &PhotoStripWidget::imageLoadingFinished);

        widget.loadFromJson(projectMeta({}));
        ASSERT_EQ(finishedSpy.count(), 1);
        EXPECT_TRUE(finishedSpy.at(0).at(0).toBool());
        progressSpy.clear();
        finishedSpy.clear();
        widget.loadFromJson(projectMeta({}));
        EXPECT_TRUE(progressSpy.isEmpty());
        EXPECT_TRUE(finishedSpy.isEmpty());

        QJsonArray manyImages;
        for (int index = 0; index <= 100; ++index)
        {
            manyImages.append(imageEntry(QStringLiteral("/tmp/photo-strip-refresh/in-flight-%1.tif").arg(index)));
        }
        widget.clearPhotos();
        widget.loadFromJson(projectMeta(manyImages));
        progressSpy.clear();
        finishedSpy.clear();
        widget.loadFromJson(projectMeta(manyImages, 1));

        EXPECT_TRUE(progressSpy.isEmpty());
        ASSERT_TRUE(waitFor([&finishedSpy]() { return finishedSpy.count() >= 1; }));
        ASSERT_EQ(finishedSpy.count(), 1);
        EXPECT_TRUE(finishedSpy.at(0).at(0).toBool());
    }

} // namespace

int main(int argc, char** argv)
{
    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication application(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
