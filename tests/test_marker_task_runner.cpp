#include "MarkerTaskRunner.h"

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QEventLoop>
#include <QImage>
#include <QPainter>
#include <QTemporaryDir>
#include <QTimer>

using xjw::control_points::MarkerTargetFamily;
using xjw::gui::markers::MarkerDetectionImage;
using xjw::gui::markers::MarkerDetectionJob;
using xjw::gui::markers::MarkerDetectionProgress;
using xjw::gui::markers::MarkerDetectionTaskResult;
using xjw::gui::markers::MarkerTaskRunner;

namespace
{

QImage circleImage(const QPointF &center)
{
    QImage image(240, 200, QImage::Format_Grayscale8);
    image.fill(Qt::white);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::black);
    painter.drawEllipse(center + QPointF(0.5, 0.5), 20.0, 20.0);
    return image;
}

TEST(MarkerTaskRunnerTest, ReportsRealProgressAndReturnsDetections)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());

    MarkerDetectionJob job;
    job.baseRevision = 42;
    job.targetFamilies = {MarkerTargetFamily::NonCodedCircle};
    job.maxConcurrentImages = 2;
    for (int index = 0; index < 3; ++index)
    {
        const QString path = directory.filePath(QStringLiteral("image-%1.png").arg(index));
        ASSERT_TRUE(circleImage(QPointF(100.0 + index, 90.0)).save(path));
        job.images.push_back({QStringLiteral("image-%1").arg(index),
                              path,
                              {},
                              QStringLiteral("signature-%1").arg(index)});
    }

    MarkerTaskRunner runner;
    QVector<MarkerDetectionProgress> progress;
    MarkerDetectionTaskResult result;
    bool completed = false;
    QEventLoop loop;
    QObject::connect(&runner, &MarkerTaskRunner::progressChanged,
                     &loop, [&](const MarkerDetectionProgress &value) { progress.push_back(value); });
    QObject::connect(&runner, &MarkerTaskRunner::finished,
                     &loop, [&](const MarkerDetectionTaskResult &value)
    {
        result = value;
        completed = true;
        loop.quit();
    });
    QTimer::singleShot(10000, &loop, &QEventLoop::quit);

    ASSERT_TRUE(runner.start(job));
    loop.exec();

    ASSERT_TRUE(completed);
    EXPECT_FALSE(result.cancelled);
    EXPECT_EQ(result.baseRevision, 42u);
    EXPECT_TRUE(result.errors.isEmpty()) << qPrintable(result.errors.join(QLatin1Char('\n')));
    EXPECT_EQ(result.observations.size(), 3);
    ASSERT_FALSE(progress.isEmpty());
    EXPECT_EQ(progress.back().imagesCompleted, 3);
    EXPECT_EQ(progress.back().imageCount, 3);
    EXPECT_EQ(progress.back().candidatesDetected, 3);
}

TEST(MarkerTaskRunnerTest, CancellationPreventsAResultFromBeingApplied)
{
    MarkerDetectionJob job;
    job.baseRevision = 9;
    job.targetFamilies = {MarkerTargetFamily::NonCodedCircle};
    for (int index = 0; index < 20; ++index)
    {
        job.images.push_back({QStringLiteral("missing-%1").arg(index),
                              QStringLiteral("E:/missing/%1.png").arg(index),
                              {},
                              {}});
    }

    MarkerTaskRunner runner;
    MarkerDetectionTaskResult result;
    bool completed = false;
    QEventLoop loop;
    QObject::connect(&runner, &MarkerTaskRunner::finished,
                     &loop, [&](const MarkerDetectionTaskResult &value)
    {
        result = value;
        completed = true;
        loop.quit();
    });
    ASSERT_TRUE(runner.start(job));
    runner.cancel();
    QTimer::singleShot(10000, &loop, &QEventLoop::quit);
    loop.exec();

    ASSERT_TRUE(completed);
    EXPECT_TRUE(result.cancelled);
    EXPECT_TRUE(result.observations.isEmpty());
}

} // namespace

int main(int argc, char **argv)
{
    QCoreApplication application(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
