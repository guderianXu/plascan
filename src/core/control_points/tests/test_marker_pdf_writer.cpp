#include "detection/AprilTagDetector.h"
#include "print/MarkerPdfWriter.h"
#include "print/MarkerSheetRenderer.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QGuiApplication>
#include <QSet>
#include <QTemporaryDir>

using xjw::control_points::AprilTagDetector;
using xjw::control_points::AprilTagFamily;
using xjw::control_points::MarkerPdfWriter;
using xjw::control_points::MarkerPrintRequest;
using xjw::control_points::MarkerSheetRenderer;
using xjw::control_points::MarkerTargetFamily;

TEST(MarkerPdfWriterTest, RenderedAprilTagSheetDecodesRequestedIds)
{
    MarkerPrintRequest request;
    request.family = MarkerTargetFamily::AprilTag36h11;
    request.ids = {1, 2, 3};
    request.targetDiameterMm = 30.0;
    request.showLabels = false;

    const auto rendered = MarkerSheetRenderer::render(request, 150);
    ASSERT_TRUE(rendered.ok) << qPrintable(rendered.error);
    ASSERT_EQ(rendered.pages.size(), 1);
    const auto detections = AprilTagDetector(AprilTagFamily::Tag36h11)
                                .detect(rendered.pages.front(), {}, {});
    QSet<int> ids;
    for (const auto &detection : detections)
    {
        ids.insert(detection.targetId);
    }
    EXPECT_EQ(ids, QSet<int>({1, 2, 3}));
}

TEST(MarkerPdfWriterTest, WritesPdfSignatureAndRejectsImpossibleLayout)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    MarkerPrintRequest request;
    request.family = MarkerTargetFamily::AprilTag16h5;
    request.ids = {1, 2};
    request.targetDiameterMm = 35.0;
    request.showLabels = false;
    const QString path = directory.filePath(QStringLiteral("markers.pdf"));

    const auto written = MarkerPdfWriter::write(request, path, 150);
    ASSERT_TRUE(written.ok) << qPrintable(written.error);
    EXPECT_EQ(written.pageCount, 1);
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_TRUE(file.read(5).startsWith("%PDF"));

    request.targetDiameterMm = 500.0;
    const auto invalid = MarkerSheetRenderer::render(request, 150);
    EXPECT_FALSE(invalid.ok);
}

TEST(MarkerPdfWriterTest, RefusesCircularCodesWithoutCompatibilityCorpus)
{
    MarkerPrintRequest request;
    request.family = MarkerTargetFamily::Circular12Bit;
    request.ids = {1};
    const auto rendered = MarkerSheetRenderer::render(request, 150);
    EXPECT_FALSE(rendered.ok);
    EXPECT_TRUE(rendered.error.contains(QStringLiteral("语料")));
}

int main(int argc, char **argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
    {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QGuiApplication application(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
