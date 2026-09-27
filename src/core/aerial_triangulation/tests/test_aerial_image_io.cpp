#include "reporting/AerialTriangulationResultWriter.h"

#include <placamera/frame_numeric_state.h>
#include "io/ImageIO.h"
#include "reconstruction/SfmReconstruction.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <array>
#include <memory>

namespace
{

    enum class ColorInput
    {
        Rgb,
        SmallRaster,
        Gray16,
        Missing,
        Corrupt
    };

    class AerialImageColorTest : public testing::TestWithParam<ColorInput>
    {
    };

    TEST_P(AerialImageColorTest, ExportsFirstObservationColorAndPreservesGeometry)
    {
        const QString root = QString::fromUtf8(PLASCAN_AERIAL_IO_TEST_TMP_DIR);
        ASSERT_TRUE(QDir().mkpath(root));
        QTemporaryDir temporary(root + QStringLiteral("/color-XXXXXX"));
        ASSERT_TRUE(temporary.isValid());
        const ColorInput mode = GetParam();
        const QString first =
            temporary.filePath(mode == ColorInput::Gray16 ? QStringLiteral("月球.tif") : QStringLiteral("月球.png"));
        const QString second = temporary.filePath(QStringLiteral("second.png"));
        std::array<unsigned char, 3> expected{20, 40, 60};
        if (mode == ColorInput::Rgb || mode == ColorInput::SmallRaster)
        {
            const bool small = mode == ColorInput::SmallRaster;
            cv::Mat image(small ? 6 : 48, small ? 8 : 64, CV_8UC3, cv::Scalar(1, 2, 3));
            image.at<cv::Vec3b>(small ? 5 : 25, small ? 7 : 33) = cv::Vec3b(60, 40, 20);
            ASSERT_TRUE(xjw::common::io::writeImage(first, image));
        }
        else if (mode == ColorInput::Gray16)
        {
            ASSERT_TRUE(xjw::common::io::writeImage(first, cv::Mat(48, 64, CV_16UC1, cv::Scalar(35072))));
            expected = {137, 137, 137};
        }
        else
        {
            expected = {128, 128, 128};
            if (mode == ColorInput::Corrupt)
            {
                QFile file(first);
                ASSERT_TRUE(file.open(QIODevice::WriteOnly));
                ASSERT_EQ(file.write("invalid image"), 13);
            }
        }
        // A readable second view must not replace or average the designated first observation.
        ASSERT_TRUE(xjw::common::io::writeImage(second, cv::Mat(48, 64, CV_8UC3, cv::Scalar(120, 100, 80))));

        auto reconstruction = std::make_shared<xjw::SfmReconstruction>();
        for (int index = 0; index < 2; ++index)
        {
            xjw::ImageData image;
            image.id = static_cast<xjw::ImageId>(index);
            image.imagePath = (index == 0 ? first : second).toStdString();
            image.keypoints = {{32.5f, 24.5f, 2.0f}};
            image.point3DIds = {xjw::kInvalidPoint3DId};
            reconstruction->addImage(image);
            const placamera::FrameId frame("local");
            const auto definition = placamera::FramePinholeDefinition::create(
                placamera::CameraDefinitionId(QStringLiteral("definition-%1").arg(index).toStdString()),
                {70.0, 70.0, 32.0, 24.0, 1.0, 1, 1},
                {},
                placamera::PixelConvention::PixelCenter,
                frame);
            const auto camera = placamera::FramePinholeNumericState::fromModel(placamera::FramePinholeModel::create(
                placamera::CameraInstanceId(QStringLiteral("instance-%1").arg(index).toStdString()),
                placamera::ImageId(QStringLiteral("image-%1").arg(index).toStdString()),
                definition,
                {64, 48},
                placamera::Pose::create(
                    frame, {index == 0 ? -0.5 : 0.5, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
            reconstruction->registerImage(image.id, camera);
        }
        xjw::Track track;
        track.elements = {{0, 0}, {1, 0}};
        const auto point = reconstruction->addPoint3DWithTrack({0.0, 0.0, 5.0}, track);
        reconstruction->point3D(point).error = 0.25;

        xjw::aerial_triangulation::PreparedAerialTriangulationInput input;
        input.images = {first, second};
        input.imageIds = {placamera::ImageId("image-0"), placamera::ImageId("image-1")};
        input.outputDir = temporary.filePath(QStringLiteral("output"));
        xjw::aerial_triangulation::SfmAttemptExecutionResult execution;
        execution.reconstruction = reconstruction;
        execution.result.success = true;
        execution.result.numRegisteredImages = 2;
        execution.result.numPoints3D = 1;
        execution.result.baTracksTotal = 1;
        execution.result.baTracksOptimized = 1;
        QString error;
        ASSERT_TRUE(xjw::aerial_triangulation::AerialTriangulationResultWriter().write(input, &execution, &error))
            << qPrintable(error);
        EXPECT_EQ(execution.result.numPoints3D, 1);
        for (const QString& path : {execution.result.sparseCloudPath, execution.result.displaySparseCloudPath})
        {
            QFile file(path);
            ASSERT_TRUE(file.open(QIODevice::ReadOnly));
            const QByteArray bytes = file.readAll();
            const qsizetype offset = bytes.indexOf("end_header\n") + 11;
            ASSERT_GT(offset, 10);
            ASSERT_EQ(bytes.size() - offset, 15);
            for (int channel = 0; channel < 3; ++channel)
            {
                EXPECT_EQ(static_cast<unsigned char>(bytes.at(offset + 12 + channel)), expected[channel]);
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(RasterInputs,
                             AerialImageColorTest,
                             testing::Values(ColorInput::Rgb,
                                             ColorInput::SmallRaster,
                                             ColorInput::Gray16,
                                             ColorInput::Missing,
                                             ColorInput::Corrupt));

} // namespace
