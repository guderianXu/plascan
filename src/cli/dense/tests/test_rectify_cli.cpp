#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QDir>
#include <QFileInfo>
#include <QProcess>
#include <QTemporaryDir>

#include <filesystem>
#include <fstream>

namespace
{

    void writeCamera(const std::filesystem::path& path, double center_x, double radial_k1)
    {
        std::ofstream output(path);
        ASSERT_TRUE(output.is_open()) << path;
        output << "VERSION_3\n"
               << "fu = 100\n"
               << "fv = 100\n"
               << "cu = 32\n"
               << "cv = 24\n"
               << "u_direction = 1 0 0\n"
               << "v_direction = 0 1 0\n"
               << "w_direction = 0 0 1\n"
               << "C = " << center_x << " 0 0\n"
               << "R = 1 0 0 0 1 0 0 0 1\n"
               << "pitch = 1\n"
               << "k1 = " << radial_k1 << "\n";
    }

} // namespace

TEST(RectifyCliTest, ProducesRectifiedImagesAndHomographiesFromPlaCameraTsaiModels)
{
    const QString temporary_root = QString::fromUtf8(PLASCAN_RECTIFY_TEST_TMP_ROOT);
    ASSERT_TRUE(QDir().mkpath(temporary_root));
    QTemporaryDir directory(QDir(temporary_root).filePath(QStringLiteral("case-XXXXXX")));
    ASSERT_TRUE(directory.isValid()) << directory.errorString().toStdString();

    const auto root = std::filesystem::path(directory.path().toStdString());
    const auto left_path = root / "left.png";
    const auto right_path = root / "right.png";
    const auto left_camera_path = root / "left.tsai";
    const auto right_camera_path = root / "right.tsai";
    const auto prefix = root / "rectified";

    cv::Mat image(48, 64, CV_8UC1);
    for (int row = 0; row < image.rows; ++row)
    {
        for (int column = 0; column < image.cols; ++column)
        {
            image.at<unsigned char>(row, column) = static_cast<unsigned char>((row * 7 + column * 3) % 251);
        }
    }
    ASSERT_TRUE(cv::imwrite(left_path.string(), image));
    ASSERT_TRUE(cv::imwrite(right_path.string(), image));
    writeCamera(left_camera_path, 0.0, 0.02);
    writeCamera(right_camera_path, 0.2, 0.02);

    const int exit_code = QProcess::execute(QString::fromUtf8(PLASCAN_RECTIFY_CLI_PATH),
                                            {QStringLiteral("-L"),
                                             QString::fromStdString(left_path.string()),
                                             QStringLiteral("-R"),
                                             QString::fromStdString(right_path.string()),
                                             QStringLiteral("--camL"),
                                             QString::fromStdString(left_camera_path.string()),
                                             QStringLiteral("--camR"),
                                             QString::fromStdString(right_camera_path.string()),
                                             QStringLiteral("--ground-frame"),
                                             QStringLiteral("rectify-test-world"),
                                             QStringLiteral("-o"),
                                             QString::fromStdString(prefix.string())});
    ASSERT_EQ(exit_code, 0);

    const cv::Mat left_rectified = cv::imread(prefix.string() + "_L.tif", cv::IMREAD_GRAYSCALE);
    const cv::Mat right_rectified = cv::imread(prefix.string() + "_R.tif", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(left_rectified.empty());
    ASSERT_FALSE(right_rectified.empty());
    EXPECT_EQ(left_rectified.size(), image.size());
    EXPECT_EQ(right_rectified.size(), image.size());

    cv::FileStorage parameters(prefix.string() + ".xml", cv::FileStorage::READ);
    ASSERT_TRUE(parameters.isOpened());
    cv::Mat left_inverse;
    cv::Mat right_inverse;
    int original_width = 0;
    int original_height = 0;
    parameters["H1inv"] >> left_inverse;
    parameters["H2inv"] >> right_inverse;
    parameters["origW"] >> original_width;
    parameters["origH"] >> original_height;
    EXPECT_EQ(left_inverse.size(), cv::Size(3, 3));
    EXPECT_EQ(right_inverse.size(), cv::Size(3, 3));
    EXPECT_EQ(original_width, image.cols);
    EXPECT_EQ(original_height, image.rows);
}
