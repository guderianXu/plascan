#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QProcess>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

std::unique_ptr<QTemporaryDir> makeUniqueTempDir()
{
    auto dir = std::make_unique<QTemporaryDir>(
        QString::fromStdString((fs::temp_directory_path() / "plascan_triangulate_cli_XXXXXX").string()));
    EXPECT_TRUE(dir->isValid()) << dir->errorString().toStdString();
    return dir;
}

int runTriangulateCli(const QStringList &arguments)
{
#ifdef PLASCAN_TRIANGULATE_CLI_PATH
    return QProcess::execute(QString::fromUtf8(PLASCAN_TRIANGULATE_CLI_PATH), arguments);
#else
    Q_UNUSED(arguments);
    return -1;
#endif
}

void writeCamera(const fs::path &path, double cx)
{
    std::ofstream out(path);
    ASSERT_TRUE(out.is_open()) << path;
    out << "VERSION_3\n";
    out << "fu = 100\n";
    out << "fv = 100\n";
    out << "cu = 50\n";
    out << "cv = 50\n";
    out << "u_direction = 1 0 0\n";
    out << "v_direction = 0 1 0\n";
    out << "w_direction = 0 0 1\n";
    out << "C = " << cx << " 0 0\n";
    out << "R = 1 0 0 0 1 0 0 0 1\n";
    out << "pitch = 1\n";
}

std::array<double, 4> readFirstPlyVertex(const fs::path &path)
{
    std::ifstream in(path);
    EXPECT_TRUE(in.is_open()) << path;

    std::string line;
    while (std::getline(in, line))
    {
        if (line == "end_header")
        {
            break;
        }
    }

    std::array<double, 4> vertex{0.0, 0.0, 0.0, 0.0};
    in >> vertex[0] >> vertex[1] >> vertex[2] >> vertex[3];
    EXPECT_TRUE(in.good() || in.eof());
    return vertex;
}

struct FirstPlyVertexWithIntensity
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double error = 0.0;
    int intensity = 0;
};

FirstPlyVertexWithIntensity readFirstPlyVertexWithIntensity(const fs::path &path)
{
    std::ifstream in(path);
    EXPECT_TRUE(in.is_open()) << path;

    std::string line;
    std::vector<std::string> propertyNames;
    while (std::getline(in, line))
    {
        if (line == "end_header")
        {
            break;
        }
        std::istringstream header(line);
        std::string propertyToken;
        std::string typeToken;
        std::string nameToken;
        if ((header >> propertyToken >> typeToken >> nameToken) && propertyToken == "property")
        {
            propertyNames.push_back(nameToken);
        }
    }
    const auto errorIt = std::find(propertyNames.begin(), propertyNames.end(), "error");
    const auto intensityIt = std::find(propertyNames.begin(), propertyNames.end(), "intensity");
    EXPECT_NE(errorIt, propertyNames.end());
    EXPECT_NE(intensityIt, propertyNames.end());

    std::vector<double> values(propertyNames.size(), 0.0);
    for (double &value : values)
    {
        in >> value;
    }
    EXPECT_TRUE(in.good() || in.eof());

    auto valueOf = [&](const std::string &name) {
        const auto it = std::find(propertyNames.begin(), propertyNames.end(), name);
        EXPECT_NE(it, propertyNames.end());
        if (it == propertyNames.end())
        {
            return 0.0;
        }
        return values[static_cast<std::size_t>(std::distance(propertyNames.begin(), it))];
    };
    FirstPlyVertexWithIntensity vertex;
    vertex.x = valueOf("x");
    vertex.y = valueOf("y");
    vertex.z = valueOf("z");
    vertex.error = valueOf("error");
    vertex.intensity = static_cast<int>(valueOf("intensity"));
    return vertex;
}

} // namespace

TEST(TriangulateCli, WritesAbsoluteWorldCoordinates)
{
#ifndef PLASCAN_TRIANGULATE_CLI_PATH
    GTEST_SKIP() << "triangulate_cli path is not configured";
#else
    auto tempDir = makeUniqueTempDir();
    ASSERT_TRUE(tempDir->isValid());
    const fs::path root = tempDir->path().toStdString();

    const fs::path leftCamera = root / "left.tsai";
    const fs::path rightCamera = root / "right.tsai";
    const fs::path rectPath = root / "rect.xml";
    const fs::path disparityPath = root / "disp.tif";
    const fs::path outPath = root / "cloud.ply";

    writeCamera(leftCamera, 0.0);
    writeCamera(rightCamera, 1.0);

    cv::Mat disparity(101, 101, CV_32FC1, cv::Scalar(0.0f));
    disparity.at<float>(50, 50) = 10.0f;
    ASSERT_TRUE(cv::imwrite(disparityPath.string(), disparity));

    cv::FileStorage fsOut(rectPath.string(), cv::FileStorage::WRITE);
    ASSERT_TRUE(fsOut.isOpened());
    fsOut << "H1inv" << cv::Mat::eye(3, 3, CV_64F);
    fsOut << "H2inv" << cv::Mat::eye(3, 3, CV_64F);
    fsOut.release();

    const QStringList arguments{
        QStringLiteral("--disparity"), QString::fromStdString(disparityPath.string()),
        QStringLiteral("--rect"), QString::fromStdString(rectPath.string()),
        QStringLiteral("--camL"), QString::fromStdString(leftCamera.string()),
        QStringLiteral("--camR"), QString::fromStdString(rightCamera.string()),
        QStringLiteral("--output"), QString::fromStdString(outPath.string()),
        QStringLiteral("--threads"), QStringLiteral("1")
    };
    ASSERT_EQ(runTriangulateCli(arguments), 0);

    const auto vertex = readFirstPlyVertex(outPath);
    EXPECT_NEAR(vertex[0], 0.0, 1e-6);
    EXPECT_NEAR(vertex[1], 0.0, 1e-6);
    EXPECT_NEAR(vertex[2], 10.0, 1e-6);
#endif
}

TEST(TriangulateCli, WritesIntensityWhenImageProvided)
{
#ifndef PLASCAN_TRIANGULATE_CLI_PATH
    GTEST_SKIP() << "triangulate_cli path is not configured";
#else
    auto tempDir = makeUniqueTempDir();
    ASSERT_TRUE(tempDir->isValid());
    const fs::path root = tempDir->path().toStdString();

    const fs::path leftCamera = root / "left.tsai";
    const fs::path rightCamera = root / "right.tsai";
    const fs::path rectPath = root / "rect.xml";
    const fs::path disparityPath = root / "disp.tif";
    const fs::path intensityPath = root / "left_rect.tif";
    const fs::path outPath = root / "cloud.ply";

    writeCamera(leftCamera, 0.0);
    writeCamera(rightCamera, 1.0);

    cv::Mat disparity(101, 101, CV_32FC1, cv::Scalar(0.0f));
    disparity.at<float>(50, 50) = 10.0f;
    ASSERT_TRUE(cv::imwrite(disparityPath.string(), disparity));

    cv::Mat intensity(101, 101, CV_8UC1, cv::Scalar(0));
    intensity.at<uchar>(50, 50) = 51;
    ASSERT_TRUE(cv::imwrite(intensityPath.string(), intensity));

    cv::FileStorage fsOut(rectPath.string(), cv::FileStorage::WRITE);
    ASSERT_TRUE(fsOut.isOpened());
    fsOut << "H1inv" << cv::Mat::eye(3, 3, CV_64F);
    fsOut << "H2inv" << cv::Mat::eye(3, 3, CV_64F);
    fsOut.release();

    const QStringList arguments{
        QStringLiteral("--disparity"), QString::fromStdString(disparityPath.string()),
        QStringLiteral("--rect"), QString::fromStdString(rectPath.string()),
        QStringLiteral("--camL"), QString::fromStdString(leftCamera.string()),
        QStringLiteral("--camR"), QString::fromStdString(rightCamera.string()),
        QStringLiteral("--output"), QString::fromStdString(outPath.string()),
        QStringLiteral("--intensity-image"), QString::fromStdString(intensityPath.string()),
        QStringLiteral("--threads"), QStringLiteral("1")
    };
    ASSERT_EQ(runTriangulateCli(arguments), 0);

    const auto vertex = readFirstPlyVertexWithIntensity(outPath);
    EXPECT_NEAR(vertex.x, 0.0, 1e-6);
    EXPECT_NEAR(vertex.y, 0.0, 1e-6);
    EXPECT_NEAR(vertex.z, 10.0, 1e-6);
    EXPECT_NEAR(vertex.error, 0.0, 1e-6);
    EXPECT_EQ(vertex.intensity, 51);
#endif
}
