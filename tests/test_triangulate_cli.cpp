#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace
{

std::string shellQuote(const std::string &value)
{
    std::string quoted = "'";
    for (char ch : value)
    {
        if (ch == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
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

} // namespace

TEST(TriangulateCli, WritesAbsoluteWorldCoordinates)
{
#ifndef PLASCAN_TRIANGULATE_CLI_PATH
    GTEST_SKIP() << "triangulate_cli path is not configured";
#else
    const fs::path root = fs::temp_directory_path() / "plascan_triangulate_cli_regression";
    fs::remove_all(root);
    fs::create_directories(root);

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

    std::ostringstream command;
    command << shellQuote(PLASCAN_TRIANGULATE_CLI_PATH)
            << " --disparity " << shellQuote(disparityPath.string())
            << " --rect " << shellQuote(rectPath.string())
            << " --camL " << shellQuote(leftCamera.string())
            << " --camR " << shellQuote(rightCamera.string())
            << " --output " << shellQuote(outPath.string())
            << " --threads 1";

    ASSERT_EQ(std::system(command.str().c_str()), 0);

    const auto vertex = readFirstPlyVertex(outPath);
    EXPECT_NEAR(vertex[0], 0.0, 1e-6);
    EXPECT_NEAR(vertex[1], 0.0, 1e-6);
    EXPECT_NEAR(vertex[2], 10.0, 1e-6);
}
#endif
