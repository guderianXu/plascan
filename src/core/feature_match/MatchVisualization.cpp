#include "MatchVisualization.h"
#include "io/PathIO.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <QDir>
#include <QFileInfo>

#include <fstream>

namespace xjw::feature_match
{

namespace
{

bool writeFallbackPpm(const QString &outputPath,
                      const cv::Mat &image)
{
    const QFileInfo outputInfo(outputPath);
    QString ppmPath;
    if (!outputInfo.completeBaseName().isEmpty())
    {
        ppmPath = outputInfo.dir().filePath(outputInfo.completeBaseName() + QStringLiteral(".ppm"));
    }
    else
    {
        ppmPath = outputPath + QStringLiteral(".ppm");
    }

    cv::Mat rgb;
    if (image.type() == CV_8UC3)
    {
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    }
    else if (image.type() == CV_8UC1)
    {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    }
    else
    {
        image.convertTo(rgb, CV_8U, 255.0);
        if (rgb.channels() == 1)
        {
            cv::cvtColor(rgb, rgb, cv::COLOR_GRAY2RGB);
        }
        else if (rgb.channels() == 4)
        {
            cv::cvtColor(rgb, rgb, cv::COLOR_BGRA2RGB);
        }
        else if (rgb.channels() == 3)
        {
            cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
        }
    }

    std::ofstream out = xjw::common::io::openOutputFile(ppmPath, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
    {
        return false;
    }

    out << "P6\n" << rgb.cols << " " << rgb.rows << "\n255\n";
    for (int row = 0; row < rgb.rows; ++row)
    {
        const unsigned char *ptr = rgb.ptr<unsigned char>(row);
        out.write(reinterpret_cast<const char *>(ptr), rgb.cols * 3);
    }
    return true;
}

} // namespace

bool saveMatchVisualization(const cv::Mat &image0,
                            const cv::Mat &image1,
                            const std::vector<cv::KeyPoint> &keypoints0,
                            const std::vector<cv::KeyPoint> &keypoints1,
                            const MatchResult &result,
                            const QString &outputPath)
{
    cv::Mat outputImage;
    cv::drawMatches(image0,
                    keypoints0,
                    image1,
                    keypoints1,
                    result.cvMatches,
                    outputImage,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const std::string label = "Matches: " + std::to_string(result.numMatches);
    cv::putText(outputImage,
                label,
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 0),
                2);

    try
    {
        if (xjw::common::io::writeImage(outputPath, outputImage))
        {
            return true;
        }
    }
    catch (const cv::Exception &)
    {
    }

    return writeFallbackPpm(outputPath, outputImage);
}

} // namespace xjw::feature_match
