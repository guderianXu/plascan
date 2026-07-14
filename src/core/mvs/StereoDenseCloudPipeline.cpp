#include "StereoDenseCloudPipeline.h"

#include "MvsImagePreprocessor.h"
#include "StereoDenseCloudPipelinePaths.h"
#include "io/PathIO.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace xjw
{
namespace mvs
{
namespace
{

cv::Mat toGray8U(const cv::Mat &img)
{
    cv::Mat gray;
    if (img.channels() == 1)
    {
        gray = img;
    }
    else
    {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }

    if (gray.type() != CV_8U)
    {
        double minV;
        double maxV;
        cv::minMaxLoc(gray, &minV, &maxV);
        if (maxV > 255.0)
        {
            gray.convertTo(gray, CV_8U, 255.0 / maxV);
        }
        else
        {
            gray.convertTo(gray, CV_8U);
        }
    }

    return gray;
}

} // namespace

StereoDenseCloudPipeline::StereoDenseCloudPipeline(QObject *parent)
    : QObject(parent)
{
    _config.patchMatch.numIterations = 16;
    _config.patchMatch.downsampleFactor = 1;
    _config.patchMatch.geomConsistency = false;
    _config.patchMatch.numSourceViews = 1;
    _config.triangulation.maxTriangulationError = 0.001f;
}

bool StereoDenseCloudPipeline::run(const std::string &leftImagePath,
                                   const std::string &rightImagePath,
                                   const std::string &leftCameraPath,
                                   const std::string &rightCameraPath,
                                   const std::string &outputDir,
                                   StereoPipelineResult *result)
{
    cv::Mat leftImg = xjw::common::io::readImage(leftImagePath, cv::IMREAD_UNCHANGED);
    cv::Mat rightImg = xjw::common::io::readImage(rightImagePath, cv::IMREAD_UNCHANGED);
    if (leftImg.empty() || rightImg.empty())
    {
        if (result)
        {
            result->errorMsg = "Failed to load images";
        }
        return false;
    }

    Camera camL;
    Camera camR;
    if (!camL.loadFromFile(leftCameraPath) || !camR.loadFromFile(rightCameraPath))
    {
        if (result)
        {
            result->errorMsg = "Failed to load camera files";
        }
        return false;
    }

    return run(leftImg, rightImg, camL, camR, outputDir, result);
}

bool StereoDenseCloudPipeline::run(const cv::Mat &leftImage,
                                   const cv::Mat &rightImage,
                                   const Camera &leftCamera,
                                   const Camera &rightCamera,
                                   const std::string &outputDir,
                                   StereoPipelineResult *result)
{
    StereoPipelineResult res;

    emit progressChanged("Preprocessing", 0.0f);
    cv::Mat grayL = toGray8U(leftImage);
    cv::Mat grayR = toGray8U(rightImage);
    cv::Mat preparedLeft;
    cv::Mat preparedRight;
    Camera preparedLeftCamera;
    Camera preparedRightCamera;
    std::string preprocessError;
    if (!prepareMvsImage(grayL,
                         leftCamera,
                         &preparedLeft,
                         &preparedLeftCamera,
                         &preprocessError))
    {
        res.errorMsg = "Left image preprocessing failed: " + preprocessError;
        if (result)
        {
            *result = res;
        }
        emit finished(false);
        return false;
    }
    if (!prepareMvsImage(grayR,
                         rightCamera,
                         &preparedRight,
                         &preparedRightCamera,
                         &preprocessError))
    {
        res.errorMsg = "Right image preprocessing failed: " + preprocessError;
        if (result)
        {
            *result = res;
        }
        emit finished(false);
        return false;
    }

    grayL = std::move(preparedLeft);
    grayR = std::move(preparedRight);
    std::fprintf(stderr, "[StereoPipeline] Images: L=%dx%d R=%dx%d\n",
                 grayL.cols, grayL.rows, grayR.cols, grayR.rows);

    bool ok = false;
    if (_config.geometryMode == StereoPipelineGeometryMode::OriginalDepth)
    {
        ok = runOriginalDepthPath(grayL,
                                  grayR,
                                  preparedLeftCamera,
                                  preparedRightCamera,
                                  outputDir,
                                  _config,
                                  res,
                                  this);
    }
    else
    {
        ok = runRectifiedDisparityPath(grayL,
                                       grayR,
                                       preparedLeftCamera,
                                       preparedRightCamera,
                                       outputDir,
                                       _config,
                                       res,
                                       this);
    }

    if (result)
    {
        *result = res;
    }
    emit finished(ok);
    return ok;
}

} // namespace mvs
} // namespace xjw
