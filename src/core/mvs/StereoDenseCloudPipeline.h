#pragma once

#include "MvsTypes.h"
#include "SubpixelRefiner.h"
#include "DisparityFilter.h"
#include "DisparityTriangulator.h"

#include <QObject>
#include <string>

namespace xjw
{
namespace mvs
{

struct StereoPipelineFilterConfig
{
    bool enableLeftRightDepthCheck = true;
    float leftRightDepthRatio = 0.05f;
    bool enableLocalDepthConsistency = true;
    int localWindowRadius = 2;
    int localMinNeighbors = 5;
    float localDepthRatio = 0.02f;
    bool enableIqrFilter = true;
    float iqrMultiplier = 1.5f;
};

struct StereoPipelineDepthRangeConfig
{
    double nearScale = 0.7;
    double farScale = 1.5;
};

enum class StereoPipelineGeometryMode
{
    OriginalDepth,
    RectifiedDisparity
};

struct StereoPipelineConfig
{
    // Feature matching
    std::string featureAlgorithm = "disk+lightglue";
    float matchScoreThreshold = 0.2f;

    StereoPipelineGeometryMode geometryMode = StereoPipelineGeometryMode::OriginalDepth;

    // PatchMatch
    PatchMatchConfig patchMatch;

    // Subpixel
    SubpixelConfig subpixel;

    // Disparity filter
    DisparityFilterConfig disparityFilter;

    // Triangulation
    TriangulationConfig triangulation;

    StereoPipelineFilterConfig filters;
    StereoPipelineDepthRangeConfig depthRange;

    // Output
    bool outputTif = true;
    bool outputPly = true;
    bool keepIntermediateMasks = false;
    int numThreads = 0;
};

struct StereoPipelineResult
{
    std::string tifPath;
    std::string plyPath;
    int totalPoints = 0;
    int validPoints = 0;
    int depthValidBeforeFiltering = 0;
    int leftRightRejected = 0;
    int leftRightRejectedOob = 0;
    int leftRightRejectedNoReverse = 0;
    int leftRightRejectedMismatch = 0;
    int localRejected = 0;
    int iqrRejected = 0;
    int validAfterFiltering = 0;
    float coveragePercent = 0.f;
    double medianTriError = 0.0;
    std::string errorMsg;
};

class StereoDenseCloudPipeline : public QObject
{
    Q_OBJECT
public:
    explicit StereoDenseCloudPipeline(QObject *parent = nullptr);

    bool run(const std::string &leftImagePath,
             const std::string &rightImagePath,
             const std::string &leftCameraPath,
             const std::string &rightCameraPath,
             const std::string &outputDir,
             StereoPipelineResult *result = nullptr);

    bool run(const cv::Mat &leftImage,
             const cv::Mat &rightImage,
             const Camera &leftCamera,
             const Camera &rightCamera,
             const std::string &outputDir,
             StereoPipelineResult *result = nullptr);

    void setConfig(const StereoPipelineConfig &cfg) { m_config = cfg; }
    void cancel() { m_cancelled = true; }

signals:
    void progressChanged(QString stage, float ratio);
    void finished(bool success);

private:
    StereoPipelineConfig m_config;
    bool m_cancelled = false;
};

} // namespace mvs
} // namespace xjw
