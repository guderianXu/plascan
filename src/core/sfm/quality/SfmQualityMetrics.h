#pragma once

#include <map>
#include <string>
#include <vector>

namespace xjw
{

struct SfmQualityObservation
{
    int imageId = -1;
    double x = 0.0;
    double y = 0.0;
};

struct SfmQualityPoint
{
    int trackLength = 0;
    double reprojectionErrorPx = 0.0;
    double triangulationAngleDeg = 0.0;
    std::vector<SfmQualityObservation> observations;
};

struct SfmQualityMetricsOptions
{
    int totalImageCount = 0;
    int registeredImageCount = 0;
    double imageWidth = 0.0;
    double imageHeight = 0.0;
    int coverageGridColumns = 4;
    int coverageGridRows = 4;
    int minTrackLength = 3;
    double minTriangulationAngleDeg = 2.0;
    double maxReprojectionErrorPx = 3.0;
    double minRegisteredImageRatioForMvs = 0.50;
    double warnTwoViewTrackRatioForMvs = 0.70;
    double maxTwoViewTrackRatioForMvs = 0.85;
    double maxHighReprojectionErrorRatioForMvs = 0.30;
    double maxWeakTriangulationAngleRatioForMvs = 0.60;
    double minObservationGridCoverageMeanForMvs = 0.0;
};

struct SfmNumericSummary
{
    int count = 0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double p50 = 0.0;
    double p84 = 0.0;
    double p95 = 0.0;
};

struct SfmQualityMetrics
{
    int totalImageCount = 0;
    int registeredImageCount = 0;
    int pointCount = 0;
    int twoViewTrackCount = 0;
    int multiViewTrackCount = 0;
    int weakTrackCount = 0;
    int weakTriangulationAngleCount = 0;
    int highReprojectionErrorCount = 0;
    int coverageGridColumns = 0;
    int coverageGridRows = 0;
    double observationGridCoverageMean = 0.0;
    bool acceptableForMvs = true;
    std::string qualityStatus = "ok";
    std::vector<std::string> qualityAdvisories;
    std::vector<std::string> qualityWarnings;

    SfmNumericSummary trackLength;
    SfmNumericSummary reprojectionError;
    SfmNumericSummary triangulationAngle;
    std::map<int, int> trackLengthHistogram;
};

SfmQualityMetrics computeSfmQualityMetrics(const std::vector<SfmQualityPoint> &points,
                                           const SfmQualityMetricsOptions &options);

} // namespace xjw
