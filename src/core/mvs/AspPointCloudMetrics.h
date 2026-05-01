#pragma once

#include <array>
#include <string>

namespace xjw
{
namespace mvs
{

struct AspPointCloudMetricsThresholds
{
    double minCoverageRatio = 0.80;
    double maxMedianDistance = 0.001;
    double maxP95Distance = 0.01;
};

struct AspPointCloudMetricsResult
{
    int aspWidth = 0;
    int aspHeight = 0;
    int aspValidPoints = 0;
    int plascanWidth = 0;
    int plascanHeight = 0;
    int plascanValidPoints = 0;
    double coverageRatio = 0.0;
    double bboxContainmentRatio = 0.0;
    double nnMean = 0.0;
    double nnMedian = 0.0;
    double nnP90 = 0.0;
    double nnP95 = 0.0;
    double nnMax = 0.0;
    std::array<double, 3> meanOffset = {0.0, 0.0, 0.0};
    bool passed = false;
    std::string failureReason;
};

class AspPointCloudMetrics
{
public:
    static bool compare(const std::string &plascanTifPath,
                        const std::string &aspTifPath,
                        const AspPointCloudMetricsThresholds &thresholds,
                        AspPointCloudMetricsResult &result,
                        std::string *errorMessage = nullptr);

    static std::string toTextReport(const AspPointCloudMetricsResult &result);
    static std::string toJson(const AspPointCloudMetricsResult &result);
};

} // namespace mvs
} // namespace xjw
