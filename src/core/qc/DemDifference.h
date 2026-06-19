#pragma once

#include <QString>

#include <vector>

namespace xjw::qc
{

struct DemGrid
{
    int width = 0;
    int height = 0;
    double nodata = -9999.0;
    QString projection;
    std::vector<double> values;
};

struct DemDifferenceResult
{
    bool success = false;
    QString error;
    int validCount = 0;
    double mean = 0.0;
    double rmse = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    std::vector<double> differences;
};

class DemDifference
{
public:
    static DemDifferenceResult compareSameGrid(const DemGrid &candidate,
                                               const DemGrid &reference,
                                               bool allowProjectionMismatch = false);
};

} // namespace xjw::qc
