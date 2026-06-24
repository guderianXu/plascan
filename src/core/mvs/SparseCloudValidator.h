#pragma once
// =============================================================================
// 文件: SparseCloudValidator.h
// 模块: MVS - 稀疏点云质量校验（轻量实现）
// =============================================================================

#include <string>
#include <vector>
#include <array>

namespace xjw {
namespace mvs {

struct SparseCloudValidatorOptions {
    int   minPoints      = 100;
    bool  filterOutliers = false;
    float outlierRadius  = 0.5f;
    int   minNeighbors   = 5;
};

struct SparseCloudStats {
    int   pointCount     = 0;
    float minDensity     = 0;
    float maxDensity     = 0;
    float meanDensity    = 0;
    std::array<float,3> minPt = {0,0,0};
    std::array<float,3> maxPt = {0,0,0};
};

class SparseCloudValidator
{
public:
    explicit SparseCloudValidator(const SparseCloudValidatorOptions &opts = SparseCloudValidatorOptions{})
        : _options(opts)
    {
    }

    /// 校验点云文件（.xyz 或 .ply），返回 true 表示质量合格
    bool validate(const std::string &cloudPath,
                  SparseCloudStats  *stats    = nullptr,
                  void              *unused   = nullptr,
                  std::string       *errorMsg = nullptr) const;

private:
    SparseCloudValidatorOptions _options;
};

} // namespace mvs
} // namespace xjw
