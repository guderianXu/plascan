#pragma once

#include <cstdint>
#include <vector>

namespace xjw::mesh
{

struct DepthAuxiliaryBridgeNode
{
    int frameIndex = -1;
    int refIndex = -1;
    bool primary = false;
    std::vector<int> geometrySourceIndices;
    double sparseAbsoluteDepthMedianLogError = -1.0;
    double validWithinMaskRatio = -1.0;
    double consistencyRetentionRatio = -1.0;
    double largestComponentRatio = -1.0;
    double meanConfidence = -1.0;
    int sourceViewCount = 0;
    int qualityReasonCount = 0;
    std::uint64_t trustedPixelCount = 0;
};

struct DepthAuxiliaryBridgeSelectionOptions
{
    double maximumSparseAbsoluteDepthMedianLogError = 0.02;
    double minimumValidWithinMaskRatio = 0.90;
    double minimumConsistencyRetentionRatio = 0.90;
    double minimumLargestComponentRatio = 0.95;
    double minimumMeanConfidence = 0.60;
    int minimumSourceViewCount = 2;
    std::uint64_t minimumTrustedPixelCount = 4096;
};

struct DepthAuxiliaryBridgeSelectionResult
{
    int primaryComponentCount = 0;
    std::vector<int> selectedAuxiliaryFrameIndices;
    std::vector<int> selectedAuxiliaryRefIndices;
    bool connected = false;
    bool failClosed = false;
};

class DepthAuxiliaryBridgeSelector
{
public:
    static DepthAuxiliaryBridgeSelectionResult select(
        const std::vector<DepthAuxiliaryBridgeNode> &nodes,
        const DepthAuxiliaryBridgeSelectionOptions &options = {});
};

} // namespace xjw::mesh
