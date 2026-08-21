#pragma once

#include <functional>
#include <cstdint>
#include <string>

#include <QVector>

namespace xjw::mesh
{

struct MeshColorView;

enum class TextureMappingMode
{
    AutoProjective,
    KeepExistingUv
};

enum class TextureBlendMode
{
    BestView,
    Natural,
    WeightedAverage
};

enum class TextureHoleFillMode
{
    Disabled,
    TextureSpaceSmallHoles,
    NeighborViewRecovery
};

/**
 * @brief 纹理映射配置。
 */
struct TextureMappingConfig
{
    int textureSize = 8192;
    int imageDownscale = 1;
    int padding = 8;
    int maximumCandidateViews = 8;
    int maximumBlendedViews = 4;
    int labelOptimizationPasses = 8;
    int minimumChartFaces = 32;
    float minimumConfidence = 0.25f;
    float minimumViewCosine = 0.20f;
    float relativeDepthTolerance = 0.005f;
    float edgeLengthDepthTolerance = 2.0f;
    float labelSmoothness = 0.80f;
    float labelColorPenalty = 1.25f;
    float coherentReplacementRatio = 0.60f;
    float ghostColorThreshold = 36.0f;
    float sharpeningStrength = 0.35f;
    int seamBorderBlendRadiusPixels = 16;
    float seamMaximumLinearCorrection = 0.08f;
    TextureMappingMode mappingMode = TextureMappingMode::AutoProjective;
    TextureBlendMode blendMode = TextureBlendMode::Natural;
    TextureHoleFillMode holeFillMode = TextureHoleFillMode::NeighborViewRecovery;
    bool enableGhostFilter = true;
    bool enableOutOfFocusFilter = false;
    bool enableColorCorrection = false;
    bool enableSeamLeveling = true;
    bool enableFinalMeshVisibility = true;
    bool keepUnmapped = true;
    bool enableV4 = true;
    std::string blendMethod = "natural";
    std::string uvMethod = "auto_projective";
    std::function<void(const std::string &, int)> progressFn;
    std::function<bool()> isCancelled;
};

/**
 * @brief 纹理映射输出结果。
 */
struct TextureMappingResult
{
    std::string modelObjPath;
    std::string modelMtlPath;
    std::string texturePngPath;
    int textureSize = 0;
    std::string textureAlgorithm;
    std::string uvMethod;
    std::string blendMethod;
    int sourceViewCount = 0;
    int mappedFaceCount = 0;
    int fallbackMappedFaceCount = 0;
    int coherenceAdjustedFaceCount = 0;
    int unmappedFaceCount = 0;
    int strictMappedFaceCount = 0;
    int chartCount = 0;
    int usedViewCount = 0;
    std::uint64_t candidateEvaluationCount = 0;
    std::uint64_t rejectedProjectionCount = 0;
    std::uint64_t rejectedMaskCount = 0;
    std::uint64_t rejectedDepthCount = 0;
    std::uint64_t rejectedVisibilityCount = 0;
    std::uint64_t rejectedAngleCount = 0;
    std::uint64_t rejectedResolutionCount = 0;
    std::uint64_t rejectedColorOutlierCount = 0;
    std::uint64_t visibilityRasterizedPixelCount = 0;
    std::uint64_t noTexelFaceCount = 0;
    std::uint64_t centerRecoveredFaceCount = 0;
    std::uint64_t unresolvedBakeFaceCount = 0;
    std::uint64_t exposureCorrectionObservationCount = 0;
    std::uint64_t exposureCorrectionCandidatePairCount = 0;
    std::uint64_t exposureCorrectionAcceptedPairCount = 0;
    std::uint64_t exposureCorrectionRejectedInsufficientPairCount = 0;
    std::uint64_t exposureCorrectionRejectedHighMadPairCount = 0;
    double atlasOccupancy = 0.0;
    double medianTexelDensity = 0.0;
    double seamColorDifference = 0.0;
    int seamConstraintCount = 0;
    int seamAdjustedChartCount = 0;
    int seamAdjustedPixelCount = 0;
    float seamMaximumAppliedLinearCorrection = 0.0f;
    double peakMemoryEstimateMiB = 0.0;
    double exposureCorrectionMaximumAcceptedLogMad = 0.0;
    float exposureCorrectionMinimumGain = 1.0f;
    float exposureCorrectionMaximumGain = 1.0f;
    bool exposureCorrectionGraphConnected = false;
    bool exposureCorrectionApplied = false;
    std::string exposureCorrectionStatus = "disabled";
    bool cancelled = false;
};

/**
 * @brief 负责为网格生成 UV、纹理图集以及 OBJ/MTL 输出。
 */
class TextureMapper
{
public:
    /**
     * @brief 从网格文件生成带纹理模型。
     */
    static bool generateTexturedModelFromMeshFile(const std::string &meshPath,
                                                  const std::string &productsDir,
                                                  const TextureMappingConfig &config,
                                                  TextureMappingResult *result,
                                                  std::string *errorMsg = nullptr);

    static bool generateCameraTexturedModelFromMeshFile(
        const std::string &meshPath,
        const std::string &productsDir,
        const TextureMappingConfig &config,
        const QVector<MeshColorView> &views,
        TextureMappingResult *result,
        std::string *errorMsg = nullptr);
};

} // namespace xjw::mesh
