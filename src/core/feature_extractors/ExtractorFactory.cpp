// =============================================================================
// 文件: ExtractorFactory.cpp
// 功能: 提取器工厂 — 根据算法名创建对应提取器实例
// =============================================================================
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

#include "ExtractorFactory.h"
#include "SuperPoint.h"
#include "DiskExtractor.h"
#include "AlikedExtractor.h"
#include "TraditionalFeatureExtractor.h"
#include <stdexcept>

// 传统提取器适配器：包装静态 detect() 为 IExtractor 接口
namespace
{
class TraditionalAdapter : public IExtractor
{
public:
    TraditionalAdapter(const std::string &algo, const SuperPointConfig &cfg, bool useCuda, int cudaDevice)
        : _algorithm(algo), _config(cfg), _useCuda(useCuda), _cudaDevice(cudaDevice)
    {
    }

    FeatureOutput extract(const cv::Mat &gray) override
    {
        return xjw::feature_extractors::TraditionalFeatureExtractor::detect(
            gray, _config, _algorithm, _useCuda, _cudaDevice);
    }

    std::string algorithmName() const override
    {
        return _algorithm;
    }

private:
    std::string _algorithm;
    SuperPointConfig _config;
    bool _useCuda = false;
    int _cudaDevice = 0;
};
} // anonymous

namespace xjw::feature_extractors
{

std::unique_ptr<IExtractor> createExtractor(const std::string &algo,
                                             const ExtractorConfig &cfg)
{
    std::string norm = TraditionalFeatureExtractor::normalizeAlgorithmName(algo);

    // 传统算法 → 适配器包装
    if (TraditionalFeatureExtractor::isTraditionalAlgorithm(norm))
    {
        SuperPointConfig spCfg;
        spCfg.max_num_keypoints   = cfg.maxKeypoints;
        spCfg.detection_threshold = cfg.detThreshold;
        spCfg.nms_radius          = cfg.nmsRadius;
        spCfg.remove_borders      = cfg.removeBorder;
        spCfg.grayscale_min       = cfg.grayscaleMin;
        spCfg.grayscale_max       = cfg.grayscaleMax;
        spCfg.allow_device_fallback = true;

        return std::make_unique<TraditionalAdapter>(norm, spCfg, cfg.useCuda, cfg.cudaDevice);
    }

    // 深度学习提取器
    if (norm == "superpoint")
    {
        SuperPointConfig spCfg;
        spCfg.max_num_keypoints   = cfg.maxKeypoints;
        spCfg.detection_threshold = cfg.detThreshold;
        spCfg.nms_radius          = cfg.nmsRadius;
        spCfg.remove_borders      = cfg.removeBorder;
        spCfg.grayscale_min       = cfg.grayscaleMin;
        spCfg.grayscale_max       = cfg.grayscaleMax;
        spCfg.allow_device_fallback = true;

        auto sp = std::make_unique<SuperPoint>(cfg.modelPath, spCfg);
        return sp;
    }

    if (norm == "disk")
    {
        DiskConfig dcfg;
        dcfg.modelPath    = cfg.modelPath;
        dcfg.maxKeypoints = cfg.maxKeypoints;
        dcfg.scoreThreshold = cfg.detThreshold;
        dcfg.maxImageDim  = cfg.maxImageDim;
        dcfg.grayscaleMin = cfg.grayscaleMin;
        dcfg.grayscaleMax = cfg.grayscaleMax;
        dcfg.useCuda      = cfg.useCuda;
        dcfg.cudaDevice   = cfg.cudaDevice;
        return std::make_unique<DiskExtractor>(dcfg);
    }

    if (norm == "aliked")
    {
        AlikedConfig acfg;
        acfg.modelPath    = cfg.modelPath;
        acfg.maxKeypoints = cfg.maxKeypoints;
        acfg.scoreThreshold = cfg.detThreshold;
        acfg.maxImageDim  = cfg.maxImageDim;
        acfg.grayscaleMin = cfg.grayscaleMin;
        acfg.grayscaleMax = cfg.grayscaleMax;
        acfg.useCuda      = cfg.useCuda;
        acfg.cudaDevice   = cfg.cudaDevice;
        return std::make_unique<AlikedExtractor>(acfg);
    }

    throw std::runtime_error("unsupported extractor: " + algo);
}

} // namespace xjw::feature_extractors

#ifdef _MSC_VER
#pragma warning(pop)
#endif
