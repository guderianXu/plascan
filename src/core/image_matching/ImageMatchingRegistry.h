#pragma once

/**
 * @file ImageMatchingRegistry.h
 * @brief 匹配算法描述符和工厂的集中注册表。
 */

#include "ImageMatchingAlgorithm.h"

#include <functional>
#include <memory>
#include <vector>

namespace xjw::image_matching
{

using ImageMatchingAlgorithmFactory =
    std::function<std::unique_ptr<IImageMatchingAlgorithm>(const ImageMatchingRuntimeConfig &)>;

/**
 * @brief 统一影像匹配算法注册表。
 *
 * 核心工作流只依赖算法 ID 和本接口，不直接包含具体 SIFT/LightGlue 头文件。
 * 新算法需要提供稳定 ID、单调递增版本和工厂；注册表负责 ID 规范化、重复
 * 注册检查及实例构造。文件格式与下游空三因此不需要随算法实现一起修改。
 */
class ImageMatchingRegistry
{
public:
    /// 注册算法描述符和工厂；同一规范化 ID 只能注册一次。
    static bool registerAlgorithm(const ImageMatchingAlgorithmDescriptor &descriptor,
                                  ImageMatchingAlgorithmFactory factory,
                                  QString *errorMessage = nullptr);

    /// 返回所有已注册算法的稳定描述符，供 GUI、CLI 和缓存版本检查使用。
    static std::vector<ImageMatchingAlgorithmDescriptor> descriptors();

    /// 判断规范化算法 ID 是否可用，不会触发模型或 CUDA 上下文初始化。
    static bool contains(const QString &algorithmId);

    /// 按运行配置创建算法实例；失败时返回空指针并给出明确原因。
    static std::unique_ptr<IImageMatchingAlgorithm> create(
        const QString &algorithmId,
        const ImageMatchingRuntimeConfig &config,
        QString *errorMessage = nullptr);
};

} // namespace xjw::image_matching
