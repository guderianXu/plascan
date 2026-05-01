// =============================================================================
// 文件: IExtractor.h
// 功能: 特征提取器统一接口 (所有提取器实现此接口)
// =============================================================================
#pragma once

#include "FeatureOutput.h"
#include <string>

struct IExtractor
{
    virtual ~IExtractor() = default;
    virtual FeatureOutput extract(const cv::Mat &grayImage) = 0;
    virtual std::string   algorithmName() const = 0;
};
