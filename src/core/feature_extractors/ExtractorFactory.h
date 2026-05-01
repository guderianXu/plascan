// =============================================================================
// 文件: ExtractorFactory.h
// 功能: 提取器工厂 — 根据算法名创建对应提取器 (依赖具体实现)
// =============================================================================
#pragma once

#include "IExtractor.h"
#include <memory>
#include <string>

// 配置结构 (各提取器按需取用对应字段)
struct ExtractorConfig
{
    std::string modelPath;
    int  maxKeypoints   = 4096;
    float detThreshold  = 0.003f;
    int  nmsRadius      = 3;
    int  removeBorder   = 4;
    int  maxImageDim    = 2048;
    bool useCuda        = true;
    int  cudaDevice     = 0;
};

std::unique_ptr<IExtractor> createExtractor(const std::string &algo,
                                             const ExtractorConfig &cfg);
