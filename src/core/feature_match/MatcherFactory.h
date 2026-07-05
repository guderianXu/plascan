// =============================================================================
// 文件: MatcherFactory.h
// 功能: 匹配器工厂 — 根据算法名创建对应匹配器
// =============================================================================
#pragma once

#include "IMatcher.h"
#include <memory>
#include <string>

struct MatcherConfig
{
    std::string modelPath;
    std::string spModelPath;       // LightGlue E2E SP模型
    float matchThreshold = 0.2f;
    int   maxKeypoints   = 2048;
    int   maxImageDim    = 2048;
    bool  useCuda        = true;
    int   cudaDevice     = 0;
};

std::unique_ptr<IMatcher> createMatcher(const std::string &algo,
                                         const MatcherConfig &cfg);
