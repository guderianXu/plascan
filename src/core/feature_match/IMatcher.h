// =============================================================================
// 文件: IMatcher.h
// 功能: 特征匹配器统一接口
// =============================================================================
#pragma once

#include <string>

struct IMatcher
{
    virtual ~IMatcher() = default;

    // 返回匹配点数, 失败返回 -1
    // sp1/sp2: 特征文件路径 (稀疏匹配)
    // imgL/imgR: 影像路径 (密集匹配, 如 LoFTR)
    // outPath: 输出 .match 文件路径
    virtual int match(const std::string &sp1, const std::string &sp2,
                      const std::string &imgL, const std::string &imgR,
                      const std::string &outPath) = 0;

    virtual std::string algorithmName() const = 0;
    virtual bool needsFeatureFiles() const { return true; }
};
