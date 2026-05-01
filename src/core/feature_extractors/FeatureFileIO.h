// =============================================================================
// 文件: FeatureFileIO.h
// 功能: 多提取器特征文件二进制 I/O (Qt 依赖)
// 支持: SPBT/DSKB/ALKB/SFTB/ORBB/AKZB/DEDE 全部 magic bytes
// =============================================================================
#pragma once

#include <QString>
#include <string>
#include <opencv2/core.hpp>

// 前向声明
struct SuperPointOutput;

// ── 提取器类型 → 文件后缀 / Magic Bytes ──
namespace ExtractorSuffix
{
    inline const char* superpoint = ".sp";
    inline const char* disk       = ".dsk";
    inline const char* aliked     = ".alk";
    inline const char* sift       = ".sift";
    inline const char* orb        = ".orb";
    inline const char* akaze      = ".akz";
    inline const char* dedode     = ".dedode";

    inline const char* forAlgorithm(const std::string &algo)
    {
        if (algo == "superpoint") return superpoint;
        if (algo == "disk")       return disk;
        if (algo == "aliked")     return aliked;
        if (algo == "sift")       return sift;
        if (algo == "orb")        return orb;
        if (algo == "akaze")      return akaze;
        if (algo == "dedode")     return dedode;
        return ".sp";
    }
}

// ── 特征文件 I/O ──
class FeatureFileIO
{
public:
    // 写入 (自动根据 algoName 选择 magic bytes)
    static bool write(const QString& path, const QString& imageName,
                      const SuperPointOutput& output,
                      const std::string &algoName = "superpoint");

    // 读取
    static bool read(const QString& path, QString& imageName, SuperPointOutput& output);

    // 快速读取算法类型 (不加载描述子)
    static std::string peekAlgorithm(const QString& path);

    // 读取关键点数量
    static int peekCount(const QString& path);
};
