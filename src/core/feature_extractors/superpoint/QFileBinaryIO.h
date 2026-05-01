// =============================================================================
// 文件: QFileBinaryIO.h
// 模块: SuperPoint 结果二进制 I/O
// 说明:
//   基于 Qt QFile/QDataStream 的轻量级二进制序列化工具类。
//   将 SuperPointOutput（关键点+分数+描述子）持久化到磁盘，
//   便于 GUI 程序缓存权重较大的描述子数据，避免每次重新运行模型。
//
//   依赖: Qt5/Qt6 (QFile, QDataStream), LibTorch (仅在 .cpp 中), OpenCV (cv::KeyPoint)
//   设计原则:
//     头文件只做前向声明，避免抛入 LibTorch 头文件
//     （LibTorch 的 宏定义与 Qt 宏定义存在冲突）
// =============================================================================
#ifndef QFILEBINARYIO_H
#define QFILEBINARYIO_H

#include <QString>
#include <QFile>
#include <QDataStream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Forward declarations to keep this header lightweight and avoid
// including heavy third-party headers (LibTorch) here which can
// cause macro / symbol conflicts when combined with Qt headers.
struct SuperPointOutput; // defined in SuperPoint.h

// 简要说明：
// 该类提供基于 Qt `QFile`/`QDataStream` 的简单二进制序列化，用于将 `SuperPointOutput`
// 的检测结果快速写入与读取，适合 GUI 程序在磁盘与内存之间传递结果或做缓存。
//
// 文件格式（按 QDataStream 默认小端序）：
//  - magic: 4 字节 ASCII 标识 'SPBT'（用于快速验证文件类型）
//  - version: quint32（目前为 1，便于未来格式升级）
//  - image_name_len: quint32
//  - image_name: bytes（UTF-8 编码）
//  - num_keypoints: quint32
//    对每个关键点 i 顺序写入：
//      - float x (像素坐标，列)
//      - float y (像素坐标，行)
//      - float score
//  - descriptor_dim: quint32（若为 0 表示无描述子）
//    若 descriptor_dim > 0，则按行优先写入每个关键点的 descriptor_dim 个 float 值，
//    总共写入 num_keypoints * descriptor_dim 个浮点数（对应 [N, D] 的行主序）。
//
// 兼容性与注意事项：
//  - 本格式为轻量、易用的本地二进制格式，便于 Qt 环境下读写；如果需要跨语言或长期存储，
//    可考虑使用标准化格式（例如 HDF5、NPY、Protobuf）。
//  - 写入/读取时会将描述子转换到 CPU（若在 GPU 上），因此写入文件不会包含设备信息。
//  - 读回的 `SuperPointOutput` 中 `descriptors` 为 `torch::Tensor`（float32），keypoints 使用 `cv::KeyPoint`。
//  - 由于 header 采用前向声明以避免与 LibTorch/Qt 的头冲突，本文件只在实现文件中包含
//    `SuperPoint.h` 与 LibTorch。

// ── 提取器类型 → 文件后缀映射 ──
namespace ExtractorSuffix
{
    inline const char* superpoint = ".sp";
    inline const char* disk       = ".dsk";
    inline const char* aliked     = ".alk";
    inline const char* sift       = ".sift";
    inline const char* orb        = ".orb";
    inline const char* akaze      = ".akz";

    // 通过算法名获取后缀
    inline const char* forAlgorithm(const std::string &algo)
    {
        if (algo == "superpoint") return superpoint;
        if (algo == "disk")       return disk;
        if (algo == "aliked")     return aliked;
        if (algo == "sift")       return sift;
        if (algo == "orb")        return orb;
        if (algo == "akaze")      return akaze;
        return ".sp";
    }

    // 通过后缀反查算法名
    inline std::string toAlgorithm(const std::string &suffix)
    {
        if (suffix == ".sp")   return "superpoint";
        if (suffix == ".dsk")  return "disk";
        if (suffix == ".alk")  return "aliked";
        if (suffix == ".sift") return "sift";
        if (suffix == ".orb")  return "orb";
        if (suffix == ".akz")  return "akaze";
        return "superpoint";
    }
}

// 特征文件二进制 I/O 工具类 (支持所有提取器类型)
class QFileBinaryIO
{
public:
    // 写入 (自动根据 algoName 选择 magic bytes)
    // algoName: "superpoint"/"disk"/"aliked"/"sift"/"orb"/"akaze"
    static bool write(const QString& path, const QString& image_name,
                      const SuperPointOutput& output,
                      const std::string &algoName = "superpoint");

    // 读取
    static bool read(const QString& path, QString& image_name, SuperPointOutput& output);

    // 快速读取算法类型 (不加载描述子)
    static std::string peekAlgorithm(const QString& path);

    // 快速读取关键点数量
    static int peekKeypointCount(const QString& path);
};

#endif // QFILEBINARYIO_H
