#pragma once

#include "data/PointCloud.h"

#include <string>

namespace xjw::pointcloud
{

/**
 * @brief 点云文件格式枚举。
 *
 * `Auto` 表示根据文件扩展名推断；PLY 细分为 ASCII 和小端二进制，
 * 便于调用方在写出时明确指定目标格式。
 */
enum class PointCloudFileFormat
{
    Auto,
    Obj,
    Xyz,
    PlyAscii,
    PlyBinaryLittleEndian
};

struct PointCloudReadOptions
{
    /// 指定输入格式；为 Auto 时由扩展名推断。
    PointCloudFileFormat format = PointCloudFileFormat::Auto;
};

/**
 * @brief 点云写出选项。
 *
 * 不同格式支持的属性不同：OBJ 仅写顶点和可选法向量/颜色，
 * PLY 可写位置、法向量和颜色。调用方可通过开关控制输出字段。
 */
struct PointCloudWriteOptions
{
    PointCloudFileFormat format = PointCloudFileFormat::Auto;
    bool writeNormals = true;
    bool writeColors = true;
    bool writeTextureCoordinates = true;
    bool writeFaces = true;
    std::string materialLibraryFile;
    std::string materialName = "material0";
    std::string textureImageFile;
    int precision = 6;
};

struct PointCloudIOResult
{
    /// 实际识别到的文件格式。
    PointCloudFileFormat detectedFormat = PointCloudFileFormat::Auto;

    /// 失败时的错误信息；成功时为空。
    std::string errorMessage;
};

/**
 * @brief 根据扩展名或显式格式读取点云文件。
 */
bool readPointCloud(const std::string &path,
                    PointCloud *pointCloud,
                    const PointCloudReadOptions &options = {},
                    PointCloudIOResult *result = nullptr);

/**
 * @brief 根据扩展名或显式格式写出点云文件。
 */
bool writePointCloud(const std::string &path,
                     const PointCloud &pointCloud,
                     const PointCloudWriteOptions &options = {},
                     PointCloudIOResult *result = nullptr);

/**
 * @brief 从 PLY 文件读取点云。
 *
 * 当前支持 ASCII 和 binary_little_endian 两种顶点云 PLY。
 */
bool readPlyPointCloud(const std::string &path,
                       PointCloud *pointCloud,
                       PointCloudIOResult *result = nullptr);

/**
 * @brief 将点云写出为 PLY。
 *
 * 当 `options.format` 为 `PlyBinaryLittleEndian` 时输出二进制版本，
 * 否则默认输出 ASCII PLY。
 */
bool writePlyPointCloud(const std::string &path,
                        const PointCloud &pointCloud,
                        const PointCloudWriteOptions &options = {},
                        PointCloudIOResult *result = nullptr);

/**
 * @brief 从 OBJ 顶点文件读取点云。
 *
 * 支持读取 `mtllib`、`v`、`vt`、`vn`、`f` 等常见字段。
 * 当 `f` 使用 `v/vt` 或 `v/vt/vn` 时，会为点云挂接 UV 与三角面索引。
 */
bool readObjPointCloud(const std::string &path,
                       PointCloud *pointCloud,
                       PointCloudIOResult *result = nullptr);

/**
 * @brief 将点云写出为 OBJ（支持纹理与三角面）。
 *
 * 当点云中存在 UV 和三角面且 `writeFaces=true` 时，会输出：
 * `mtllib`、`v`、`vt`、`f`（索引从 1 开始）。
 */
bool writeObjPointCloud(const std::string &path,
                        const PointCloud &pointCloud,
                        const PointCloudWriteOptions &options = {},
                        PointCloudIOResult *result = nullptr);

/**
 * @brief 从简单 XYZ 文本读取点云。
 *
 * 每行至少包含 `x y z`，其后可选 `r g b` 或 `r g b a`。
 * 颜色既兼容 `[0, 1]` 浮点，也兼容 `[0, 255]` 整数。
 * 同时兼容以逗号分隔的 `.csv/.asc/.pts` 文本点云。
 */
bool readXyzPointCloud(const std::string &path,
                       PointCloud *pointCloud,
                       PointCloudIOResult *result = nullptr);

/**
 * @brief 将点云写出为简单 XYZ 文本。
 *
 * 默认只输出坐标；当 `writeColors` 为真且点云包含颜色时，
 * 会在每行后追加 `r g b a` 四个整数字段。
 */
bool writeXyzPointCloud(const std::string &path,
                        const PointCloud &pointCloud,
                        const PointCloudWriteOptions &options = {},
                        PointCloudIOResult *result = nullptr);

/**
 * @brief 从 BA 运行摘要 JSON 中提取点云。
 *
 * 当前兼容 `ba_run_summary.json` 中的 `points` 数组，
 * 自动读取 `point_xyz`、`track_len`、`rms_after`、`valid` 等字段。
 */
bool readBaRunJsonPointCloud(const std::string &path,
                             PointCloud *pointCloud,
                             PointCloudIOResult *result = nullptr);

} // namespace xjw::pointcloud