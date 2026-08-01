// ============================================================
// 文件：CameraFormatConverter.cpp
// 功能：把 Middlebury、EPFL/Strecha、COLMAP text 和 Metashape 工程
//       统一转换为 PlaScan 可消费的 Tsai 相机数据集。
//
// 转换分为五步：识别格式 -> 解析为 CameraRecord -> 定位影像目录 ->
// 写出 cameras/*.tsai 与 image_camera.lis -> 汇总 summary.json。
// 不能无损表达的外部参数进入 warnings；输入/写出异常由公开入口捕获并
// 写入 CameraConversionResult，而内部辅助函数使用异常保持失败上下文。
// ============================================================

#include "CameraFormatConverter.h"
#include "io/PathIO.h"
#include "string_utils/StringTransform.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <zip.h>

namespace xjw::camera
{
using common::string_utils::asciiLowerCopy;
using common::string_utils::trimAsciiWhitespace;

namespace
{

// ---------- 跨格式统一中间表示 ----------

/**
 * @brief 一张影像及其相机参数的规范化记录。
 *
 * `k` 是行优先 3×3 像素内参矩阵；`distortion` 顺序固定为
 * `[k1, k2, k3, p1, p2]`。外参统一为 camera-to-world 旋转和世界系
 * 相机中心，使所有输入格式最终都能直接映射到 PlaScan Tsai 语义。
 */
struct CameraRecord
{
    /// 相对于最终 `sourceDir` 的影像名或子路径。
    std::string imageName;
    /// 输出到 `cameras/` 下的目标 Tsai 文件名，通常由影像 stem 派生。
    std::string cameraFileName;
    std::array<double, 9> k{{0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0}};
    std::array<double, 5> distortion{{0.0, 0.0, 0.0, 0.0, 0.0}};
    std::array<double, 9> rotationCameraToWorld{{1.0, 0.0, 0.0,
                                                 0.0, 1.0, 0.0,
                                                 0.0, 0.0, 1.0}};
    std::array<double, 3> center{{0.0, 0.0, 0.0}};
    /// 该记录在有损转换中无法表达的信息，最终会加上影像名前缀汇总。
    std::vector<std::string> warnings;
};

/// COLMAP 的 cameras.txt 只定义共享内参，因此先按 CAMERA_ID 独立缓存。
struct ColmapCameraModel
{
    std::array<double, 9> k{{0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0}};
    std::vector<std::string> warnings;
};

/// Metashape chunk 既可能是展开的 doc.xml，也可能封装在 chunk.zip 中。
struct MetashapeProject
{
    std::filesystem::path docPath;
    std::filesystem::path chunkZipPath;
    std::filesystem::path searchRoot;
};

/// Metashape sensor 保存可被多个 camera 实例复用的标定参数。
struct MetashapeSensor
{
    std::array<double, 9> k{{0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0}};
    std::array<double, 5> distortion{{0.0, 0.0, 0.0, 0.0, 0.0}};
    std::vector<std::string> warnings;
};

// ---------- 文本规范化与输出转义 ----------

std::string normalizedFormatName(std::string name)
{
    name = asciiLowerCopy(trimAsciiWhitespace(name));
    std::replace(name.begin(), name.end(), '_', '-');
    return name;
}

std::string formatNumber(double value)
{
    // 把舍入后接近零的值稳定写为 0，避免 summary/Tsai 出现难读的 -0 或微小噪声。
    if (std::abs(value) < 1e-14)
    {
        value = 0.0;
    }
    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

std::string jsonEscape(const std::string &value)
{
    // summary 只写字符串标量，因此覆盖 JSON 字符串中必须处理的常见转义字符。
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value)
    {
        switch (ch)
        {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped.push_back(ch);
            break;
        }
    }
    return escaped;
}

std::string shellQuote(const std::string &value)
{
    // image_camera.lis 采用空白分隔 token；含空格或引号的路径按 POSIX shell
    // 单引号规则保护，简单路径保持原样以提高人工可读性。
    if (value.empty())
    {
        return "''";
    }

    const bool simple = std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isalnum(ch) != 0 || ch == '_' || ch == '-' || ch == '.' || ch == '/' || ch == ':';
    });
    if (simple)
    {
        return value;
    }

    std::string quoted = "'";
    for (const char ch : value)
    {
        if (ch == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

// ---------- 小型线性代数辅助函数（全部为行优先 3×3） ----------

std::array<double, 9> transpose(const std::array<double, 9> &matrix)
{
    return std::array<double, 9>{{
        matrix[0], matrix[3], matrix[6],
        matrix[1], matrix[4], matrix[7],
        matrix[2], matrix[5], matrix[8]
    }};
}

std::array<double, 3> matVecMul(const std::array<double, 9> &matrix,
                                const std::array<double, 3> &vector)
{
    return std::array<double, 3>{{
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2]
    }};
}

std::array<double, 9> colmapQvecToWorldToCameraRotation(double qw, double qx, double qy, double qz)
{
    // COLMAP images.txt 的 qvec 表示 world-to-camera。先归一化可消除
    // 文本精度或上游计算带来的轻微单位四元数误差。
    const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (norm <= 0.0)
    {
        throw std::runtime_error("COLMAP images.txt 中存在无效四元数");
    }
    qw /= norm;
    qx /= norm;
    qy /= norm;
    qz /= norm;

    return std::array<double, 9>{{
        1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
        2.0 * qx * qy - 2.0 * qz * qw,
        2.0 * qx * qz + 2.0 * qy * qw,
        2.0 * qx * qy + 2.0 * qz * qw,
        1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
        2.0 * qy * qz - 2.0 * qx * qw,
        2.0 * qx * qz - 2.0 * qy * qw,
        2.0 * qy * qz + 2.0 * qx * qw,
        1.0 - 2.0 * qx * qx - 2.0 * qy * qy
    }};
}

// ---------- 通用文件、ZIP 与轻量 XML 读取 ----------

std::vector<std::string> readDataLines(const std::filesystem::path &path)
{
    // Middlebury/EPFL/COLMAP 的数据行均允许空行和 # 注释，在解析器前统一过滤。
    std::ifstream in = xjw::common::io::openInputFile(path, std::ios::in);
    if (!in)
    {
        throw std::runtime_error("无法打开相机文件: " + path.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
    {
        line = trimAsciiWhitespace(line);
        if (line.empty() || line.rfind("#", 0) == 0)
        {
            continue;
        }
        lines.push_back(line);
    }
    return lines;
}

std::vector<double> parseNumericLine(const std::string &line, int expectedCount)
{
    // expectedCount < 0 表示接受可变长度；其它情况严格校验字段数量，
    // 使格式错误在接近来源的位置失败。
    std::istringstream in(line);
    std::vector<double> values;
    double value = 0.0;
    while (in >> value)
    {
        values.push_back(value);
    }
    if (expectedCount >= 0 && static_cast<int>(values.size()) != expectedCount)
    {
        throw std::runtime_error("数值行字段数量错误: " + line);
    }
    return values;
}

std::string readTextFile(const std::filesystem::path &path)
{
    std::ifstream in = xjw::common::io::openInputFile(path);
    if (!in)
    {
        throw std::runtime_error("无法打开文本文件: " + path.string());
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string zipErrorMessage(zip_error_t *zipError)
{
    const char *message = zipError ? zip_error_strerror(zipError) : nullptr;
    return message ? std::string(message) : std::string("未知 libzip 错误");
}

zip_t *openZipArchive(const std::filesystem::path &zipPath, std::string *error)
{
#ifdef _WIN32
    // Windows 使用宽字符 source，避免含中文或其它 Unicode 字符的工程路径
    // 在 libzip 的窄字符入口发生本地代码页损失。
    zip_error_t zipError;
    zip_error_init(&zipError);
    zip_source_t *source = zip_source_win32w_create(zipPath.wstring().c_str(), 0, -1, &zipError);
    if (!source)
    {
        if (error)
        {
            *error = zipErrorMessage(&zipError);
        }
        zip_error_fini(&zipError);
        return nullptr;
    }

    zip_t *archive = zip_open_from_source(source, ZIP_RDONLY, &zipError);
    if (!archive)
    {
        if (error)
        {
            *error = zipErrorMessage(&zipError);
        }
        zip_source_free(source);
        zip_error_fini(&zipError);
        return nullptr;
    }

    zip_error_fini(&zipError);
    return archive;
#else
    int errorCode = 0;
    const std::string zipPathUtf8 = xjw::common::io::toUtf8Path(zipPath);
    zip_t *archive = zip_open(zipPathUtf8.c_str(), ZIP_RDONLY, &errorCode);
    if (!archive && error)
    {
        zip_error_t zipError;
        zip_error_init_with_code(&zipError, errorCode);
        *error = zipErrorMessage(&zipError);
        zip_error_fini(&zipError);
    }
    return archive;
#endif
}

std::string readZipEntryText(const std::filesystem::path &zipPath,
                             const std::string &entryPath)
{
    // archive/file 两层句柄在每条失败路径都显式关闭，避免转换批次泄漏资源。
    std::string openError;
    zip_t *archive = openZipArchive(zipPath, &openError);
    if (!archive)
    {
        throw std::runtime_error("无法打开 Metashape zip 文件: " + zipPath.string() + ": " + openError);
    }

    zip_file_t *file = zip_fopen(archive, entryPath.c_str(), 0);
    if (!file)
    {
        const std::string error = zip_strerror(archive);
        zip_close(archive);
        throw std::runtime_error("Metashape zip 中缺少 " + entryPath + ": " + error);
    }

    std::string text;
    std::array<char, 16384> buffer{};
    while (true)
    {
        const zip_int64_t bytesRead = zip_fread(file, buffer.data(), buffer.size());
        if (bytesRead < 0)
        {
            const std::string error = zip_file_strerror(file);
            zip_fclose(file);
            zip_close(archive);
            throw std::runtime_error("读取 Metashape zip 条目失败: " + error);
        }
        if (bytesRead == 0)
        {
            break;
        }
        text.append(buffer.data(), static_cast<size_t>(bytesRead));
    }

    zip_fclose(file);
    zip_close(archive);
    return text;
}

std::vector<std::string> xmlBlocks(const std::string &xml,
                                   const std::string &tag)
{
    // 这里只处理 Metashape doc.xml 中结构稳定的小片段，不试图实现通用 XML
    // 解析器；`[\s\S]` 允许块内容跨行。
    std::vector<std::string> blocks;
    const std::regex re("<" + tag + "\\b[^>]*>[\\s\\S]*?</" + tag + ">");
    for (auto it = std::sregex_iterator(xml.begin(), xml.end(), re);
         it != std::sregex_iterator(); ++it)
    {
        blocks.push_back(it->str());
    }
    return blocks;
}

std::optional<std::string> xmlElementText(const std::string &xml,
                                          const std::string &tag)
{
    const std::regex re("<" + tag + "\\b[^>]*>([\\s\\S]*?)</" + tag + ">");
    std::smatch match;
    if (!std::regex_search(xml, match, re) || match.size() < 2)
    {
        return std::nullopt;
    }
    return trimAsciiWhitespace(match[1].str());
}

std::optional<std::string> xmlAttribute(const std::string &xml,
                                        const std::string &name)
{
    const std::regex re(name + "\\s*=\\s*[\"']([^\"']*)[\"']");
    std::smatch match;
    if (!std::regex_search(xml, match, re) || match.size() < 2)
    {
        return std::nullopt;
    }
    return match[1].str();
}

double xmlDoubleOr(const std::string &xml,
                   const std::string &tag,
                   double fallback)
{
    const auto value = xmlElementText(xml, tag);
    if (!value.has_value() || value->empty())
    {
        return fallback;
    }
    return std::stod(*value);
}

bool xmlHasNonZeroTags(const std::string &xml,
                       const std::vector<std::string> &tags)
{
    for (const std::string &tag : tags)
    {
        const auto value = xmlElementText(xml, tag);
        if (value.has_value() && std::abs(std::stod(*value)) > 1e-12)
        {
            return true;
        }
    }
    return false;
}

void warnIfUnsupportedSkew(const std::array<double, 9> &k,
                           std::vector<std::string> *warnings)
{
    if (!warnings)
    {
        return;
    }
    if (std::abs(k[1]) > 1e-12 || std::abs(k[3]) > 1e-12)
    {
        // PlaScan Tsai 输出只写 fx/fy/cx/cy，非零 K01/K10 不能静默丢失。
        warnings->push_back("camera skew terms are not represented in PlaScan tsai output "
                            "(k01=" + formatNumber(k[1]) + ", k10=" + formatNumber(k[3]) + ")");
    }
}

// ---------- Middlebury 与 EPFL/Strecha 解析 ----------

std::string cameraFileNameFromStem(const std::string &imageName)
{
    std::filesystem::path imagePath(imageName);
    return imagePath.stem().string() + ".tsai";
}

CameraRecord parseMiddleburyCameraLine(const std::string &line)
{
    // Middlebury 每行布局为 imageName + K(9) + R_wc(9) + t_wc(3)。
    std::istringstream in(line);
    CameraRecord record;
    if (!(in >> record.imageName))
    {
        throw std::runtime_error("Middlebury 相机行缺少影像名: " + line);
    }

    std::vector<double> values;
    double value = 0.0;
    while (in >> value)
    {
        values.push_back(value);
    }
    if (values.size() != 21)
    {
        throw std::runtime_error("Middlebury 相机行必须包含影像名和 21 个数值: " + line);
    }

    std::copy(values.begin(), values.begin() + 9, record.k.begin());
    std::array<double, 9> rotationWorldToCamera{};
    std::copy(values.begin() + 9, values.begin() + 18, rotationWorldToCamera.begin());
    const std::array<double, 3> translation{{
        values[18], values[19], values[20]
    }};

    // 输入外参满足 Xc = R_wc*Xw + t_wc，因此输出所需的
    // R_cw = R_wc^T，世界系相机中心 C = -R_cw*t_wc。
    record.rotationCameraToWorld = transpose(rotationWorldToCamera);
    const auto centerRaw = matVecMul(record.rotationCameraToWorld, translation);
    record.center = std::array<double, 3>{{
        -centerRaw[0], -centerRaw[1], -centerRaw[2]
    }};
    record.cameraFileName = cameraFileNameFromStem(record.imageName);
    warnIfUnsupportedSkew(record.k, &record.warnings);
    return record;
}

std::vector<CameraRecord> parseMiddleburyPar(const std::filesystem::path &path)
{
    std::vector<std::string> lines = readDataLines(path);
    if (!lines.empty())
    {
        std::istringstream first(lines.front());
        int ignoredCount = 0;
        std::string trailing;
        if ((first >> ignoredCount) && !(first >> trailing))
        {
            // 官方 par 文件首行可仅包含相机数量；实际记录数仍以成功解析的行数为准。
            lines.erase(lines.begin());
        }
    }
    if (lines.empty())
    {
        throw std::runtime_error("Middlebury par 文件中没有相机记录: " + path.string());
    }

    std::vector<CameraRecord> records;
    records.reserve(lines.size());
    for (const std::string &line : lines)
    {
        records.push_back(parseMiddleburyCameraLine(line));
    }
    return records;
}

CameraRecord parseEpflCameraFile(const std::filesystem::path &path)
{
    // EPFL/Strecha .camera 已直接给出 K、camera-to-world R 和世界系中心 C，
    // 无需像 Middlebury/COLMAP 那样从 t_wc 反求中心。
    const std::vector<std::string> lines = readDataLines(path);
    if (lines.size() < 8)
    {
        throw std::runtime_error("EPFL .camera 文件至少需要 8 行数值: " + path.string());
    }

    CameraRecord record;
    const auto k0 = parseNumericLine(lines[0], 3);
    const auto k1 = parseNumericLine(lines[1], 3);
    const auto k2 = parseNumericLine(lines[2], 3);
    record.k = std::array<double, 9>{{
        k0[0], k0[1], k0[2],
        k1[0], k1[1], k1[2],
        k2[0], k2[1], k2[2]
    }};

    const auto radial = parseNumericLine(lines[3], 3);
    if (std::any_of(radial.begin(), radial.end(), [](double value) { return std::abs(value) > 1e-12; }))
    {
        // 该三元素 EPFL 径向模型与 PlaScan 当前五参数约定没有可靠的一一映射，
        // 因而保留告警而不猜测系数含义。
        record.warnings.push_back("radial distortion terms are not exported to PlaScan tsai output");
    }

    const auto r0 = parseNumericLine(lines[4], 3);
    const auto r1 = parseNumericLine(lines[5], 3);
    const auto r2 = parseNumericLine(lines[6], 3);
    record.rotationCameraToWorld = std::array<double, 9>{{
        r0[0], r0[1], r0[2],
        r1[0], r1[1], r1[2],
        r2[0], r2[1], r2[2]
    }};

    const auto c = parseNumericLine(lines[7], 3);
    record.center = std::array<double, 3>{{c[0], c[1], c[2]}};

    const std::string cameraName = path.filename().string();
    const std::string suffix = ".camera";
    if (cameraName.size() > suffix.size()
        && cameraName.compare(cameraName.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
        record.imageName = cameraName.substr(0, cameraName.size() - suffix.size());
    }
    else
    {
        record.imageName = path.stem().string();
    }
    record.cameraFileName = record.imageName + ".tsai";
    warnIfUnsupportedSkew(record.k, &record.warnings);
    return record;
}

// ---------- COLMAP text 解析 ----------

void warnIfColmapDistortion(const std::string &model,
                            const std::vector<double> &params,
                            size_t firstDistortionIndex,
                            std::vector<std::string> *warnings)
{
    if (!warnings || firstDistortionIndex >= params.size())
    {
        return;
    }

    const bool hasDistortion = std::any_of(params.begin() + static_cast<std::ptrdiff_t>(firstDistortionIndex),
                                           params.end(),
                                           [](double value) { return std::abs(value) > 1e-12; });
    if (hasDistortion)
    {
        // COLMAP 的相机模型族包含多种径向/切向/鱼眼参数排列；当前转换器
        // 只保留兼容的针孔 K，明确报告其余非零项以避免误认为无损转换。
        warnings->push_back("COLMAP " + model + " distortion terms are not exported to PlaScan tsai output");
    }
}

ColmapCameraModel parseColmapCameraLine(const std::string &line)
{
    // cameras.txt 行格式：CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]。
    // width/height 用于验证语法但不写入当前 Tsai 输出。
    std::istringstream in(line);
    int cameraId = 0;
    std::string model;
    int width = 0;
    int height = 0;
    if (!(in >> cameraId >> model >> width >> height))
    {
        throw std::runtime_error("COLMAP cameras.txt 相机行格式错误: " + line);
    }

    std::vector<double> params;
    double value = 0.0;
    while (in >> value)
    {
        params.push_back(value);
    }

    ColmapCameraModel camera;
    // 每个分支按 COLMAP 官方参数顺序提取 fx/fy/cx/cy；简单模型共享单焦距。
    if (model == "SIMPLE_PINHOLE")
    {
        if (params.size() != 3)
        {
            throw std::runtime_error("COLMAP SIMPLE_PINHOLE 参数数量错误: " + line);
        }
        camera.k = std::array<double, 9>{{params[0], 0.0, params[1], 0.0, params[0], params[2], 0.0, 0.0, 1.0}};
    }
    else if (model == "PINHOLE")
    {
        if (params.size() != 4)
        {
            throw std::runtime_error("COLMAP PINHOLE 参数数量错误: " + line);
        }
        camera.k = std::array<double, 9>{{params[0], 0.0, params[2], 0.0, params[1], params[3], 0.0, 0.0, 1.0}};
    }
    else if (model == "SIMPLE_RADIAL" || model == "SIMPLE_RADIAL_FISHEYE")
    {
        if (params.size() != 4)
        {
            throw std::runtime_error("COLMAP " + model + " 参数数量错误: " + line);
        }
        camera.k = std::array<double, 9>{{params[0], 0.0, params[1], 0.0, params[0], params[2], 0.0, 0.0, 1.0}};
        warnIfColmapDistortion(model, params, 3, &camera.warnings);
    }
    else if (model == "RADIAL" || model == "RADIAL_FISHEYE")
    {
        if (params.size() != 5)
        {
            throw std::runtime_error("COLMAP " + model + " 参数数量错误: " + line);
        }
        camera.k = std::array<double, 9>{{params[0], 0.0, params[1], 0.0, params[0], params[2], 0.0, 0.0, 1.0}};
        warnIfColmapDistortion(model, params, 3, &camera.warnings);
    }
    else if (model == "OPENCV" || model == "OPENCV_FISHEYE")
    {
        if (params.size() != 8)
        {
            throw std::runtime_error("COLMAP " + model + " 参数数量错误: " + line);
        }
        camera.k = std::array<double, 9>{{params[0], 0.0, params[2], 0.0, params[1], params[3], 0.0, 0.0, 1.0}};
        warnIfColmapDistortion(model, params, 4, &camera.warnings);
    }
    else if (model == "FULL_OPENCV")
    {
        if (params.size() != 12)
        {
            throw std::runtime_error("COLMAP " + model + " 参数数量错误: " + line);
        }
        camera.k = std::array<double, 9>{{params[0], 0.0, params[2], 0.0, params[1], params[3], 0.0, 0.0, 1.0}};
        warnIfColmapDistortion(model, params, 4, &camera.warnings);
    }
    else
    {
        throw std::runtime_error("暂不支持的 COLMAP 相机模型: " + model);
    }

    (void)cameraId;
    (void)width;
    (void)height;
    return camera;
}

std::unordered_map<int, ColmapCameraModel> parseColmapCameras(const std::filesystem::path &path)
{
    // CAMERA_ID 是 images.txt 关联共享内参的唯一键。
    std::unordered_map<int, ColmapCameraModel> cameras;
    for (const std::string &line : readDataLines(path))
    {
        std::istringstream in(line);
        int cameraId = 0;
        if (!(in >> cameraId))
        {
            throw std::runtime_error("COLMAP cameras.txt 相机 ID 格式错误: " + line);
        }
        cameras.emplace(cameraId, parseColmapCameraLine(line));
    }
    if (cameras.empty())
    {
        throw std::runtime_error("COLMAP cameras.txt 中没有相机记录: " + path.string());
    }
    return cameras;
}

CameraRecord parseColmapImageLine(const std::string &line,
                                  const std::unordered_map<int, ColmapCameraModel> &cameras)
{
    // images.txt 头行格式：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME。
    std::istringstream in(line);
    int imageId = 0;
    double qw = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    std::array<double, 3> t{{0.0, 0.0, 0.0}};
    int cameraId = 0;
    if (!(in >> imageId >> qw >> qx >> qy >> qz >> t[0] >> t[1] >> t[2] >> cameraId))
    {
        throw std::runtime_error("COLMAP images.txt 影像行格式错误: " + line);
    }

    std::string imageName;
    std::getline(in, imageName);
    imageName = trimAsciiWhitespace(imageName);
    if (imageName.empty())
    {
        throw std::runtime_error("COLMAP images.txt 影像行缺少影像名: " + line);
    }

    const auto cameraIt = cameras.find(cameraId);
    if (cameraIt == cameras.end())
    {
        throw std::runtime_error("COLMAP images.txt 引用了不存在的 CAMERA_ID: " + std::to_string(cameraId));
    }

    CameraRecord record;
    record.imageName = imageName;
    record.k = cameraIt->second.k;
    record.warnings = cameraIt->second.warnings;

    // COLMAP 位姿满足 Xc = R_wc*Xw+t_wc；PlaScan 需要 R_cw 和 C，故：
    // R_cw = R_wc^T，C = -R_cw*t_wc。
    const std::array<double, 9> rotationWorldToCamera = colmapQvecToWorldToCameraRotation(qw, qx, qy, qz);
    record.rotationCameraToWorld = transpose(rotationWorldToCamera);
    const auto centerRaw = matVecMul(record.rotationCameraToWorld, t);
    record.center = std::array<double, 3>{{-centerRaw[0], -centerRaw[1], -centerRaw[2]}};
    record.cameraFileName = cameraFileNameFromStem(record.imageName);
    warnIfUnsupportedSkew(record.k, &record.warnings);
    (void)imageId;
    return record;
}

std::vector<CameraRecord> parseColmapImages(const std::filesystem::path &path,
                                            const std::unordered_map<int, ColmapCameraModel> &cameras)
{
    std::ifstream in = xjw::common::io::openInputFile(path, std::ios::in);
    if (!in)
    {
        throw std::runtime_error("无法打开相机文件: " + path.string());
    }

    std::vector<CameraRecord> records;
    std::string line;
    int lineNumber = 0;
    while (std::getline(in, line))
    {
        ++lineNumber;
        const std::string header = trimAsciiWhitespace(line);
        if (header.empty() || header.rfind("#", 0) == 0)
        {
            continue;
        }

        records.push_back(parseColmapImageLine(header, cameras));

        // COLMAP 每个影像头行后固定跟一行 POINTS2D。转换只需外参，
        // 但仍消费该行以保持下一次循环落在正确记录边界。
        if (!std::getline(in, line))
        {
            throw std::runtime_error("COLMAP images.txt 第 " + std::to_string(lineNumber)
                                     + " 行影像记录缺少 POINTS2D 行: " + path.string());
        }
        ++lineNumber;
    }

    if (records.empty())
    {
        throw std::runtime_error("COLMAP images.txt 中没有影像记录: " + path.string());
    }
    return records;
}

void sortRecordsByImageName(std::vector<CameraRecord> *records)
{
    // 稳定、大小写不敏感排序使 image_camera.lis 与文件系统遍历顺序无关，
    // 同时为相同规范名保留输入顺序。
    std::stable_sort(records->begin(), records->end(), [](const CameraRecord &lhs, const CameraRecord &rhs) {
        return asciiLowerCopy(std::filesystem::path(lhs.imageName).generic_string())
            < asciiLowerCopy(std::filesystem::path(rhs.imageName).generic_string());
    });
}

// ---------- Middlebury、EPFL 与 COLMAP 来源发现 ----------

bool hasExtension(const std::filesystem::path &path, const std::string &extension)
{
    return normalizedFormatName(path.extension().string()) == normalizedFormatName(extension);
}

std::vector<std::filesystem::path> findFiles(const std::filesystem::path &directory,
                                             const std::string &suffix)
{
    std::vector<std::filesystem::path> matches;
    if (!std::filesystem::exists(directory))
    {
        return matches;
    }
    for (const auto &entry : std::filesystem::directory_iterator(directory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.size() >= suffix.size()
            && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
        {
            matches.push_back(entry.path());
        }
    }
    std::sort(matches.begin(), matches.end());
    return matches;
}

std::filesystem::path findMiddleburyParFile(const std::filesystem::path &inputPath)
{
    // 允许用户直接指定唯一主文件；目录模式在多候选时拒绝猜测。
    if (std::filesystem::is_regular_file(inputPath))
    {
        const std::string name = inputPath.filename().string();
        if (name.size() >= 8 && name.find("_par.txt") != std::string::npos)
        {
            return inputPath;
        }
        throw std::runtime_error("指定文件不是 Middlebury *_par.txt: " + inputPath.string());
    }

    const auto matches = findFiles(inputPath, "_par.txt");
    if (matches.empty())
    {
        throw std::runtime_error("目录下未找到 Middlebury *_par.txt: " + inputPath.string());
    }
    if (matches.size() > 1)
    {
        throw std::runtime_error("目录下存在多个 Middlebury *_par.txt，请直接指定其中一个文件: "
                                 + inputPath.string());
    }
    return matches.front();
}

std::vector<std::filesystem::path> findEpflCameraFiles(const std::filesystem::path &inputPath)
{
    if (std::filesystem::is_regular_file(inputPath))
    {
        if (!hasExtension(inputPath, ".camera"))
        {
            throw std::runtime_error("指定文件不是 EPFL .camera: " + inputPath.string());
        }
        return {inputPath};
    }

    const auto matches = findFiles(inputPath, ".camera");
    if (matches.empty())
    {
        throw std::runtime_error("目录下未找到 EPFL .camera 文件: " + inputPath.string());
    }
    return matches;
}

bool isColmapTextDirectory(const std::filesystem::path &path)
{
    return std::filesystem::is_directory(path)
        && std::filesystem::exists(path / "cameras.txt")
        && std::filesystem::exists(path / "images.txt");
}

std::filesystem::path findColmapTextDirectory(const std::filesystem::path &inputPath)
{
    // 兼容常见 COLMAP 布局：数据集根/sparse、sparse/0，以及用户直接传入 0 目录。
    if (std::filesystem::is_regular_file(inputPath))
    {
        const std::filesystem::path parent = inputPath.parent_path();
        if (isColmapTextDirectory(parent))
        {
            return parent;
        }
        throw std::runtime_error("指定文件所在目录不是 COLMAP text sparse 目录: " + inputPath.string());
    }

    if (isColmapTextDirectory(inputPath))
    {
        return inputPath;
    }
    if (isColmapTextDirectory(inputPath / "sparse"))
    {
        return inputPath / "sparse";
    }
    if (isColmapTextDirectory(inputPath / "sparse" / "0"))
    {
        return inputPath / "sparse" / "0";
    }
    if (isColmapTextDirectory(inputPath / "0"))
    {
        return inputPath / "0";
    }
    throw std::runtime_error("目录下未找到 COLMAP text cameras.txt/images.txt: " + inputPath.string());
}

std::filesystem::path findColmapImageRoot(const std::filesystem::path &inputPath,
                                          const std::filesystem::path &colmapDir,
                                          const std::vector<CameraRecord> &records)
{
    // 候选覆盖“输入根/images”、sparse 的一至两级父目录/images 和 sparse
    // 相邻目录。只有包含全部记录影像的目录才会被接受，避免生成部分有效列表。
    std::vector<std::filesystem::path> candidates;
    if (std::filesystem::is_directory(inputPath))
    {
        candidates.push_back(inputPath / "images");
    }
    candidates.push_back(colmapDir.parent_path() / "images");
    candidates.push_back(colmapDir.parent_path().parent_path() / "images");
    candidates.push_back(colmapDir);
    candidates.push_back(colmapDir.parent_path());

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    // 排序去重保证不同输入入口得到确定的搜索次序。
    for (const std::filesystem::path &candidate : candidates)
    {
        if (!std::filesystem::is_directory(candidate))
        {
            continue;
        }
        const bool allImagesFound = std::all_of(records.begin(), records.end(), [&](const CameraRecord &record) {
            return std::filesystem::exists(candidate / record.imageName);
        });
        if (allImagesFound)
        {
            return candidate;
        }
    }

    throw std::runtime_error("未找到 COLMAP 影像目录，已检查 sparse 目录相邻 images 目录");
}

// ---------- Metashape 工程发现与 XML 标定解析 ----------

bool isImageFile(const std::filesystem::path &path)
{
    const std::string ext = asciiLowerCopy(path.extension().string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".tif" || ext == ".tiff" || ext == ".ppm";
}

std::filesystem::path metashapeSearchRootFrom(const std::filesystem::path &path)
{
    // 从 Project.files/0 或 Metashape 目录回退到数据集根，后续才能找到
    // 常见的同级 Depthimages/Images 影像目录。
    if (path.filename() == "0" && path.parent_path().extension() == ".files")
    {
        const std::filesystem::path metashapeDir = path.parent_path().parent_path();
        if (asciiLowerCopy(metashapeDir.filename().string()) == "metashape")
        {
            return metashapeDir.parent_path();
        }
        return metashapeDir;
    }

    if (asciiLowerCopy(path.filename().string()) == "metashape")
    {
        return path.parent_path();
    }
    return path;
}

std::optional<MetashapeProject> makeMetashapeProjectFromFilesDir(const std::filesystem::path &filesDir,
                                                                  const std::filesystem::path &searchRoot)
{
    // Metashape 项目保存后可能保留展开 XML，也可能只保留压缩 chunk；二者语义等价。
    const std::filesystem::path chunkDir = filesDir / "0";
    if (std::filesystem::exists(chunkDir / "doc.xml"))
    {
        return MetashapeProject{chunkDir / "doc.xml", {}, searchRoot};
    }
    if (std::filesystem::exists(chunkDir / "chunk.zip"))
    {
        return MetashapeProject{{}, chunkDir / "chunk.zip", searchRoot};
    }
    return std::nullopt;
}

std::optional<MetashapeProject> findMetashapeProject(const std::filesystem::path &inputPath)
{
    // 接受 doc.xml、chunk.zip、.psx 主文件以及数据集/Metashape 目录，
    // 使 GUI 选择不同层级路径时仍能定位同一 chunk。
    if (std::filesystem::is_regular_file(inputPath))
    {
        const std::string fileName = inputPath.filename().string();
        if (fileName == "doc.xml")
        {
            return MetashapeProject{inputPath, {}, metashapeSearchRootFrom(inputPath.parent_path())};
        }
        if (fileName == "chunk.zip")
        {
            return MetashapeProject{{}, inputPath, metashapeSearchRootFrom(inputPath.parent_path())};
        }
        if (hasExtension(inputPath, ".psx"))
        {
            const std::filesystem::path filesDir =
                inputPath.parent_path() / (inputPath.stem().string() + ".files");
            return makeMetashapeProjectFromFilesDir(filesDir,
                                                    metashapeSearchRootFrom(inputPath.parent_path()));
        }
        return std::nullopt;
    }

    if (!std::filesystem::is_directory(inputPath))
    {
        return std::nullopt;
    }

    std::vector<std::filesystem::path> dirs{inputPath};
    if (std::filesystem::is_directory(inputPath / "Metashape"))
    {
        dirs.push_back(inputPath / "Metashape");
    }

    for (const std::filesystem::path &dir : dirs)
    {
        if (std::filesystem::exists(dir / "doc.xml"))
        {
            return MetashapeProject{dir / "doc.xml", {}, metashapeSearchRootFrom(dir)};
        }
        if (std::filesystem::exists(dir / "chunk.zip"))
        {
            return MetashapeProject{{}, dir / "chunk.zip", metashapeSearchRootFrom(dir)};
        }
        if (std::filesystem::exists(dir / "0" / "doc.xml"))
        {
            return MetashapeProject{dir / "0" / "doc.xml", {}, metashapeSearchRootFrom(dir / "0")};
        }
        if (std::filesystem::exists(dir / "0" / "chunk.zip"))
        {
            return MetashapeProject{{}, dir / "0" / "chunk.zip", metashapeSearchRootFrom(dir / "0")};
        }

        for (const auto &entry : std::filesystem::directory_iterator(dir))
        {
            const std::filesystem::path path = entry.path();
            if (entry.is_regular_file() && hasExtension(path, ".psx"))
            {
                const auto project =
                    makeMetashapeProjectFromFilesDir(path.parent_path() / (path.stem().string() + ".files"),
                                                     metashapeSearchRootFrom(path.parent_path()));
                if (project.has_value())
                {
                    return project;
                }
            }
            else if (entry.is_directory() && path.extension() == ".files")
            {
                const auto project = makeMetashapeProjectFromFilesDir(path, metashapeSearchRootFrom(dir));
                if (project.has_value())
                {
                    return project;
                }
            }
        }
    }

    return std::nullopt;
}

std::string readMetashapeDocXml(const MetashapeProject &project)
{
    // 优先读取展开文件；压缩工程只提取 chunk 根下的 doc.xml 文本。
    if (!project.docPath.empty())
    {
        return readTextFile(project.docPath);
    }
    if (!project.chunkZipPath.empty())
    {
        return readZipEntryText(project.chunkZipPath, "doc.xml");
    }
    throw std::runtime_error("Metashape 工程缺少 doc.xml 或 chunk.zip");
}

std::unordered_map<int, MetashapeSensor> parseMetashapeSensors(const std::string &xml)
{
    // sensor 定义标定，camera 通过 sensor_id 复用它；先建立 ID 索引可避免
    // 为每个 camera 重复解析 calibration 块。
    std::unordered_map<int, MetashapeSensor> sensors;
    for (const std::string &sensorBlock : xmlBlocks(xml, "sensor"))
    {
        const auto idText = xmlAttribute(sensorBlock, "id");
        if (!idText.has_value())
        {
            continue;
        }

        const auto resolutionMatch = [&]() -> std::optional<std::string> {
            const std::regex re("<resolution\\b[^>]*/?>");
            std::smatch match;
            if (std::regex_search(sensorBlock, match, re))
            {
                return match[0].str();
            }
            return std::nullopt;
        }();
        if (!resolutionMatch.has_value())
        {
            throw std::runtime_error("Metashape sensor 缺少 resolution");
        }

        const auto widthText = xmlAttribute(*resolutionMatch, "width");
        const auto heightText = xmlAttribute(*resolutionMatch, "height");
        if (!widthText.has_value() || !heightText.has_value())
        {
            throw std::runtime_error("Metashape resolution 缺少 width/height");
        }
        const double width = std::stod(*widthText);
        const double height = std::stod(*heightText);

        const std::vector<std::string> calibrationBlocks = xmlBlocks(sensorBlock, "calibration");
        if (calibrationBlocks.empty())
        {
            throw std::runtime_error("Metashape sensor 缺少 calibration");
        }

        // adjusted 是 Metashape 优化后的最终标定；缺失时退回第一个原始标定块。
        std::string calibration = calibrationBlocks.front();
        for (const std::string &block : calibrationBlocks)
        {
            const auto klass = xmlAttribute(block, "class");
            if (klass.has_value() && *klass == "adjusted")
            {
                calibration = block;
                break;
            }
        }

        const double f = xmlDoubleOr(calibration, "f", 0.0);
        const double fx = xmlDoubleOr(calibration, "fx", f);
        const double fy = xmlDoubleOr(calibration, "fy", f);
        if (fx <= 0.0 || fy <= 0.0)
        {
            throw std::runtime_error("Metashape calibration 缺少有效焦距");
        }

        // Metashape cx/cy 以影像中心为原点，PlaScan/Tsai 主点使用左上角
        // 像素坐标，因此需要加上 width/2、height/2。
        const double cx = xmlDoubleOr(calibration, "cx", 0.0);
        const double cy = xmlDoubleOr(calibration, "cy", 0.0);

        MetashapeSensor sensor;
        sensor.k = std::array<double, 9>{{
            fx, 0.0, width * 0.5 + cx,
            0.0, fy, height * 0.5 + cy,
            0.0, 0.0, 1.0
        }};
        sensor.distortion = std::array<double, 5>{{
            xmlDoubleOr(calibration, "k1", 0.0),
            xmlDoubleOr(calibration, "k2", 0.0),
            xmlDoubleOr(calibration, "k3", 0.0),
            xmlDoubleOr(calibration, "p1", 0.0),
            xmlDoubleOr(calibration, "p2", 0.0)
        }};

        if (xmlHasNonZeroTags(calibration, {"k4", "b1", "b2"}))
        {
            // 当前 Tsai 记录只承载 k1..k3/p1/p2；其它非零标定项必须显式告警。
            sensor.warnings.push_back("unsupported Metashape calibration terms are not exported to PlaScan tsai output");
        }
        warnIfUnsupportedSkew(sensor.k, &sensor.warnings);
        sensors.emplace(std::stoi(*idText), sensor);
    }

    if (sensors.empty())
    {
        throw std::runtime_error("Metashape doc.xml 中没有可用 sensor");
    }
    return sensors;
}

std::string stripMetashapeCameraSuffix(const std::string &label)
{
    // Metashape 可把重复/分组 camera 标为 IMG_0001_0；影像文件通常仍是
    // IMG_0001.JPG，因此只剥离末尾纯数字后缀，不改动普通下划线名称。
    const size_t slash = label.find_last_of("/\\");
    const std::string filename = slash == std::string::npos ? label : label.substr(slash + 1);
    const std::filesystem::path path(filename);
    std::string stem = path.has_extension() ? path.stem().string() : filename;
    const size_t underscore = stem.find_last_of('_');
    if (underscore != std::string::npos && underscore + 1 < stem.size())
    {
        const std::string suffix = stem.substr(underscore + 1);
        const bool numericSuffix = std::all_of(suffix.begin(), suffix.end(), [](unsigned char ch) {
            return std::isdigit(ch) != 0;
        });
        if (numericSuffix)
        {
            stem = stem.substr(0, underscore);
        }
    }
    return stem;
}

std::string metashapeLabelBasename(const std::string &label)
{
    const size_t slash = label.find_last_of("/\\");
    return slash == std::string::npos ? label : label.substr(slash + 1);
}

std::optional<CameraRecord> parseMetashapeCameraBlock(
    const std::string &cameraBlock,
    const std::unordered_map<int, MetashapeSensor> &sensors)
{
    const auto label = xmlAttribute(cameraBlock, "label");
    const auto sensorId = xmlAttribute(cameraBlock, "sensor_id");
    const auto transform = xmlElementText(cameraBlock, "transform");
    if (!label.has_value() || !sensorId.has_value() || !transform.has_value())
    {
        return std::nullopt;
    }

    const auto sensorIt = sensors.find(std::stoi(*sensorId));
    if (sensorIt == sensors.end())
    {
        throw std::runtime_error("Metashape camera 引用了不存在的 sensor_id: " + *sensorId);
    }

    // camera transform 是行优先 4×4 camera-to-world 齐次矩阵。
    // 左上 3×3 直接作为 R_cw，最后一列前三项是世界系相机中心 C。
    const std::vector<double> values = parseNumericLine(*transform, 16);

    CameraRecord record;
    record.imageName = metashapeLabelBasename(*label);
    record.k = sensorIt->second.k;
    record.distortion = sensorIt->second.distortion;
    record.warnings = sensorIt->second.warnings;
    record.rotationCameraToWorld = std::array<double, 9>{{
        values[0], values[1], values[2],
        values[4], values[5], values[6],
        values[8], values[9], values[10]
    }};
    record.center = std::array<double, 3>{{values[3], values[7], values[11]}};
    return record;
}

std::vector<CameraRecord> parseMetashapeRecords(const std::string &xml)
{
    const auto sensors = parseMetashapeSensors(xml);
    std::vector<CameraRecord> records;
    for (const std::string &cameraBlock : xmlBlocks(xml, "camera"))
    {
        auto record = parseMetashapeCameraBlock(cameraBlock, sensors);
        if (record.has_value())
        {
            records.push_back(*record);
        }
    }
    if (records.empty())
    {
        throw std::runtime_error("Metashape doc.xml 中没有带 transform 的相机记录");
    }
    return records;
}

std::optional<std::filesystem::path> findMetashapeImage(const std::filesystem::path &imageRoot,
                                                        const std::string &label)
{
    if (!std::filesystem::is_directory(imageRoot))
    {
        return std::nullopt;
    }

    std::vector<std::string> stems{label, stripMetashapeCameraSuffix(label)};
    std::sort(stems.begin(), stems.end());
    stems.erase(std::unique(stems.begin(), stems.end()), stems.end());

    // 先尝试 label 原值/去数字后缀和常见扩展名，可避免不必要的目录扫描。
    const std::vector<std::string> extensions{".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png", ".TIF", ".tif", ".TIFF", ".tiff"};
    for (const std::string &stem : stems)
    {
        const std::filesystem::path direct = imageRoot / stem;
        if (std::filesystem::exists(direct) && std::filesystem::is_regular_file(direct))
        {
            return direct;
        }
        for (const std::string &ext : extensions)
        {
            const std::filesystem::path candidate = imageRoot / (stem + ext);
            if (std::filesystem::exists(candidate) && std::filesystem::is_regular_file(candidate))
            {
                return candidate;
            }
        }
    }

    // 最后做一次大小写不敏感扫描，兼容 Windows/Linux 间迁移后扩展名或文件名大小写变化。
    for (const auto &entry : std::filesystem::directory_iterator(imageRoot))
    {
        if (!entry.is_regular_file() || !isImageFile(entry.path()))
        {
            continue;
        }
        const std::string fileStem = asciiLowerCopy(entry.path().stem().string());
        const std::string fileName = asciiLowerCopy(entry.path().filename().string());
        for (const std::string &stem : stems)
        {
            if (fileStem == asciiLowerCopy(stem) || fileName == asciiLowerCopy(stem))
            {
                return entry.path();
            }
        }
    }
    return std::nullopt;
}

std::filesystem::path assignMetashapeImageRoot(const std::filesystem::path &inputPath,
                                               const MetashapeProject &project,
                                               std::vector<CameraRecord> *records)
{
    if (!records)
    {
        throw std::runtime_error("内部错误：Metashape records 为空");
    }

    // 候选覆盖数据集根、用户输入目录和 chunk 相邻位置下的
    // Depthimages/depthimages/Images/images 变体。
    std::vector<std::filesystem::path> candidates;
    auto pushCandidates = [&](const std::filesystem::path &root) {
        if (root.empty())
        {
            return;
        }
        candidates.push_back(root / "Depthimages");
        candidates.push_back(root / "depthimages");
        candidates.push_back(root / "Images");
        candidates.push_back(root / "images");
        candidates.push_back(root);
    };

    pushCandidates(project.searchRoot);
    if (std::filesystem::is_directory(inputPath))
    {
        pushCandidates(inputPath);
    }
    if (!project.docPath.empty())
    {
        pushCandidates(project.docPath.parent_path());
    }
    if (!project.chunkZipPath.empty())
    {
        pushCandidates(project.chunkZipPath.parent_path());
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    for (const std::filesystem::path &candidate : candidates)
    {
        // 必须为全部 camera 找到影像后才接受候选根，防止多个相似目录混合出
        // 一份不完整或跨目录的 image_camera.lis。
        std::vector<std::filesystem::path> images;
        bool allFound = true;
        for (const CameraRecord &record : *records)
        {
            const auto image = findMetashapeImage(candidate, record.imageName);
            if (!image.has_value())
            {
                allFound = false;
                break;
            }
            images.push_back(*image);
        }

        if (!allFound)
        {
            continue;
        }

        for (size_t i = 0; i < records->size(); ++i)
        {
            (*records)[i].imageName = images[i].filename().string();
            (*records)[i].cameraFileName = cameraFileNameFromStem((*records)[i].imageName);
        }
        return candidate;
    }

    throw std::runtime_error("未找到 Metashape 影像目录，已检查 Depthimages/Images/images 和工程相邻目录");
}

std::vector<CameraRecord> loadMetashapeRecords(const std::filesystem::path &inputPath,
                                               std::filesystem::path *sourceDir)
{
    // 位姿/标定来自工程 XML，影像名和最终 sourceDir 由完整影像匹配决定。
    const auto project = findMetashapeProject(inputPath);
    if (!project.has_value())
    {
        throw std::runtime_error("目录下未找到 Metashape doc.xml 或 Project.files/0/chunk.zip: "
                                 + inputPath.string());
    }

    std::vector<CameraRecord> records = parseMetashapeRecords(readMetashapeDocXml(*project));
    *sourceDir = assignMetashapeImageRoot(inputPath, *project, &records);
    return records;
}

// ---------- 格式识别与统一加载分派 ----------

CameraFormat detectFormat(const std::filesystem::path &inputPath)
{
    // 探测优先级保持稳定：Middlebury -> EPFL -> COLMAP -> Metashape。
    // 显式 --format 会绕过此函数，适合目录同时包含多种元数据的情况。
    if (std::filesystem::is_regular_file(inputPath))
    {
        const std::string name = inputPath.filename().string();
        if (name.find("_par.txt") != std::string::npos)
        {
            return CameraFormat::MiddleburyPar;
        }
        if (hasExtension(inputPath, ".camera"))
        {
            return CameraFormat::EpflCamera;
        }
        if (isColmapTextDirectory(inputPath.parent_path()))
        {
            return CameraFormat::ColmapText;
        }
        if (findMetashapeProject(inputPath).has_value())
        {
            return CameraFormat::MetashapeXml;
        }
    }
    else if (std::filesystem::is_directory(inputPath))
    {
        if (!findFiles(inputPath, "_par.txt").empty())
        {
            return CameraFormat::MiddleburyPar;
        }
        if (!findFiles(inputPath, ".camera").empty())
        {
            return CameraFormat::EpflCamera;
        }
        try
        {
            (void)findColmapTextDirectory(inputPath);
            return CameraFormat::ColmapText;
        }
        catch (const std::exception &)
        {
        }
        if (findMetashapeProject(inputPath).has_value())
        {
            return CameraFormat::MetashapeXml;
        }
    }
    throw std::runtime_error("无法自动识别相机格式，请使用 --format 指定 middlebury-par、epfl-camera、colmap-text 或 metashape-xml");
}

std::filesystem::path sourceDirectoryFor(const std::filesystem::path &inputPath)
{
    if (std::filesystem::is_regular_file(inputPath))
    {
        return inputPath.parent_path();
    }
    return inputPath;
}

std::vector<CameraRecord> loadRecords(CameraFormat format,
                                      const std::filesystem::path &inputPath,
                                      std::filesystem::path *sourceDir)
{
    if (!sourceDir)
    {
        throw std::runtime_error("内部错误：sourceDir 为空");
    }

    // sourceDir 最终必须指向影像相对路径的共同基准；不同格式解析器可在
    // 找到真实影像根后覆盖初始值。
    *sourceDir = sourceDirectoryFor(inputPath);
    if (format == CameraFormat::MiddleburyPar)
    {
        const std::filesystem::path parFile = findMiddleburyParFile(inputPath);
        *sourceDir = parFile.parent_path();
        return parseMiddleburyPar(parFile);
    }
    if (format == CameraFormat::EpflCamera)
    {
        std::vector<CameraRecord> records;
        for (const std::filesystem::path &path : findEpflCameraFiles(inputPath))
        {
            records.push_back(parseEpflCameraFile(path));
        }
        return records;
    }
    if (format == CameraFormat::ColmapText)
    {
        const std::filesystem::path colmapDir = findColmapTextDirectory(inputPath);
        std::vector<CameraRecord> records =
            parseColmapImages(colmapDir / "images.txt", parseColmapCameras(colmapDir / "cameras.txt"));
        sortRecordsByImageName(&records);
        *sourceDir = findColmapImageRoot(inputPath, colmapDir, records);
        return records;
    }
    if (format == CameraFormat::MetashapeXml)
    {
        return loadMetashapeRecords(inputPath, sourceDir);
    }

    throw std::runtime_error("内部错误：不能直接加载 auto 格式");
}

// ---------- PlaScan Tsai、配对列表与 summary 输出 ----------

std::string relativeToken(const std::filesystem::path &path,
                          const std::filesystem::path &baseDir)
{
    // 优先生成相对输出目录的可迁移 token；跨盘符等 relative 失败场景
    // 回退绝对路径，保证列表仍能定位文件。
    std::error_code ec;
    std::filesystem::path rel = std::filesystem::relative(path, baseDir, ec);
    // MSVC 在跨盘符时可能返回空路径但不设置 error_code；空 token 会把
    // image_camera.lis 变成 "'' camera.tsai"，下游因此误报缺少影像列。
    if (ec || rel.empty())
    {
        rel = std::filesystem::absolute(path);
    }
    return rel.generic_string();
}

void writeTsai(const std::filesystem::path &path, const CameraRecord &record)
{
    std::ofstream out = xjw::common::io::openOutputFile(path, std::ios::out | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("无法写入 tsai 文件: " + path.string());
    }

    // CameraRecord 的 K 已是像素单位，因此写 pitch=1，使 Camera 读取时
    // 除以 pitch 后仍得到原始 fx/fy/cx/cy。
    out << "VERSION_3\n";
    out << "PINHOLE\n";
    out << "TSAI\n";
    out << "fu = " << formatNumber(record.k[0]) << "\n";
    out << "fv = " << formatNumber(record.k[4]) << "\n";
    out << "cu = " << formatNumber(record.k[2]) << "\n";
    out << "cv = " << formatNumber(record.k[5]) << "\n";
    out << "u_direction = 1 0 0\n";
    out << "v_direction = 0 1 0\n";
    out << "w_direction = 0 0 1\n";
    out << "pitch = 1\n";
    out << "k1 = " << formatNumber(record.distortion[0]) << "\n";
    out << "k2 = " << formatNumber(record.distortion[1]) << "\n";
    out << "k3 = " << formatNumber(record.distortion[2]) << "\n";
    out << "p1 = " << formatNumber(record.distortion[3]) << "\n";
    out << "p2 = " << formatNumber(record.distortion[4]) << "\n";
    out << "C = " << formatNumber(record.center[0]) << " "
        << formatNumber(record.center[1]) << " "
        << formatNumber(record.center[2]) << "\n";
    out << "R = ";
    for (size_t i = 0; i < record.rotationCameraToWorld.size(); ++i)
    {
        if (i != 0)
        {
            out << " ";
        }
        out << formatNumber(record.rotationCameraToWorld[i]);
    }
    out << "\n";
}

void writeSummary(const std::filesystem::path &summaryPath,
                  const CameraConversionResult &result)
{
    // summary 使用绝对路径便于日志诊断，image_camera.lis 则使用相对路径
    // 支持整体移动输出目录；两者服务于不同消费场景。
    std::ofstream out = xjw::common::io::openOutputFile(summaryPath, std::ios::out | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("无法写入 summary.json: " + summaryPath.string());
    }

    out << "{\n";
    out << "  \"dataset_id\": \"" << jsonEscape(result.datasetId) << "\",\n";
    out << "  \"input_format\": \"" << cameraFormatName(result.inputFormat) << "\",\n";
    out << "  \"source_dir\": \"" << jsonEscape(xjw::common::io::toUtf8Path(std::filesystem::absolute(result.sourceDir))) << "\",\n";
    out << "  \"output_dir\": \"" << jsonEscape(xjw::common::io::toUtf8Path(std::filesystem::absolute(result.outputDir))) << "\",\n";
    out << "  \"image_camera_list\": \""
        << jsonEscape(xjw::common::io::toUtf8Path(std::filesystem::absolute(result.imageCameraList))) << "\",\n";
    out << "  \"camera_count\": " << result.cameraCount << ",\n";
    out << "  \"warnings\": [";
    if (!result.warnings.empty())
    {
        out << "\n";
        for (size_t i = 0; i < result.warnings.size(); ++i)
        {
            out << "    \"" << jsonEscape(result.warnings[i]) << "\"";
            if (i + 1 != result.warnings.size())
            {
                out << ",";
            }
            out << "\n";
        }
        out << "  ";
    }
    out << "]\n";
    out << "}\n";
}

bool directoryIsEmpty(const std::filesystem::path &path)
{
    return std::filesystem::is_directory(path) && std::filesystem::directory_iterator(path)
        == std::filesystem::directory_iterator();
}

void prepareOutputDirectory(const std::filesystem::path &sourceDir,
                            const std::filesystem::path &outputDir,
                            bool overwrite)
{
    if (outputDir.empty())
    {
        throw std::runtime_error("输出目录不能为空");
    }

    std::error_code ec;
    if (std::filesystem::exists(sourceDir, ec) && std::filesystem::exists(outputDir, ec)
        && std::filesystem::equivalent(sourceDir, outputDir, ec))
    {
        // 该保护必须早于 remove_all，防止 overwrite 把输入相机/影像来源删除。
        throw std::runtime_error("输出目录不能等于输入相机目录: " + outputDir.string());
    }

    if (std::filesystem::exists(outputDir))
    {
        if (!std::filesystem::is_directory(outputDir))
        {
            throw std::runtime_error("输出路径已存在且不是目录: " + outputDir.string());
        }
        if (!directoryIsEmpty(outputDir))
        {
            if (!overwrite)
            {
                throw std::runtime_error("输出目录非空，请使用 --overwrite 覆盖: " + outputDir.string());
            }
            // overwrite 的契约是重建完整产物目录，而不是混合旧相机和新结果。
            std::filesystem::remove_all(outputDir);
        }
    }

    std::filesystem::create_directories(outputDir / "cameras");
}

} // namespace

// ---------- 公开格式与转换 API ----------

std::vector<std::string> supportedFormatNames()
{
    return {"auto", "middlebury-par", "epfl-camera", "colmap-text", "metashape-xml"};
}

std::string cameraFormatName(CameraFormat format)
{
    switch (format)
    {
    case CameraFormat::Auto:
        return "auto";
    case CameraFormat::MiddleburyPar:
        return "middlebury-par";
    case CameraFormat::EpflCamera:
        return "epfl-camera";
    case CameraFormat::ColmapText:
        return "colmap-text";
    case CameraFormat::MetashapeXml:
        return "metashape-xml";
    }
    return "unknown";
}

std::string cameraFormatDescription(CameraFormat format)
{
    switch (format)
    {
    case CameraFormat::Auto:
        return "自动识别";
    case CameraFormat::MiddleburyPar:
        return "Middlebury *_par.txt";
    case CameraFormat::EpflCamera:
        return "EPFL/Strecha *.camera";
    case CameraFormat::ColmapText:
        return "COLMAP text sparse cameras.txt/images.txt";
    case CameraFormat::MetashapeXml:
        return "Metashape doc.xml / Project.files/0/chunk.zip";
    }
    return "未知格式";
}

std::optional<CameraFormat> parseCameraFormat(const std::string &name)
{
    const std::string normalized = normalizedFormatName(name);
    if (normalized.empty() || normalized == "auto")
    {
        return CameraFormat::Auto;
    }
    if (normalized == "middlebury-par" || normalized == "middlebury")
    {
        return CameraFormat::MiddleburyPar;
    }
    if (normalized == "epfl-camera" || normalized == "epfl" || normalized == "strecha-camera")
    {
        return CameraFormat::EpflCamera;
    }
    if (normalized == "colmap-text" || normalized == "colmap")
    {
        return CameraFormat::ColmapText;
    }
    if (normalized == "metashape-xml" || normalized == "metashape" || normalized == "agisoft-metashape")
    {
        return CameraFormat::MetashapeXml;
    }
    return std::nullopt;
}

CameraConversionResult convertCameraDataset(const CameraConversionOptions &options)
{
    // 先填入不依赖解析的输出路径，失败结果也能告诉调用方预期产物位置。
    CameraConversionResult result;
    result.outputDir = options.outputDir;
    result.imageCameraList = options.outputDir / "image_camera.lis";
    result.summaryPath = options.outputDir / "summary.json";
    result.datasetId = options.datasetId;

    try
    {
        if (options.inputPath.empty())
        {
            throw std::runtime_error("输入路径不能为空");
        }
        if (!std::filesystem::exists(options.inputPath))
        {
            throw std::runtime_error("输入路径不存在: " + options.inputPath.string());
        }

        // Auto 只决定具体解析器，结果始终记录真正使用的格式而不是 Auto。
        result.inputFormat = options.format == CameraFormat::Auto
            ? detectFormat(options.inputPath)
            : options.format;

        std::vector<CameraRecord> records = loadRecords(result.inputFormat, options.inputPath, &result.sourceDir);
        if (records.empty())
        {
            throw std::runtime_error("没有可转换的相机记录");
        }

        if (result.datasetId.empty())
        {
            result.datasetId = result.sourceDir.filename().string();
        }

        // 相机记录解析完成且影像根已定位后才触碰输出目录；逐记录影像存在性
        // 仍在下方写出循环中验证，因此失败结果可能保留部分产物。
        prepareOutputDirectory(result.sourceDir, options.outputDir, options.overwrite);

        std::ofstream listFile = xjw::common::io::openOutputFile(result.imageCameraList,
                                                                 std::ios::out | std::ios::trunc);
        if (!listFile)
        {
            throw std::runtime_error("无法写入 image_camera.lis: " + result.imageCameraList.string());
        }

        const std::filesystem::path camerasDir = options.outputDir / "cameras";
        for (const CameraRecord &record : records)
        {
            const std::filesystem::path imagePath = result.sourceDir / record.imageName;
            if (!std::filesystem::exists(imagePath))
            {
                throw std::runtime_error("相机记录引用的影像不存在: " + imagePath.string());
            }

            const std::filesystem::path tsaiPath = camerasDir / record.cameraFileName;
            writeTsai(tsaiPath, record);
            result.writtenCameraFiles.push_back(tsaiPath);

            // 每行严格对应“影像 token + 相机 token”，路径相对输出根并独立转义。
            listFile << shellQuote(relativeToken(imagePath, options.outputDir)) << " "
                     << shellQuote(relativeToken(tsaiPath, options.outputDir)) << "\n";

            // 在统一 summary 中加影像名前缀，便于用户定位有损转换来源。
            for (const std::string &warning : record.warnings)
            {
                result.warnings.push_back(record.imageName + ": " + warning);
            }
        }
        listFile.close();

        result.cameraCount = static_cast<int>(records.size());
        writeSummary(result.summaryPath, result);
        result.success = true;
    }
    catch (const std::exception &exc)
    {
        // 公开 API 把内部异常折叠为结构化失败，GUI/CLI 无需跨线程传播异常。
        // 已写出的部分文件不会在这里自动清理，调用方必须以 success 为准。
        result.errorMessage = exc.what();
        result.success = false;
    }

    return result;
}

} // namespace xjw::camera
