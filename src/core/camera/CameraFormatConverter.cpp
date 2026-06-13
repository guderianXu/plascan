#include "CameraFormatConverter.h"

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
namespace
{

struct CameraRecord
{
    std::string imageName;
    std::string cameraFileName;
    std::array<double, 9> k{{0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0}};
    std::array<double, 9> rotationCameraToWorld{{1.0, 0.0, 0.0,
                                                 0.0, 1.0, 0.0,
                                                 0.0, 0.0, 1.0}};
    std::array<double, 3> center{{0.0, 0.0, 0.0}};
    std::vector<std::string> warnings;
};

struct ColmapCameraModel
{
    std::array<double, 9> k{{0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0}};
    std::vector<std::string> warnings;
};

struct MetashapeProject
{
    std::filesystem::path docPath;
    std::filesystem::path chunkZipPath;
    std::filesystem::path searchRoot;
};

struct MetashapeSensor
{
    std::array<double, 9> k{{0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0}};
    std::vector<std::string> warnings;
};

std::string trim(const std::string &text)
{
    const auto begin = std::find_if_not(text.begin(), text.end(),
                                        [](unsigned char ch) { return std::isspace(ch) != 0; });
    const auto end = std::find_if_not(text.rbegin(), text.rend(),
                                      [](unsigned char ch) { return std::isspace(ch) != 0; }).base();
    if (begin >= end)
    {
        return {};
    }
    return std::string(begin, end);
}

std::string toLower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

std::string normalizedFormatName(std::string name)
{
    name = toLower(trim(name));
    std::replace(name.begin(), name.end(), '_', '-');
    return name;
}

std::string formatNumber(double value)
{
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

std::vector<std::string> readDataLines(const std::filesystem::path &path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("无法打开相机文件: " + path.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
    {
        line = trim(line);
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
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("无法打开文本文件: " + path.string());
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string readZipEntryText(const std::filesystem::path &zipPath,
                             const std::string &entryPath)
{
    int errorCode = 0;
    zip_t *archive = zip_open(zipPath.string().c_str(), ZIP_RDONLY, &errorCode);
    if (!archive)
    {
        throw std::runtime_error("无法打开 Metashape zip 文件: " + zipPath.string());
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
    return trim(match[1].str());
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
        warnings->push_back("camera skew terms are not represented in PlaScan tsai output "
                            "(k01=" + formatNumber(k[1]) + ", k10=" + formatNumber(k[3]) + ")");
    }
}

std::string cameraFileNameFromStem(const std::string &imageName)
{
    std::filesystem::path imagePath(imageName);
    return imagePath.stem().string() + ".tsai";
}

CameraRecord parseMiddleburyCameraLine(const std::string &line)
{
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
        warnings->push_back("COLMAP " + model + " distortion terms are not exported to PlaScan tsai output");
    }
}

ColmapCameraModel parseColmapCameraLine(const std::string &line)
{
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
    imageName = trim(imageName);
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
    std::ifstream in(path);
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
        const std::string header = trim(line);
        if (header.empty() || header.rfind("#", 0) == 0)
        {
            continue;
        }

        records.push_back(parseColmapImageLine(header, cameras));

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
    std::stable_sort(records->begin(), records->end(), [](const CameraRecord &lhs, const CameraRecord &rhs) {
        return toLower(std::filesystem::path(lhs.imageName).generic_string())
            < toLower(std::filesystem::path(rhs.imageName).generic_string());
    });
}

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

bool isImageFile(const std::filesystem::path &path)
{
    const std::string ext = toLower(path.extension().string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".tif" || ext == ".tiff" || ext == ".ppm";
}

std::filesystem::path metashapeSearchRootFrom(const std::filesystem::path &path)
{
    if (path.filename() == "0" && path.parent_path().extension() == ".files")
    {
        const std::filesystem::path metashapeDir = path.parent_path().parent_path();
        if (toLower(metashapeDir.filename().string()) == "metashape")
        {
            return metashapeDir.parent_path();
        }
        return metashapeDir;
    }

    if (toLower(path.filename().string()) == "metashape")
    {
        return path.parent_path();
    }
    return path;
}

std::optional<MetashapeProject> makeMetashapeProjectFromFilesDir(const std::filesystem::path &filesDir,
                                                                  const std::filesystem::path &searchRoot)
{
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

        const double cx = xmlDoubleOr(calibration, "cx", 0.0);
        const double cy = xmlDoubleOr(calibration, "cy", 0.0);

        MetashapeSensor sensor;
        sensor.k = std::array<double, 9>{{
            fx, 0.0, width * 0.5 + cx,
            0.0, fy, height * 0.5 + cy,
            0.0, 0.0, 1.0
        }};

        if (xmlHasNonZeroTags(calibration, {"k1", "k2", "k3", "k4", "p1", "p2", "b1", "b2"}))
        {
            sensor.warnings.push_back("Metashape distortion terms are not exported to PlaScan tsai output");
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

    const std::vector<double> values = parseNumericLine(*transform, 16);

    CameraRecord record;
    record.imageName = metashapeLabelBasename(*label);
    record.k = sensorIt->second.k;
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

    for (const auto &entry : std::filesystem::directory_iterator(imageRoot))
    {
        if (!entry.is_regular_file() || !isImageFile(entry.path()))
        {
            continue;
        }
        const std::string fileStem = toLower(entry.path().stem().string());
        const std::string fileName = toLower(entry.path().filename().string());
        for (const std::string &stem : stems)
        {
            if (fileStem == toLower(stem) || fileName == toLower(stem))
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

CameraFormat detectFormat(const std::filesystem::path &inputPath)
{
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

std::string relativeToken(const std::filesystem::path &path,
                          const std::filesystem::path &baseDir)
{
    std::error_code ec;
    std::filesystem::path rel = std::filesystem::relative(path, baseDir, ec);
    if (ec)
    {
        rel = std::filesystem::absolute(path);
    }
    return rel.generic_string();
}

void writeTsai(const std::filesystem::path &path, const CameraRecord &record)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("无法写入 tsai 文件: " + path.string());
    }

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
    out << "k1 = 0\n";
    out << "k2 = 0\n";
    out << "k3 = 0\n";
    out << "p1 = 0\n";
    out << "p2 = 0\n";
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
    std::ofstream out(summaryPath);
    if (!out)
    {
        throw std::runtime_error("无法写入 summary.json: " + summaryPath.string());
    }

    out << "{\n";
    out << "  \"dataset_id\": \"" << jsonEscape(result.datasetId) << "\",\n";
    out << "  \"input_format\": \"" << cameraFormatName(result.inputFormat) << "\",\n";
    out << "  \"source_dir\": \"" << jsonEscape(std::filesystem::absolute(result.sourceDir).string()) << "\",\n";
    out << "  \"output_dir\": \"" << jsonEscape(std::filesystem::absolute(result.outputDir).string()) << "\",\n";
    out << "  \"image_camera_list\": \""
        << jsonEscape(std::filesystem::absolute(result.imageCameraList).string()) << "\",\n";
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
            std::filesystem::remove_all(outputDir);
        }
    }

    std::filesystem::create_directories(outputDir / "cameras");
}

} // namespace

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

        prepareOutputDirectory(result.sourceDir, options.outputDir, options.overwrite);

        std::ofstream listFile(result.imageCameraList);
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

            listFile << shellQuote(relativeToken(imagePath, options.outputDir)) << " "
                     << shellQuote(relativeToken(tsaiPath, options.outputDir)) << "\n";

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
        result.errorMessage = exc.what();
        result.success = false;
    }

    return result;
}

} // namespace xjw::camera
