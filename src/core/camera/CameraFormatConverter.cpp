#include "CameraFormatConverter.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
    }
    throw std::runtime_error("无法自动识别相机格式，请使用 --format 指定 middlebury-par 或 epfl-camera");
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
    return {"auto", "middlebury-par", "epfl-camera"};
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
