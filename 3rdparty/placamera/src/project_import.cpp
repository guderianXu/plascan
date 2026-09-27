#include "placamera/project_import.h"

#include "project_import_internal.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace placamera
{
    namespace
    {

        std::string lower(std::string text)
        {
            std::transform(text.begin(),
                           text.end(),
                           text.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            return text;
        }

        std::string pathUtf8(const std::filesystem::path& path)
        {
            const auto bytes = path.u8string();
            return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
        }

        std::string trim(std::string_view text)
        {
            const auto first = text.find_first_not_of(" \t\r\n");
            if (first == std::string_view::npos)
            {
                return {};
            }
            const auto last = text.find_last_not_of(" \t\r\n");
            return std::string(text.substr(first, last - first + 1));
        }

        bool endsWith(std::string_view text, std::string_view suffix)
        {
            return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
        }

        std::ifstream openText(const std::filesystem::path& path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("cannot read camera file: " + pathUtf8(path));
            }
            return input;
        }

        std::vector<std::filesystem::path> filesWithSuffix(const std::filesystem::path& path, std::string_view suffix)
        {
            if (std::filesystem::is_regular_file(path))
            {
                return {path};
            }
            std::vector<std::filesystem::path> matches;
            if (std::filesystem::is_directory(path))
            {
                for (const auto& entry : std::filesystem::directory_iterator(path))
                {
                    if (entry.is_regular_file() && endsWith(lower(pathUtf8(entry.path().filename())), suffix))
                    {
                        matches.push_back(entry.path());
                    }
                }
            }
            std::sort(matches.begin(), matches.end());
            return matches;
        }

        std::filesystem::path colmapDirectory(const std::filesystem::path& path)
        {
            const auto root = std::filesystem::is_regular_file(path) ? path.parent_path() : path;
            for (const auto& directory : {root, root / "sparse", root / "sparse" / "0"})
            {
                if (std::filesystem::is_regular_file(directory / "cameras.txt") &&
                    std::filesystem::is_regular_file(directory / "images.txt"))
                {
                    return directory;
                }
            }
            throw std::runtime_error("COLMAP project has no cameras.txt/images.txt pair: " + pathUtf8(root));
        }

        std::filesystem::path metashapeDocumentPath(const std::filesystem::path& path)
        {
            if (std::filesystem::is_regular_file(path) && lower(pathUtf8(path.filename())) == "chunk.zip")
            {
                throw std::runtime_error("extract doc.xml from Metashape chunk.zip before importing it");
            }
            if (std::filesystem::is_regular_file(path) && lower(pathUtf8(path.extension())) == ".xml")
            {
                return path;
            }
            const auto root = std::filesystem::is_regular_file(path) ? path.parent_path() : path;
            for (const auto& candidate : {root / "doc.xml", root / "0" / "doc.xml"})
            {
                if (std::filesystem::is_regular_file(candidate))
                {
                    return candidate;
                }
            }
            if (lower(pathUtf8(path.extension())) == ".psx")
            {
                auto filesDirectory = path.stem();
                filesDirectory += ".files";
                const auto candidate = path.parent_path() / filesDirectory / "0" / "doc.xml";
                if (std::filesystem::is_regular_file(candidate))
                {
                    return candidate;
                }
            }
            throw std::runtime_error("Metashape project has no readable doc.xml: " + pathUtf8(path));
        }

        CameraProjectFormat detectFormat(const std::filesystem::path& path)
        {
            const std::string name = lower(pathUtf8(path.filename()));
            if (std::filesystem::is_regular_file(path))
            {
                if (endsWith(name, "_par.txt"))
                {
                    return CameraProjectFormat::MiddleburyPar;
                }
                if (endsWith(name, ".camera"))
                {
                    return CameraProjectFormat::EpflCamera;
                }
                if (name == "cameras.txt" || name == "images.txt")
                {
                    return CameraProjectFormat::ColmapText;
                }
                if (endsWith(name, ".xml") || endsWith(name, ".psx") || name == "chunk.zip")
                {
                    return CameraProjectFormat::MetashapeXml;
                }
                if (endsWith(name, ".txt"))
                {
                    return CameraProjectFormat::MetashapeReferenceText;
                }
            }
            if (!filesWithSuffix(path, "_par.txt").empty())
            {
                return CameraProjectFormat::MiddleburyPar;
            }
            if (!filesWithSuffix(path, ".camera").empty())
            {
                return CameraProjectFormat::EpflCamera;
            }
            try
            {
                (void)colmapDirectory(path);
                return CameraProjectFormat::ColmapText;
            }
            catch (const std::runtime_error&)
            {
            }
            (void)metashapeDocumentPath(path);
            return CameraProjectFormat::MetashapeXml;
        }

        CameraProjectImportResult importMiddlebury(const std::filesystem::path& path)
        {
            const auto files = filesWithSuffix(path, "_par.txt");
            if (files.empty())
            {
                throw std::runtime_error("no Middlebury *_par.txt file found: " + pathUtf8(path));
            }
            CameraProjectImportResult result;
            result.format = CameraProjectFormat::MiddleburyPar;
            for (const auto& file : files)
            {
                auto input = openText(file);
                auto cameras = readMiddleburyPar(input);
                if (!cameras)
                {
                    throw std::runtime_error("cannot parse Middlebury camera file " + pathUtf8(file) + ": " +
                                             cameras.message());
                }
                for (auto& camera : cameras.value())
                {
                    result.cameras.push_back({std::move(camera), std::nullopt, std::nullopt});
                }
            }
            return result;
        }

        CameraProjectImportResult importEpfl(const std::filesystem::path& path)
        {
            const auto files = filesWithSuffix(path, ".camera");
            if (files.empty())
            {
                throw std::runtime_error("no EPFL .camera file found: " + pathUtf8(path));
            }
            CameraProjectImportResult result;
            result.format = CameraProjectFormat::EpflCamera;
            for (const auto& file : files)
            {
                auto input = openText(file);
                auto camera = readEpflCamera(input);
                if (!camera)
                {
                    throw std::runtime_error("cannot parse EPFL camera file " + pathUtf8(file) + ": " +
                                             camera.message());
                }
                const auto fileName = pathUtf8(file.filename());
                camera.value().imageName = fileName.substr(0, fileName.size() - 7);
                result.cameras.push_back({camera.takeValue(), std::nullopt, std::nullopt});
            }
            return result;
        }

        CameraProjectImportResult importColmap(const std::filesystem::path& path)
        {
            const auto directory = colmapDirectory(path);
            auto camerasInput = openText(directory / "cameras.txt");
            std::unordered_map<int, ColmapCamera> cameras;
            std::string line;
            while (std::getline(camerasInput, line))
            {
                line = trim(line);
                if (line.empty() || line.front() == '#')
                {
                    continue;
                }
                auto camera = parseColmapCameraLine(line);
                if (!camera)
                {
                    throw std::runtime_error(camera.message());
                }
                const int id = camera.value().cameraId;
                if (!cameras.emplace(id, camera.takeValue()).second)
                {
                    throw std::runtime_error("duplicate CAMERA_ID in COLMAP cameras.txt: " + std::to_string(id));
                }
            }
            if (cameras.empty())
            {
                throw std::runtime_error("COLMAP cameras.txt has no camera records: " + pathUtf8(directory));
            }

            auto imagesInput = openText(directory / "images.txt");
            CameraProjectImportResult result;
            result.format = CameraProjectFormat::ColmapText;
            int lineNumber = 0;
            while (std::getline(imagesInput, line))
            {
                ++lineNumber;
                line = trim(line);
                if (line.empty() || line.front() == '#')
                {
                    continue;
                }
                auto image = parseColmapImageLine(line, FrameId("colmap-import"));
                if (!image)
                {
                    throw std::runtime_error(image.message());
                }
                const auto camera = cameras.find(image.value().cameraId);
                if (camera == cameras.end())
                {
                    throw std::runtime_error("COLMAP images.txt references unknown CAMERA_ID: " +
                                             std::to_string(image.value().cameraId));
                }
                auto intrinsics = colmapRasterIntrinsics(camera->second);
                if (!intrinsics)
                {
                    throw std::runtime_error(intrinsics.message());
                }
                ImportedCamera geometry;
                geometry.imageName = image.value().imageName;
                geometry.calibration.intrinsicMatrix = {intrinsics.value().focalX,
                                                        intrinsics.value().skew,
                                                        intrinsics.value().principalX,
                                                        0.0,
                                                        intrinsics.value().focalY,
                                                        intrinsics.value().principalY,
                                                        0.0,
                                                        0.0,
                                                        1.0};
                geometry.cameraToWorldRotation = image.value().pose.cameraToWorldRotation;
                geometry.center = image.value().pose.center;
                auto distortion = colmapBrownConradyDistortion(camera->second);
                if (distortion)
                {
                    geometry.calibration.distortion = distortion.value();
                }
                else
                {
                    geometry.compatibility.unsupportedReason = distortion.message();
                }
                result.cameras.push_back({std::move(geometry), camera->second, image.value().imageId});
                if (!std::getline(imagesInput, line))
                {
                    throw std::runtime_error("COLMAP images.txt record at line " + std::to_string(lineNumber) +
                                             " has no POINTS2D row: " + pathUtf8(directory));
                }
                ++lineNumber;
            }
            if (result.cameras.empty())
            {
                throw std::runtime_error("COLMAP images.txt has no image records: " + pathUtf8(directory));
            }
            return result;
        }

    } // namespace

    Result<CameraProjectImportResult> importMetashapeDocument(std::string_view xml)
    {
        CameraProjectImportResult result;
        result.format = CameraProjectFormat::MetashapeXml;
        auto cameras = parseMetashapeDocument(xml);
        if (!cameras)
        {
            return Result<CameraProjectImportResult>::failure(cameras.error());
        }
        for (auto& camera : cameras.value())
        {
            result.cameras.push_back({std::move(camera), std::nullopt, std::nullopt});
        }
        return Result<CameraProjectImportResult>::success(std::move(result));
    }

    Result<CameraProjectImportResult> importCameraProject(const std::filesystem::path& path,
                                                          CameraProjectFormat format,
                                                          const std::filesystem::path& gnssOffsetPath)
    {
        if (path.empty())
        {
            return Result<CameraProjectImportResult>::failure(CameraErrorCode::InvalidArgument,
                                                              "camera project path is empty");
        }
        std::error_code path_error;
        if (!std::filesystem::exists(path, path_error))
        {
            return Result<CameraProjectImportResult>::failure(
                path_error ? CameraErrorCode::IoFailure : CameraErrorCode::InvalidArgument,
                path_error ? "cannot inspect camera project path: " + path_error.message()
                           : "camera project path does not exist",
                pathUtf8(path));
        }
        if (format != CameraProjectFormat::MetashapeReferenceText && !gnssOffsetPath.empty())
        {
            return Result<CameraProjectImportResult>::failure(
                CameraErrorCode::InvalidArgument,
                "GNSS offset files are only valid for Metashape reference text import",
                pathUtf8(gnssOffsetPath));
        }
        try
        {
            if (format == CameraProjectFormat::Auto)
            {
                format = detectFormat(path);
            }
            switch (format)
            {
            case CameraProjectFormat::MiddleburyPar:
                return Result<CameraProjectImportResult>::success(importMiddlebury(path));
            case CameraProjectFormat::EpflCamera:
                return Result<CameraProjectImportResult>::success(importEpfl(path));
            case CameraProjectFormat::ColmapText:
                return Result<CameraProjectImportResult>::success(importColmap(path));
            case CameraProjectFormat::MetashapeXml:
            {
                const auto document = metashapeDocumentPath(path);
                auto input = openText(document);
                const std::string xml{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
                return importMetashapeDocument(xml);
            }
            case CameraProjectFormat::MetashapeReferenceText:
                return Result<CameraProjectImportResult>::success(
                    internal::importMetashapeReferenceFiles(path, gnssOffsetPath));
            case CameraProjectFormat::Auto:
                break;
            }
        }
        catch (const std::filesystem::filesystem_error& error)
        {
            return Result<CameraProjectImportResult>::failure(CameraErrorCode::IoFailure, error.what(), pathUtf8(path));
        }
        catch (const std::exception& error)
        {
            return Result<CameraProjectImportResult>::failure(
                CameraErrorCode::ParseFailure, error.what(), pathUtf8(path));
        }
        return Result<CameraProjectImportResult>::failure(
            CameraErrorCode::UnsupportedFormat, "unknown camera project format", pathUtf8(path));
    }

} // namespace placamera
