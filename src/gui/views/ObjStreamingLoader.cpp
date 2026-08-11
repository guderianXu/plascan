#include "ObjStreamingLoader.h"
#include "ObjStreamingLoaderInternal.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <plapoint/io/obj_io.h>

namespace
{

constexpr std::uint64_t kProgressIntervalBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kStreamBufferBytes = 8ULL * 1024ULL * 1024ULL;

struct ObjCounts
{
    std::size_t vertices = 0;
    std::size_t normals = 0;
    std::size_t textureCoordinates = 0;
    std::size_t triangles = 0;
};

bool isCancelled(const std::atomic_bool *flag)
{
    return flag && flag->load(std::memory_order_relaxed);
}

std::runtime_error parseError(const std::string &path,
                              std::size_t lineNumber,
                              const std::string &message)
{
    return std::runtime_error(
        "OBJ parse error in " + path + " line "
        + std::to_string(lineNumber) + ": " + message);
}

float parseFloat(std::string_view token,
                 const std::string &path,
                 std::size_t lineNumber,
                 const char *label)
{
    float value = 0.0f;
    const auto parsed = std::from_chars(
        token.data(), token.data() + token.size(), value, std::chars_format::general);
    if (token.empty() || parsed.ec != std::errc()
        || parsed.ptr != token.data() + token.size() || !std::isfinite(value))
    {
        throw parseError(path, lineNumber, std::string(label) + " must be a finite number");
    }
    return value;
}

std::ifstream openBufferedStream(const std::string &path, std::vector<char> *buffer)
{
    std::ifstream stream;
    buffer->resize(kStreamBufferBytes);
    stream.rdbuf()->pubsetbuf(buffer->data(), static_cast<std::streamsize>(buffer->size()));
    stream.open(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Cannot open OBJ file: " + path);
    }
    return stream;
}

std::uint64_t fileSize(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
    {
        throw std::runtime_error("Cannot open OBJ file: " + path);
    }
    const std::streamoff size = stream.tellg();
    if (size < 0)
    {
        throw std::runtime_error("Cannot determine OBJ file size: " + path);
    }
    return static_cast<std::uint64_t>(size);
}

void reportByteProgress(const ObjLoadProgressCallback &progress,
                        std::uint64_t processedBytes,
                        std::uint64_t totalBytes,
                        int firstPercent,
                        int lastPercent,
                        const QString &stage,
                        std::uint64_t *nextReport)
{
    if (!progress || (processedBytes < *nextReport && processedBytes < totalBytes))
    {
        return;
    }
    const double ratio = totalBytes > 0
        ? std::min(1.0, static_cast<double>(processedBytes) / static_cast<double>(totalBytes))
        : 1.0;
    progress(firstPercent + static_cast<int>(std::lround((lastPercent - firstPercent) * ratio)),
             stage);
    *nextReport = processedBytes + kProgressIntervalBytes;
}

ObjCounts scanObj(const std::string &path,
                  std::uint64_t totalBytes,
                  const ObjLoadProgressCallback &progress,
                  const std::atomic_bool *cancellationFlag)
{
    ObjCounts counts;
    std::vector<char> streamBuffer;
    std::ifstream stream = openBufferedStream(path, &streamBuffer);
    std::string line;
    std::uint64_t processedBytes = 0;
    std::uint64_t nextReport = 0;
    while (std::getline(stream, line))
    {
        processedBytes += static_cast<std::uint64_t>(line.size()) + 1;
        if (isCancelled(cancellationFlag))
        {
            return {};
        }
        const char *cursor = line.data();
        const char *end = cursor + line.size();
        const std::string_view token = plapoint::io::detail::nextObjToken(cursor, end);
        if (token == "v")
        {
            ++counts.vertices;
        }
        else if (token == "vn")
        {
            ++counts.normals;
        }
        else if (token == "vt")
        {
            ++counts.textureCoordinates;
        }
        else if (token == "f")
        {
            std::size_t corners = 0;
            while (!plapoint::io::detail::nextObjToken(cursor, end).empty())
            {
                ++corners;
            }
            if (corners >= 3)
            {
                counts.triangles += corners - 2;
            }
        }
        reportByteProgress(progress, processedBytes, totalBytes, 2, 18,
                           QStringLiteral("正在扫描 OBJ 结构..."), &nextReport);
    }
    if (!stream.eof())
    {
        throw std::runtime_error("Cannot scan OBJ file: " + path);
    }
    return counts;
}

int resolveIndex(int index,
                 std::size_t count,
                 const std::string &path,
                 std::size_t lineNumber,
                 const char *label)
{
    try
    {
        return plapoint::io::detail::resolveObjIndex(index, count, label);
    }
    catch (const std::exception &exception)
    {
        throw parseError(path, lineNumber, exception.what());
    }
}

} // namespace

std::shared_ptr<StreamingObjCloud> readObjStreaming(
    const std::string &path,
    const ObjLoadProgressCallback &progress,
    const std::atomic_bool *cancellationFlag)
{
    const std::uint64_t totalBytes = fileSize(path);
    if (progress)
    {
        progress(2, QStringLiteral("正在扫描 OBJ 结构..."));
    }
    const ObjCounts counts = scanObj(path, totalBytes, progress, cancellationFlag);
    if (isCancelled(cancellationFlag))
    {
        return {};
    }

    std::vector<float> vx, vy, vz, nx, ny, nz, tx, ty;
    std::vector<std::uint8_t> vr, vg, vb;
    std::vector<bool> hasVertexColor;
    std::vector<std::array<int, 3>> faceVertices;
    std::vector<std::array<int, 3>> faceTextures;
    vx.reserve(counts.vertices);
    vy.reserve(counts.vertices);
    vz.reserve(counts.vertices);
    vr.reserve(counts.vertices);
    vg.reserve(counts.vertices);
    vb.reserve(counts.vertices);
    hasVertexColor.reserve(counts.vertices);
    nx.reserve(counts.normals);
    ny.reserve(counts.normals);
    nz.reserve(counts.normals);
    tx.reserve(counts.textureCoordinates);
    ty.reserve(counts.textureCoordinates);
    faceVertices.reserve(counts.triangles);
    if (counts.textureCoordinates > 0)
    {
        faceTextures.reserve(counts.triangles);
    }

    std::vector<char> streamBuffer;
    std::ifstream stream = openBufferedStream(path, &streamBuffer);
    std::string line;
    std::string materialLibrary;
    std::uint64_t processedBytes = 0;
    std::uint64_t nextReport = 0;
    std::size_t lineNumber = 0;
    bool faceTexturesComplete = counts.textureCoordinates > 0;
    while (std::getline(stream, line))
    {
        ++lineNumber;
        processedBytes += static_cast<std::uint64_t>(line.size()) + 1;
        if (isCancelled(cancellationFlag))
        {
            return {};
        }
        const char *cursor = line.data();
        const char *end = cursor + line.size();
        const std::string_view token = plapoint::io::detail::nextObjToken(cursor, end);
        if (token.empty() || token.front() == '#')
        {
            continue;
        }
        if (token == "v")
        {
            vx.push_back(parseFloat(plapoint::io::detail::nextObjToken(cursor, end), path, lineNumber, "vertex x"));
            vy.push_back(parseFloat(plapoint::io::detail::nextObjToken(cursor, end), path, lineNumber, "vertex y"));
            vz.push_back(parseFloat(plapoint::io::detail::nextObjToken(cursor, end), path, lineNumber, "vertex z"));
            std::array<float, 3> extras{};
            std::size_t extraCount = 0;
            while (true)
            {
                const std::string_view extra = plapoint::io::detail::nextObjToken(cursor, end);
                if (extra.empty() || extra.front() == '#')
                {
                    break;
                }
                extras[extraCount % 3] = parseFloat(extra, path, lineNumber, "vertex colour");
                ++extraCount;
            }
            const bool hasColor = extraCount >= 3;
            const float r = hasColor ? extras[(extraCount - 3) % 3] : 0.0f;
            const float g = hasColor ? extras[(extraCount - 2) % 3] : 0.0f;
            const float b = hasColor ? extras[(extraCount - 1) % 3] : 0.0f;
            const bool normalized = hasColor && r >= 0.0f && r <= 1.0f
                && g >= 0.0f && g <= 1.0f && b >= 0.0f && b <= 1.0f;
            vr.push_back(plapoint::io::detail::objColorByte(r, normalized));
            vg.push_back(plapoint::io::detail::objColorByte(g, normalized));
            vb.push_back(plapoint::io::detail::objColorByte(b, normalized));
            hasVertexColor.push_back(hasColor);
        }
        else if (token == "vn")
        {
            nx.push_back(parseFloat(plapoint::io::detail::nextObjToken(cursor, end), path, lineNumber, "normal x"));
            ny.push_back(parseFloat(plapoint::io::detail::nextObjToken(cursor, end), path, lineNumber, "normal y"));
            nz.push_back(parseFloat(plapoint::io::detail::nextObjToken(cursor, end), path, lineNumber, "normal z"));
        }
        else if (token == "vt")
        {
            tx.push_back(parseFloat(plapoint::io::detail::nextObjToken(cursor, end), path, lineNumber, "texture u"));
            ty.push_back(parseFloat(plapoint::io::detail::nextObjToken(cursor, end), path, lineNumber, "texture v"));
        }
        else if (token == "f")
        {
            plapoint::io::detail::ObjVertexIndices first;
            plapoint::io::detail::ObjVertexIndices previous;
            std::size_t cornerCount = 0;
            while (true)
            {
                const std::string_view cornerToken = plapoint::io::detail::nextObjToken(cursor, end);
                if (cornerToken.empty() || cornerToken.front() == '#')
                {
                    break;
                }
                plapoint::io::detail::ObjVertexIndices current;
                try
                {
                    current = plapoint::io::detail::parseFaceVertex(cornerToken);
                }
                catch (const std::exception &exception)
                {
                    throw parseError(path, lineNumber, exception.what());
                }
                current.v = resolveIndex(current.v, vx.size(), path, lineNumber, "OBJ face vertex");
                if (current.has_t)
                {
                    current.t = resolveIndex(current.t, tx.size(), path, lineNumber, "OBJ face texture");
                }
                if (current.has_n)
                {
                    current.n = resolveIndex(current.n, nx.size(), path, lineNumber, "OBJ face normal");
                }
                if (cornerCount == 0)
                {
                    first = current;
                }
                else if (cornerCount >= 2)
                {
                    faceVertices.push_back({first.v, previous.v, current.v});
                    const bool triangleHasTexture = first.has_t && previous.has_t && current.has_t;
                    if (faceTexturesComplete && triangleHasTexture)
                    {
                        faceTextures.push_back({first.t, previous.t, current.t});
                    }
                    else if (faceTexturesComplete)
                    {
                        faceTexturesComplete = false;
                        std::vector<std::array<int, 3>>().swap(faceTextures);
                    }
                }
                previous = current;
                ++cornerCount;
            }
            if (cornerCount < 3)
            {
                throw parseError(path, lineNumber, "face must contain at least 3 vertices");
            }
        }
        else if (token == "mtllib")
        {
            const std::string_view name = plapoint::io::detail::nextObjToken(cursor, end);
            if (name.empty())
            {
                throw parseError(path, lineNumber, "mtllib line must contain a file name");
            }
            materialLibrary.assign(name.data(), name.size());
        }
        reportByteProgress(progress, processedBytes, totalBytes, 18, 64,
                           QStringLiteral("正在流式解析 OBJ..."), &nextReport);
    }
    if (!stream.eof())
    {
        throw std::runtime_error("Cannot parse OBJ file: " + path);
    }
    if (faceVertices.size() != counts.triangles || vx.size() != counts.vertices)
    {
        throw std::runtime_error("OBJ changed while it was being loaded: " + path);
    }

    xjw::gui::obj_streaming::ObjAssemblyInput input{
        std::move(vx),
        std::move(vy),
        std::move(vz),
        std::move(nx),
        std::move(ny),
        std::move(nz),
        std::move(tx),
        std::move(ty),
        std::move(vr),
        std::move(vg),
        std::move(vb),
        std::move(hasVertexColor),
        std::move(faceVertices),
        std::move(faceTextures),
        std::move(materialLibrary),
        faceTexturesComplete};
    return xjw::gui::obj_streaming::assembleObjCloud(
        std::move(input), progress, cancellationFlag);
}
