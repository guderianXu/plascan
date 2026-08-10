#include "DenseCloudArtifactValidation.h"

#include "io/PathIO.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>

namespace xjw::mvs::detail
{
namespace
{

struct BinaryPlyHeader
{
    bool binaryLittleEndian = false;
    bool sawVertexElement = false;
    bool sawEndHeader = false;
    bool hasX = false;
    bool hasY = false;
    bool hasZ = false;
    std::uint64_t vertexCount = 0;
    std::uint64_t vertexStride = 0;
    std::uintmax_t dataOffset = 0;
};

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

int scalarSize(const std::string &type)
{
    if (type == "char" || type == "int8" || type == "uchar" || type == "uint8")
    {
        return 1;
    }
    if (type == "short" || type == "int16" || type == "ushort" || type == "uint16")
    {
        return 2;
    }
    if (type == "int" || type == "int32" || type == "uint" || type == "uint32"
        || type == "float" || type == "float32")
    {
        return 4;
    }
    if (type == "double" || type == "float64")
    {
        return 8;
    }
    return 0;
}

bool parseElement(std::istringstream *line,
                  bool *inVertexElement,
                  BinaryPlyHeader *header,
                  std::string *errorMessage)
{
    std::string name;
    std::uint64_t count = 0;
    std::string trailing;
    if (!(*line >> name >> count) || (*line >> trailing))
    {
        setError(errorMessage, "PLY element declaration is invalid");
        return false;
    }

    *inVertexElement = name == "vertex";
    if (*inVertexElement)
    {
        if (header->sawVertexElement)
        {
            setError(errorMessage, "PLY declares more than one vertex element");
            return false;
        }
        header->sawVertexElement = true;
        header->vertexCount = count;
    }
    else if (count != 0)
    {
        setError(errorMessage, "Refined PLY contains an unexpected non-vertex element: " + name);
        return false;
    }
    return true;
}

bool parseProperty(std::istringstream *line,
                   BinaryPlyHeader *header,
                   std::string *errorMessage)
{
    std::string type;
    std::string name;
    std::string trailing;
    if (!(*line >> type >> name) || (*line >> trailing) || type == "list")
    {
        setError(errorMessage, "PLY vertex property declaration is invalid");
        return false;
    }

    const int size = scalarSize(type);
    if (size <= 0)
    {
        setError(errorMessage, "PLY vertex property type is unsupported: " + type);
        return false;
    }
    if (header->vertexStride
        > std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(size))
    {
        setError(errorMessage, "PLY vertex stride overflows");
        return false;
    }

    header->vertexStride += static_cast<std::uint64_t>(size);
    header->hasX = header->hasX || name == "x";
    header->hasY = header->hasY || name == "y";
    header->hasZ = header->hasZ || name == "z";
    return true;
}

bool parseHeader(const std::filesystem::path &path,
                 BinaryPlyHeader *header,
                 std::string *errorMessage)
{
    std::ifstream input = xjw::common::io::openInputFile(path, std::ios::in | std::ios::binary);
    if (!input)
    {
        setError(errorMessage, "Cannot open staged PLY: " + xjw::common::io::toUtf8Path(path));
        return false;
    }

    std::string line;
    if (!std::getline(input, line))
    {
        setError(errorMessage, "Staged output is not a PLY file");
        return false;
    }
    if (!line.empty() && line.back() == '\r')
    {
        line.pop_back();
    }
    if (line != "ply")
    {
        setError(errorMessage, "Staged output is not a PLY file");
        return false;
    }

    bool inVertexElement = false;
    std::size_t headerBytes = line.size() + 1;
    constexpr std::size_t kMaximumHeaderBytes = 1024 * 1024;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        headerBytes += line.size() + 1;
        if (headerBytes > kMaximumHeaderBytes)
        {
            setError(errorMessage, "PLY header exceeds the safety limit");
            return false;
        }
        if (line == "end_header")
        {
            const std::streamoff offset = input.tellg();
            if (offset < 0)
            {
                setError(errorMessage, "Cannot determine PLY data offset");
                return false;
            }
            header->dataOffset = static_cast<std::uintmax_t>(offset);
            header->sawEndHeader = true;
            break;
        }

        std::istringstream fields(line);
        std::string keyword;
        fields >> keyword;
        if (keyword == "format")
        {
            std::string format;
            std::string version;
            std::string trailing;
            if (!(fields >> format >> version) || (fields >> trailing))
            {
                setError(errorMessage, "PLY format declaration is invalid");
                return false;
            }
            header->binaryLittleEndian = format == "binary_little_endian" && version == "1.0";
        }
        else if (keyword == "element")
        {
            if (!parseElement(&fields, &inVertexElement, header, errorMessage))
            {
                return false;
            }
        }
        else if (keyword == "property" && inVertexElement)
        {
            if (!parseProperty(&fields, header, errorMessage))
            {
                return false;
            }
        }
    }

    if (!header->binaryLittleEndian || !header->sawVertexElement || !header->sawEndHeader
        || !header->hasX || !header->hasY || !header->hasZ || header->vertexStride == 0)
    {
        setError(errorMessage, "Staged PLY header is incomplete or unsupported");
        return false;
    }
    return true;
}

} // namespace

bool validateDenseCloudPlyArtifact(const std::filesystem::path &path,
                                   std::size_t expectedVertexCount,
                                   std::string *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }

    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        setError(errorMessage, "Staged PLY does not exist or is not a regular file");
        return false;
    }

    BinaryPlyHeader header;
    if (!parseHeader(path, &header, errorMessage))
    {
        return false;
    }
    if (header.vertexCount != static_cast<std::uint64_t>(expectedVertexCount))
    {
        setError(errorMessage,
                 "Staged PLY vertex count mismatch: expected "
                     + std::to_string(expectedVertexCount) + ", got "
                     + std::to_string(header.vertexCount));
        return false;
    }

    if (header.vertexCount
        > (std::numeric_limits<std::uintmax_t>::max() - header.dataOffset)
            / header.vertexStride)
    {
        setError(errorMessage, "Staged PLY byte length overflows");
        return false;
    }
    const std::uintmax_t expectedBytes = header.dataOffset
        + static_cast<std::uintmax_t>(header.vertexCount * header.vertexStride);
    const std::uintmax_t actualBytes = std::filesystem::file_size(path, error);
    if (error || actualBytes != expectedBytes)
    {
        setError(errorMessage,
                 "Staged PLY byte length mismatch: expected "
                     + std::to_string(expectedBytes) + ", got "
                     + (error ? error.message() : std::to_string(actualBytes)));
        return false;
    }
    return true;
}

} // namespace xjw::mvs::detail
