#include "reporting/SparsePlyWriter.h"
#include "file/FileIO.h"

#include <bit>
#include <limits>

namespace xjw::aerial_triangulation
{
    bool writeSparsePly(const std::filesystem::path& path,
                        std::size_t count,
                        const std::function<SparsePlyVertex(std::size_t)>& vertex,
                        std::string_view comment,
                        std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        if (!vertex || comment.find_first_of("\r\n") != std::string_view::npos)
        {
            if (error)
            {
                *error = "PLY 顶点读取器为空或注释包含换行: " + common::file::pathToUtf8(path);
            }
            return false;
        }
        common::file::AtomicFile output(path);
        if (!output.open(error))
        {
            return false;
        }
        auto& stream = output.stream();
        const std::string header = "ply\nformat binary_little_endian 1.0\ncomment " + std::string(comment) +
                                   "\nelement vertex " + std::to_string(count) +
                                   "\nproperty float x\nproperty float y\nproperty float z\n"
                                   "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
        for (std::size_t index = 0; index < count && stream; ++index)
        {
            const auto point = vertex(index);
            std::array<char, 15> bytes{};
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                const auto bits = std::bit_cast<std::uint32_t>(point.xyz[axis]);
                for (std::size_t byte = 0; byte < 4; ++byte)
                {
                    bytes[axis * 4 + byte] = static_cast<char>((bits >> (byte * 8)) & 0xffU);
                }
                bytes[12 + axis] = static_cast<char>(point.color[axis]);
            }
            stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        return output.commit(error);
    }
} // namespace xjw::aerial_triangulation
