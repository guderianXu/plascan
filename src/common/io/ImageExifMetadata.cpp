#include "io/ImageExifMetadata.h"

#include <QFile>

#include <array>
#include <cstdint>
#include <cstring>

namespace xjw::common::io
{
namespace
{

constexpr std::uint16_t kTagMake = 0x010f;
constexpr std::uint16_t kTagModel = 0x0110;
constexpr std::uint16_t kTagExifIfd = 0x8769;
constexpr std::uint16_t kTagFocalLength = 0x920a;
constexpr std::uint16_t kTagFocalLength35 = 0xa405;

class TiffView
{
public:
    TiffView(const unsigned char *data, std::size_t size, bool littleEndian)
        : _data(data), _size(size), _littleEndian(littleEndian)
    {
    }

    bool read16(std::size_t offset, std::uint16_t *value) const
    {
        if (!value || offset + 2 > _size)
        {
            return false;
        }
        if (_littleEndian)
        {
            *value = static_cast<std::uint16_t>(_data[offset]) |
                     (static_cast<std::uint16_t>(_data[offset + 1]) << 8);
        }
        else
        {
            *value = (static_cast<std::uint16_t>(_data[offset]) << 8) |
                     static_cast<std::uint16_t>(_data[offset + 1]);
        }
        return true;
    }

    bool read32(std::size_t offset, std::uint32_t *value) const
    {
        if (!value || offset + 4 > _size)
        {
            return false;
        }
        *value = 0;
        for (int index = 0; index < 4; ++index)
        {
            const int shift = _littleEndian ? index * 8 : (3 - index) * 8;
            *value |= static_cast<std::uint32_t>(_data[offset + index]) << shift;
        }
        return true;
    }

    QString readAscii(std::size_t entryOffset) const
    {
        std::uint32_t count = 0;
        std::uint32_t valueOffset = 0;
        if (!read32(entryOffset + 4, &count) || count == 0 ||
            !read32(entryOffset + 8, &valueOffset))
        {
            return {};
        }
        const std::size_t offset = count <= 4 ? entryOffset + 8 : valueOffset;
        if (offset + count > _size)
        {
            return {};
        }
        const QByteArray text(reinterpret_cast<const char *>(_data + offset),
                              static_cast<qsizetype>(count));
        return QString::fromLatin1(text.constData()).trimmed();
    }

    std::optional<double> readNumber(std::size_t entryOffset) const
    {
        std::uint16_t type = 0;
        std::uint32_t count = 0;
        std::uint32_t valueOffset = 0;
        if (!read16(entryOffset + 2, &type) || !read32(entryOffset + 4, &count) ||
            count == 0 || !read32(entryOffset + 8, &valueOffset))
        {
            return std::nullopt;
        }
        if (type == 3)
        {
            std::uint16_t value = 0;
            return read16(entryOffset + 8, &value)
                ? std::optional<double>(static_cast<double>(value)) : std::nullopt;
        }
        if (type == 4)
        {
            return static_cast<double>(valueOffset);
        }
        if (type == 5 && valueOffset + 8 <= _size)
        {
            std::uint32_t numerator = 0;
            std::uint32_t denominator = 0;
            if (read32(valueOffset, &numerator) && read32(valueOffset + 4, &denominator) &&
                denominator != 0)
            {
                return static_cast<double>(numerator) / static_cast<double>(denominator);
            }
        }
        return std::nullopt;
    }

    bool validRange(std::size_t offset, std::size_t length) const
    {
        return offset <= _size && length <= _size - offset;
    }

private:
    const unsigned char *_data = nullptr;
    std::size_t _size = 0;
    bool _littleEndian = true;
};

void parseIfd(const TiffView &view,
              std::uint32_t ifdOffset,
              ImageExifMetadata *metadata,
              std::uint32_t *exifIfdOffset)
{
    std::uint16_t count = 0;
    if (!metadata || !view.read16(ifdOffset, &count) ||
        !view.validRange(ifdOffset + 2, static_cast<std::size_t>(count) * 12))
    {
        return;
    }
    for (std::uint16_t index = 0; index < count; ++index)
    {
        const std::size_t entry = ifdOffset + 2 + static_cast<std::size_t>(index) * 12;
        std::uint16_t tag = 0;
        if (!view.read16(entry, &tag))
        {
            continue;
        }
        if (tag == kTagMake)
        {
            metadata->make = view.readAscii(entry);
        }
        else if (tag == kTagModel)
        {
            metadata->model = view.readAscii(entry);
        }
        else if (tag == kTagFocalLength)
        {
            metadata->focalLengthMm = view.readNumber(entry);
        }
        else if (tag == kTagFocalLength35)
        {
            metadata->focalLength35Mm = view.readNumber(entry);
        }
        else if (tag == kTagExifIfd && exifIfdOffset)
        {
            std::uint32_t offset = 0;
            if (view.read32(entry + 8, &offset))
            {
                *exifIfdOffset = offset;
            }
        }
    }
}

std::optional<ImageExifMetadata> parseExifSegment(const unsigned char *data, std::size_t size)
{
    if (!data || size < 14 || std::memcmp(data, "Exif\0\0", 6) != 0)
    {
        return std::nullopt;
    }
    const unsigned char *tiff = data + 6;
    const std::size_t tiffSize = size - 6;
    const bool littleEndian = tiff[0] == 'I' && tiff[1] == 'I';
    const bool bigEndian = tiff[0] == 'M' && tiff[1] == 'M';
    if (!littleEndian && !bigEndian)
    {
        return std::nullopt;
    }
    const TiffView view(tiff, tiffSize, littleEndian);
    std::uint16_t magic = 0;
    std::uint32_t ifd0 = 0;
    if (!view.read16(2, &magic) || magic != 42 || !view.read32(4, &ifd0))
    {
        return std::nullopt;
    }

    ImageExifMetadata metadata;
    std::uint32_t exifIfd = 0;
    parseIfd(view, ifd0, &metadata, &exifIfd);
    if (exifIfd != 0)
    {
        parseIfd(view, exifIfd, &metadata, nullptr);
    }
    if (metadata.make.isEmpty() && metadata.model.isEmpty() &&
        !metadata.focalLengthMm.has_value() && !metadata.focalLength35Mm.has_value())
    {
        return std::nullopt;
    }
    return metadata;
}

} // namespace

std::optional<ImageExifMetadata> readImageExifMetadata(const QString &path,
                                                       QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法打开影像 EXIF: %1").arg(path);
        }
        return std::nullopt;
    }

    const QByteArray bytes = file.read(1024 * 1024);
    const auto *data = reinterpret_cast<const unsigned char *>(bytes.constData());
    const std::size_t size = static_cast<std::size_t>(bytes.size());
    if (size < 4 || data[0] != 0xff || data[1] != 0xd8)
    {
        return std::nullopt;
    }
    std::size_t offset = 2;
    while (offset + 4 <= size)
    {
        if (data[offset] != 0xff)
        {
            ++offset;
            continue;
        }
        const unsigned char marker = data[offset + 1];
        offset += 2;
        if (marker == 0xda || marker == 0xd9)
        {
            break;
        }
        const std::uint16_t segmentLength =
            (static_cast<std::uint16_t>(data[offset]) << 8) | data[offset + 1];
        if (segmentLength < 2 || offset + segmentLength > size)
        {
            break;
        }
        if (marker == 0xe1)
        {
            const auto metadata = parseExifSegment(data + offset + 2, segmentLength - 2);
            if (metadata.has_value())
            {
                return metadata;
            }
        }
        offset += segmentLength;
    }
    return std::nullopt;
}

} // namespace xjw::common::io
