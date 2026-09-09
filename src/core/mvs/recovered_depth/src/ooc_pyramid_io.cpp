#include "metmodel/octree_prepare.hpp"

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfCompression.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfIO.h>
#include <OpenEXR/ImfOpaqueAttribute.h>
#include <OpenEXR/ImfTileDescription.h>
#include <OpenEXR/ImfTiledOutputFile.h>
#include <OpenEXR/OpenEXRConfig.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace metmodel
{
    namespace
    {

        class MemoryInputStream final : public OPENEXR_IMF_NAMESPACE::IStream
        {
        public:
            explicit MemoryInputStream(std::span<const std::byte> bytes)
                : OPENEXR_IMF_NAMESPACE::IStream("ooc-pyramid-exif"), bytes_(bytes)
            {
            }

            bool read(char destination[], int count) override
            {
                if (count < 0)
                    throw std::runtime_error("negative Exif read size");
                const std::size_t size = static_cast<std::size_t>(count);
                if (position_ > bytes_.size() || bytes_.size() - position_ < size)
                {
                    throw std::runtime_error("Exif read exceeds fixed payload");
                }
                std::memcpy(destination, bytes_.data() + position_, size);
                position_ += size;
                return position_ != bytes_.size();
            }

            std::uint64_t tellg() override
            {
                return position_;
            }

            void seekg(std::uint64_t position) override
            {
                if (position > bytes_.size())
                {
                    throw std::runtime_error("Exif seek exceeds fixed payload");
                }
                position_ = static_cast<std::size_t>(position);
            }

        private:
            std::span<const std::byte> bytes_;
            std::size_t position_{};
        };

        class MemoryOutputStream final : public OPENEXR_IMF_NAMESPACE::OStream
        {
        public:
            MemoryOutputStream() : OPENEXR_IMF_NAMESPACE::OStream("ooc-pyramid.exr")
            {
            }

            void write(const char source[], int count) override
            {
                if (count < 0)
                    throw std::runtime_error("negative EXR write size");
                const std::size_t size = static_cast<std::size_t>(count);
                if (position_ > std::numeric_limits<std::size_t>::max() - size)
                {
                    throw std::overflow_error("EXR write size overflows");
                }
                if (bytes_.size() < position_ + size)
                    bytes_.resize(position_ + size);
                std::memcpy(bytes_.data() + position_, source, size);
                position_ += size;
            }

            std::uint64_t tellp() override
            {
                return position_;
            }

            void seekp(std::uint64_t position) override
            {
                if (position > std::numeric_limits<std::size_t>::max())
                {
                    throw std::overflow_error("EXR seek exceeds size_t");
                }
                position_ = static_cast<std::size_t>(position);
                if (bytes_.size() < position_)
                    bytes_.resize(position_);
            }

            [[nodiscard]] std::vector<std::byte> take() &&
            {
                return std::move(bytes_);
            }

        private:
            std::vector<std::byte> bytes_;
            std::size_t position_{};
        };

        void append_u64_le(std::vector<std::byte>& output, std::uint64_t value)
        {
            for (std::size_t byte = 0U; byte != sizeof(value); ++byte)
            {
                output.push_back(static_cast<std::byte>(value >> (8U * byte)));
            }
        }

        void append_bytes(std::vector<std::byte>& output, std::span<const std::byte> bytes)
        {
            if (output.size() > std::numeric_limits<std::size_t>::max() - bytes.size())
            {
                throw std::overflow_error("OOC pyramid payload size overflows");
            }
            output.insert(output.end(), bytes.begin(), bytes.end());
        }

        std::vector<std::byte> encode_float_z(std::size_t width, std::size_t height, std::span<const float> values)
        {
            namespace Imf = OPENEXR_IMF_NAMESPACE;
            if (width == 0U || height == 0U || width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                height > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                width > std::numeric_limits<std::size_t>::max() / height || values.size() != width * height)
            {
                throw std::invalid_argument("invalid OOC pyramid EXR image");
            }

            Imf::Header header(static_cast<int>(width), static_cast<int>(height));
            header.channels().insert("Z", Imf::Channel(Imf::FLOAT));
            header.compression() = Imf::PXR24_COMPRESSION;
            header.setTileDescription(Imf::TileDescription(128U, 128U, Imf::ONE_LEVEL, Imf::ROUND_DOWN));

            constexpr std::array<std::byte, 18> exif{std::byte{0x0e},
                                                     std::byte{0x00},
                                                     std::byte{0x00},
                                                     std::byte{0x00},
                                                     std::byte{'E'},
                                                     std::byte{'x'},
                                                     std::byte{'i'},
                                                     std::byte{'f'},
                                                     std::byte{0x00},
                                                     std::byte{0x00},
                                                     std::byte{'I'},
                                                     std::byte{'I'},
                                                     std::byte{0x2a},
                                                     std::byte{0x00},
                                                     std::byte{0x08},
                                                     std::byte{0x00},
                                                     std::byte{0x00},
                                                     std::byte{0x00}};
            MemoryInputStream exif_stream(exif);
            Imf::OpaqueAttribute exif_attribute("blob");
            exif_attribute.readValueFrom(exif_stream, static_cast<int>(exif.size()), 2);
            header.insert("exif", exif_attribute);

            MemoryOutputStream memory;
            {
                Imf::TiledOutputFile output(memory, header, 1);
                Imf::FrameBuffer frame_buffer;
                frame_buffer.insert("Z",
                                    Imf::Slice(Imf::FLOAT,
                                               const_cast<char*>(reinterpret_cast<const char*>(values.data())),
                                               sizeof(float),
                                               width * sizeof(float)));
                output.setFrameBuffer(frame_buffer);
                output.writeTiles(0, output.numXTiles() - 1, 0, output.numYTiles() - 1);
            }
            return std::move(memory).take();
        }

    } // namespace

    bool ooc_pyramid_exact_encoder_available() noexcept
    {
        return std::string_view(OPENEXR_VERSION_STRING) == "3.2.2" && std::string_view(zlibVersion()) == "1.3.2";
    }

    std::vector<std::byte> serialize_ooc_pyramid_payload(const OocSampleScalePyramidOutput& pyramid)
    {
        if (!ooc_pyramid_exact_encoder_available())
        {
            throw std::runtime_error("target-exact OOC pyramid encoding requires OpenEXR 3.2.2 and zlib 1.3.2");
        }
        if (pyramid.levels.empty() || pyramid.levels.size() > (std::numeric_limits<std::size_t>::max() - 2U) / 4U)
        {
            throw std::invalid_argument("invalid OOC pyramid level count");
        }

        std::vector<std::uint64_t> table;
        table.reserve(4U * pyramid.levels.size() - 2U);
        std::vector<std::byte> blob;
        for (std::size_t level = 0U; level != pyramid.levels.size(); ++level)
        {
            const auto& current = pyramid.levels[level];
            const auto depth = encode_float_z(current.width, current.height, current.depth);
            const auto sample_scale = encode_float_z(current.width, current.height, current.sample_scale);
            table.push_back(static_cast<std::uint64_t>(depth.size()));
            table.push_back(static_cast<std::uint64_t>(sample_scale.size()));
            append_bytes(blob, depth);
            append_bytes(blob, sample_scale);

            if (level + 1U == pyramid.levels.size())
                continue;
            const auto& next = pyramid.levels[level + 1U];
            if (next.width != current.width / 2U || next.height != current.height / 2U ||
                next.width > std::numeric_limits<std::size_t>::max() / std::max<std::size_t>(next.height, 1U) ||
                next.temporary_u8.size() != next.width * next.height)
            {
                throw std::invalid_argument("OOC pyramid interlevel plane does not match the next level");
            }
            table.push_back(static_cast<std::uint64_t>(next.width));
            table.push_back(static_cast<std::uint64_t>(next.height));
            append_bytes(blob, std::as_bytes(std::span<const std::uint8_t>(next.temporary_u8)));
        }

        std::vector<std::byte> result;
        const std::size_t header_words = table.size() + 2U;
        if (header_words > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) ||
            header_words * sizeof(std::uint64_t) > std::numeric_limits<std::size_t>::max() - blob.size())
        {
            throw std::overflow_error("OOC pyramid segment size overflows");
        }
        result.reserve(header_words * sizeof(std::uint64_t) + blob.size());
        append_u64_le(result, static_cast<std::uint64_t>(table.size()));
        for (const std::uint64_t value : table)
            append_u64_le(result, value);
        append_u64_le(result, static_cast<std::uint64_t>(blob.size()));
        append_bytes(result, blob);
        return result;
    }

    OocPyramidBundleSingleShardMode0Output
    serialize_ooc_pyramid_bundle_single_shard_mode0(std::span<const OocPyramidBundleMode0Item> items,
                                                    std::span<const OocSampleScalePyramidOutput> pyramids)
    {
        const OocPyramidBundleMode0Group group{items, pyramids};
        OocPyramidBundleMode0Output grouped =
            serialize_ooc_pyramid_bundle_mode0(std::span<const OocPyramidBundleMode0Group>(&group, 1U));
        OocPyramidBundleSingleShardMode0Output output{std::move(grouped.registry),
                                                      std::move(grouped.payload_shards[0])};
        return output;
    }

    std::vector<OocPyramidCameraGroupRange> partition_ooc_pyramid_camera_groups(std::size_t item_count,
                                                                                std::size_t workitem_size_cameras,
                                                                                std::size_t max_workgroup_size)
    {
        if (item_count == 0U)
        {
            throw std::invalid_argument("OOC pyramid camera partition requires at least one item");
        }
        const std::size_t workitem = std::max<std::size_t>(workitem_size_cameras, 1U);
        const std::size_t maximum_groups = std::max<std::size_t>(max_workgroup_size, 1U);
        const std::size_t groups = std::min(maximum_groups, 1U + (item_count - 1U) / workitem);
        const std::size_t base = item_count / groups;
        const std::size_t remainder = item_count % groups;

        std::vector<OocPyramidCameraGroupRange> ranges;
        ranges.reserve(groups);
        std::size_t begin = 0U;
        for (std::size_t group = 0U; group != groups; ++group)
        {
            const std::size_t count = base + (group < remainder ? 1U : 0U);
            ranges.push_back({begin, count});
            begin += count;
        }
        return ranges;
    }

    OocPyramidWorkerPlan plan_ooc_pyramid_workers(std::uint64_t available_memory_bytes,
                                                  std::uint32_t available_openmp_threads,
                                                  std::size_t max_depthmap_pixels,
                                                  std::size_t camera_task_count)
    {
        if (available_openmp_threads == 0U || max_depthmap_pixels == 0U || camera_task_count == 0U)
        {
            throw std::invalid_argument("OOC pyramid worker planning requires nonzero threads, pixels, "
                                        "and camera tasks");
        }
        constexpr std::uint64_t kMinimumTargetMemory = 0x200000000ULL;
        constexpr std::uint64_t kHalfMemoryThreshold = 0x400000001ULL;
        constexpr std::uint64_t kBytesPerDepthmapPixel = 40ULL;
        if (max_depthmap_pixels > std::numeric_limits<std::uint64_t>::max() / kBytesPerDepthmapPixel)
        {
            throw std::overflow_error("OOC pyramid worker depth-map size overflows target arithmetic");
        }

        const std::uint64_t target_memory =
            available_memory_bytes > kHalfMemoryThreshold ? available_memory_bytes / 2U : kMinimumTargetMemory;
        const std::uint64_t bytes_per_camera = kBytesPerDepthmapPixel * static_cast<std::uint64_t>(max_depthmap_pixels);
        const std::uint64_t raw_memory_threads = target_memory / bytes_per_camera;
        std::uint32_t outer_threads =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(available_openmp_threads, raw_memory_threads));
        if (outer_threads == 0U)
            outer_threads = 1U;

        const std::uint32_t active_camera_slots = static_cast<std::uint32_t>(
            std::min<std::size_t>(camera_task_count, static_cast<std::size_t>(outer_threads)));
        const std::uint32_t inner_threads = 1U + (available_openmp_threads - 1U) / active_camera_slots;
        return {
            target_memory,
            available_openmp_threads,
            max_depthmap_pixels,
            camera_task_count,
            outer_threads,
            active_camera_slots,
            inner_threads,
            outer_threads != available_openmp_threads,
        };
    }

    OocPyramidBundleMode0Output serialize_ooc_pyramid_bundle_mode0(std::span<const OocPyramidBundleMode0Group> groups)
    {
        if (groups.empty())
        {
            throw std::invalid_argument("OOC pyramid bundle requires at least one camera group");
        }

        OocPyramidBundleMode0Output output;
        output.payload_shards.resize(groups.size());
        std::vector<OocPyramidRegistryRawRecord> records;
        std::size_t record_count = 0U;
        for (const auto& group : groups)
        {
            if (group.items.size() != group.pyramids.size())
            {
                throw std::invalid_argument("OOC pyramid bundle item and pyramid counts differ");
            }
            if (group.items.empty() || record_count > std::numeric_limits<std::size_t>::max() - group.items.size())
            {
                throw std::invalid_argument("OOC pyramid bundle contains an empty or oversized group");
            }
            record_count += group.items.size();
        }
        records.reserve(record_count);
        std::unordered_set<std::uint32_t> camera_ids;
        camera_ids.reserve(record_count);
        for (std::size_t group_index = 0U; group_index != groups.size(); ++group_index)
        {
            if (group_index > std::numeric_limits<std::uint64_t>::max())
            {
                throw std::overflow_error("OOC pyramid group index overflows");
            }
            const auto& group = groups[group_index];
            auto& payload = output.payload_shards[group_index];
            for (std::size_t index = 0U; index != group.items.size(); ++index)
            {
                if (!camera_ids.insert(group.items[index].camera_id).second)
                {
                    throw std::invalid_argument("OOC pyramid bundle contains a duplicate camera ID");
                }
                const std::vector<std::byte> segment = serialize_ooc_pyramid_payload(group.pyramids[index]);
                if (payload.size() > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()) ||
                    segment.size() > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()) ||
                    payload.size() > std::numeric_limits<std::size_t>::max() - segment.size())
                {
                    throw std::overflow_error("OOC pyramid shard size overflows");
                }
                const OocPyramidRegistryPayloadLocation location{static_cast<std::uint64_t>(group_index),
                                                                 static_cast<std::uint64_t>(payload.size()),
                                                                 static_cast<std::uint64_t>(segment.size())};
                records.push_back(make_ooc_pyramid_registry_record_mode0(
                    {group.items[index].camera_id, group.items[index].project, location}, group.pyramids[index]));
                append_bytes(payload, segment);
            }
        }
        output.registry = serialize_ooc_pyramid_registry(records);
        return output;
    }

} // namespace metmodel
