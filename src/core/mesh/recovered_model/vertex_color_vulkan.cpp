#include "metmodel/mesh.hpp"
#include "VertexColorOptions.h"

#include <vulkan/vulkan.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

    QString qt_path(const std::filesystem::path& path)
    {
        const auto utf8 = path.u8string();
        return QString::fromUtf8(reinterpret_cast<const char*>(utf8.data()), static_cast<qsizetype>(utf8.size()));
    }

    void vk_check(VkResult result, const char* operation)
    {
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error(std::string(operation) + " failed with Vulkan code " + std::to_string(result));
        }
    }

    std::vector<std::byte> read_bytes(const std::filesystem::path& path)
    {
        QFile stream(qt_path(path));
        if (!stream.open(QIODevice::ReadOnly))
            throw std::runtime_error("cannot open shader " + path.string());
        const qint64 size = stream.size();
        if (size <= 0 || size > 16 * 1024 * 1024)
            throw std::runtime_error("invalid shader size " + path.string());
        std::vector<std::byte> result(static_cast<std::size_t>(size));
        if (size != 0)
        {
            if (stream.read(reinterpret_cast<char*>(result.data()), size) != size)
                throw std::runtime_error("cannot read " + path.string());
        }
        return result;
    }

    template <typename T> std::vector<T> read_vector(const std::filesystem::path& path)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto bytes = read_bytes(path);
        if (bytes.size() % sizeof(T) != 0U)
            throw std::runtime_error("invalid element-aligned file " + path.string());
        std::vector<T> result(bytes.size() / sizeof(T));
        if (!bytes.empty())
            std::memcpy(result.data(), bytes.data(), bytes.size());
        return result;
    }

    std::filesystem::path find_shader(const std::filesystem::path& directory, const std::string& prefix)
    {
        const auto path = directory / (prefix + "recovered.spv");
        if (QFileInfo::exists(qt_path(path)))
            return path;
        throw std::runtime_error("missing shader prefix " + prefix);
    }

    struct Buffer
    {
        VkDevice device = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0U;

        Buffer() = default;
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept
        {
            *this = std::move(other);
        }
        Buffer& operator=(Buffer&& other) noexcept
        {
            if (this == &other)
                return *this;
            release();
            device = other.device;
            buffer = other.buffer;
            memory = other.memory;
            size = other.size;
            other.device = VK_NULL_HANDLE;
            other.buffer = VK_NULL_HANDLE;
            other.memory = VK_NULL_HANDLE;
            other.size = 0U;
            return *this;
        }
        ~Buffer()
        {
            release();
        }

        void release()
        {
            if (buffer != VK_NULL_HANDLE)
                vkDestroyBuffer(device, buffer, nullptr);
            if (memory != VK_NULL_HANDLE)
                vkFreeMemory(device, memory, nullptr);
            buffer = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
        }

        void upload(const void* source, std::size_t bytes) const
        {
            if (bytes > size)
                throw std::runtime_error("buffer upload exceeds size");
            void* mapped = nullptr;
            vk_check(vkMapMemory(device, memory, 0U, size, 0U, &mapped), "vkMapMemory(upload)");
            std::memcpy(mapped, source, bytes);
            if (bytes < size)
                std::memset(static_cast<std::byte*>(mapped) + bytes, 0, static_cast<std::size_t>(size) - bytes);
            vkUnmapMemory(device, memory);
        }

        template <typename T> std::vector<T> download(std::size_t count) const
        {
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(T) || count * sizeof(T) > size)
            {
                throw std::runtime_error("buffer download exceeds size");
            }
            void* mapped = nullptr;
            vk_check(vkMapMemory(device, memory, 0U, size, 0U, &mapped), "vkMapMemory(download)");
            std::vector<T> result(count);
            std::memcpy(result.data(), mapped, count * sizeof(T));
            vkUnmapMemory(device, memory);
            return result;
        }
    };

    struct Image
    {
        VkDevice device = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;

        Image() = default;
        Image(const Image&) = delete;
        Image& operator=(const Image&) = delete;
        Image(Image&& other) noexcept
        {
            *this = std::move(other);
        }
        Image& operator=(Image&& other) noexcept
        {
            if (this == &other)
                return *this;
            release();
            device = other.device;
            image = other.image;
            memory = other.memory;
            view = other.view;
            sampler = other.sampler;
            other.device = VK_NULL_HANDLE;
            other.image = VK_NULL_HANDLE;
            other.memory = VK_NULL_HANDLE;
            other.view = VK_NULL_HANDLE;
            other.sampler = VK_NULL_HANDLE;
            return *this;
        }
        ~Image()
        {
            release();
        }

        void release()
        {
            if (sampler)
                vkDestroySampler(device, sampler, nullptr);
            if (view)
                vkDestroyImageView(device, view, nullptr);
            if (image)
                vkDestroyImage(device, image, nullptr);
            if (memory)
                vkFreeMemory(device, memory, nullptr);
            sampler = VK_NULL_HANDLE;
            view = VK_NULL_HANDLE;
            image = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
        }
    };

    struct RasterTargets
    {
        VkDevice device = VK_NULL_HANDLE;
        VkImage depth = VK_NULL_HANDLE;
        VkDeviceMemory depth_memory = VK_NULL_HANDLE;
        VkImageView depth_framebuffer_view = VK_NULL_HANDLE;
        VkImage color = VK_NULL_HANDLE;
        VkDeviceMemory color_memory = VK_NULL_HANDLE;
        VkImageView color_framebuffer_view = VK_NULL_HANDLE;
        VkImageView color_storage_view = VK_NULL_HANDLE;
        VkRenderPass render_pass = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;

        RasterTargets() = default;
        RasterTargets(const RasterTargets&) = delete;
        RasterTargets& operator=(const RasterTargets&) = delete;
        RasterTargets(RasterTargets&& other) noexcept
            : device(std::exchange(other.device, VK_NULL_HANDLE)), depth(std::exchange(other.depth, VK_NULL_HANDLE)),
              depth_memory(std::exchange(other.depth_memory, VK_NULL_HANDLE)),
              depth_framebuffer_view(std::exchange(other.depth_framebuffer_view, VK_NULL_HANDLE)),
              color(std::exchange(other.color, VK_NULL_HANDLE)),
              color_memory(std::exchange(other.color_memory, VK_NULL_HANDLE)),
              color_framebuffer_view(std::exchange(other.color_framebuffer_view, VK_NULL_HANDLE)),
              color_storage_view(std::exchange(other.color_storage_view, VK_NULL_HANDLE)),
              render_pass(std::exchange(other.render_pass, VK_NULL_HANDLE)),
              framebuffer(std::exchange(other.framebuffer, VK_NULL_HANDLE))
        {
        }
        ~RasterTargets()
        {
            if (framebuffer)
                vkDestroyFramebuffer(device, framebuffer, nullptr);
            if (render_pass)
                vkDestroyRenderPass(device, render_pass, nullptr);
            if (color_storage_view)
                vkDestroyImageView(device, color_storage_view, nullptr);
            if (color_framebuffer_view)
                vkDestroyImageView(device, color_framebuffer_view, nullptr);
            if (depth_framebuffer_view)
                vkDestroyImageView(device, depth_framebuffer_view, nullptr);
            if (color)
                vkDestroyImage(device, color, nullptr);
            if (depth)
                vkDestroyImage(device, depth, nullptr);
            if (color_memory)
                vkFreeMemory(device, color_memory, nullptr);
            if (depth_memory)
                vkFreeMemory(device, depth_memory, nullptr);
        }
    };

    class Replay
    {
    public:
        explicit Replay(const metmodel::RecoveredVertexColorVulkanOptions& options)
        {
            try
            {
                VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
                application.pApplicationName = "metmodel-vertex-color-replay";
                application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
                application.pEngineName = "metmodel";
                application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
                application.apiVersion = VK_API_VERSION_1_1;
                VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
                instance_info.pApplicationInfo = &application;
                vk_check(vkCreateInstance(&instance_info, nullptr, &_instance), "vkCreateInstance");

                std::uint32_t device_count = 0U;
                vk_check(vkEnumeratePhysicalDevices(_instance, &device_count, nullptr),
                         "vkEnumeratePhysicalDevices(count)");
                std::vector<VkPhysicalDevice> devices(device_count);
                vk_check(vkEnumeratePhysicalDevices(_instance, &device_count, devices.data()),
                         "vkEnumeratePhysicalDevices(list)");
                for (const VkPhysicalDevice candidate : devices)
                {
                    VkPhysicalDeviceIDProperties identity{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                    VkPhysicalDeviceProperties2 device_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                    device_properties.pNext = &identity;
                    vkGetPhysicalDeviceProperties2(candidate, &device_properties);
                    const auto& properties = device_properties.properties;
                    if (options.deviceUuid &&
                        !std::equal(options.deviceUuid->begin(), options.deviceUuid->end(), identity.deviceUUID))
                        continue;
                    std::uint32_t queue_count = 0U;
                    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
                    std::vector<VkQueueFamilyProperties> queues(queue_count);
                    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
                    for (std::uint32_t family = 0U; family != queue_count; ++family)
                    {
                        constexpr VkQueueFlags required = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
                        if ((queues[family].queueFlags & required) != required)
                            continue;
                        if (_physical == VK_NULL_HANDLE ||
                            std::string(properties.deviceName).find("NVIDIA") != std::string::npos)
                        {
                            _physical = candidate;
                            _queueFamily = family;
                            _deviceName = properties.deviceName;
                        }
                        if (std::string(properties.deviceName).find("NVIDIA") != std::string::npos)
                        {
                            break;
                        }
                    }
                    if (_deviceName.find("NVIDIA") != std::string::npos)
                        break;
                }
                if (_physical == VK_NULL_HANDLE)
                    throw std::runtime_error(
                        "no Vulkan graphics/compute device matching the selected CUDA device UUID");
                vkGetPhysicalDeviceMemoryProperties(_physical, &_memoryProperties);
                const float priority = 1.0F;
                VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
                queue_info.queueFamilyIndex = _queueFamily;
                queue_info.queueCount = 1U;
                queue_info.pQueuePriorities = &priority;
                VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
                device_info.queueCreateInfoCount = 1U;
                device_info.pQueueCreateInfos = &queue_info;
                vk_check(vkCreateDevice(_physical, &device_info, nullptr, &_device), "vkCreateDevice");
                vkGetDeviceQueue(_device, _queueFamily, 0U, &_queue);

                VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                pool_info.queueFamilyIndex = _queueFamily;
                vk_check(vkCreateCommandPool(_device, &pool_info, nullptr, &_commandPool), "vkCreateCommandPool");
            }
            catch (...)
            {
                release();
                throw;
            }
        }

        Replay(const Replay&) = delete;
        Replay& operator=(const Replay&) = delete;

        ~Replay()
        {
            release();
        }

        void release()
        {
            if (_device != VK_NULL_HANDLE)
                vkDeviceWaitIdle(_device);
            if (_commandPool != VK_NULL_HANDLE)
                vkDestroyCommandPool(_device, _commandPool, nullptr);
            if (_device != VK_NULL_HANDLE)
                vkDestroyDevice(_device, nullptr);
            if (_instance != VK_NULL_HANDLE)
                vkDestroyInstance(_instance, nullptr);
            _commandPool = VK_NULL_HANDLE;
            _device = VK_NULL_HANDLE;
            _instance = VK_NULL_HANDLE;
        }

        const std::string& device_name() const
        {
            return _deviceName;
        }
        VkDevice device() const
        {
            return _device;
        }

        Buffer make_buffer(VkDeviceSize size,
                           const void* initial = nullptr,
                           std::size_t initial_bytes = 0U,
                           VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) const
        {
            if (size == 0U)
                throw std::runtime_error("zero-sized Vulkan buffer");
            Buffer result;
            result.device = _device;
            result.size = size;
            VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            buffer_info.size = size;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            vk_check(vkCreateBuffer(_device, &buffer_info, nullptr, &result.buffer), "vkCreateBuffer");
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(_device, result.buffer, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex =
                memory_type(requirements.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vk_check(vkAllocateMemory(_device, &allocation, nullptr, &result.memory), "vkAllocateMemory");
            vk_check(vkBindBufferMemory(_device, result.buffer, result.memory, 0U), "vkBindBufferMemory");
            if (initial != nullptr)
                result.upload(initial, initial_bytes);
            else
            {
                std::vector<std::byte> zero(static_cast<std::size_t>(size));
                result.upload(zero.data(), zero.size());
            }
            return result;
        }

        VkShaderModule make_shader(const std::filesystem::path& path) const
        {
            const auto code = read_vector<std::uint32_t>(path);
            if (code.empty() || code.front() != 0x07230203U)
                throw std::runtime_error("invalid SPIR-V " + path.string());
            VkShaderModuleCreateInfo create{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            create.codeSize = code.size() * sizeof(std::uint32_t);
            create.pCode = code.data();
            VkShaderModule result = VK_NULL_HANDLE;
            vk_check(vkCreateShaderModule(_device, &create, nullptr, &result), "vkCreateShaderModule");
            return result;
        }

        VkCommandBuffer make_command_buffer() const
        {
            VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocation.commandPool = _commandPool;
            allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocation.commandBufferCount = 1U;
            VkCommandBuffer result = VK_NULL_HANDLE;
            vk_check(vkAllocateCommandBuffers(_device, &allocation, &result), "vkAllocateCommandBuffers");
            return result;
        }

        void submit(VkCommandBuffer command) const
        {
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1U;
            submit.pCommandBuffers = &command;
            vk_check(vkQueueSubmit(_queue, 1U, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
            vk_check(vkQueueWaitIdle(_queue), "vkQueueWaitIdle");
            vkFreeCommandBuffers(_device, _commandPool, 1U, &command);
        }

        std::uint32_t memory_type(std::uint32_t mask, VkMemoryPropertyFlags flags) const
        {
            for (std::uint32_t index = 0U; index != _memoryProperties.memoryTypeCount; ++index)
            {
                if ((mask & (1U << index)) != 0U &&
                    (_memoryProperties.memoryTypes[index].propertyFlags & flags) == flags)
                {
                    return index;
                }
            }
            throw std::runtime_error("no host-coherent Vulkan memory type");
        }

    private:
        VkInstance _instance = VK_NULL_HANDLE;
        VkPhysicalDevice _physical = VK_NULL_HANDLE;
        VkPhysicalDeviceMemoryProperties _memoryProperties{};
        std::uint32_t _queueFamily = 0U;
        VkDevice _device = VK_NULL_HANDLE;
        VkQueue _queue = VK_NULL_HANDLE;
        VkCommandPool _commandPool = VK_NULL_HANDLE;
        std::string _deviceName;
    };

    RasterTargets make_raster_targets(const Replay& replay, std::uint32_t width, std::uint32_t height)
    {
        RasterTargets result;
        result.device = replay.device();
        const auto make_image = [&](VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory)
        {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = format;
            info.extent = {width, height, 1U};
            info.mipLevels = 1U;
            info.arrayLayers = 1U;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = usage;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            vk_check(vkCreateImage(replay.device(), &info, nullptr, &image), "vkCreateImage(raster)");
            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(replay.device(), image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex =
                replay.memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            vk_check(vkAllocateMemory(replay.device(), &allocation, nullptr, &memory), "vkAllocateMemory(raster)");
            vk_check(vkBindImageMemory(replay.device(), image, memory, 0U), "vkBindImageMemory(raster)");
        };
        make_image(VK_FORMAT_D32_SFLOAT,
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                   result.depth,
                   result.depth_memory);
        make_image(VK_FORMAT_R32_SFLOAT,
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                   result.color,
                   result.color_memory);
        const auto make_view =
            [&](VkImage image, VkFormat format, VkImageAspectFlags aspect, VkImageViewType type, VkImageView& view)
        {
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            info.image = image;
            info.viewType = type;
            info.format = format;
            info.subresourceRange = {aspect, 0U, VK_REMAINING_MIP_LEVELS, 0U, 1U};
            vk_check(vkCreateImageView(replay.device(), &info, nullptr, &view), "vkCreateImageView(raster)");
        };
        make_view(result.depth,
                  VK_FORMAT_D32_SFLOAT,
                  VK_IMAGE_ASPECT_DEPTH_BIT,
                  VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                  result.depth_framebuffer_view);
        make_view(result.color,
                  VK_FORMAT_R32_SFLOAT,
                  VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                  result.color_framebuffer_view);
        make_view(result.color,
                  VK_FORMAT_R32_SFLOAT,
                  VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_VIEW_TYPE_2D,
                  result.color_storage_view);

        const std::array attachments{VkAttachmentDescription{0U,
                                                             VK_FORMAT_D32_SFLOAT,
                                                             VK_SAMPLE_COUNT_1_BIT,
                                                             VK_ATTACHMENT_LOAD_OP_CLEAR,
                                                             VK_ATTACHMENT_STORE_OP_STORE,
                                                             VK_ATTACHMENT_LOAD_OP_CLEAR,
                                                             VK_ATTACHMENT_STORE_OP_STORE,
                                                             VK_IMAGE_LAYOUT_GENERAL,
                                                             VK_IMAGE_LAYOUT_GENERAL},
                                     VkAttachmentDescription{0U,
                                                             VK_FORMAT_R32_SFLOAT,
                                                             VK_SAMPLE_COUNT_1_BIT,
                                                             VK_ATTACHMENT_LOAD_OP_CLEAR,
                                                             VK_ATTACHMENT_STORE_OP_STORE,
                                                             VK_ATTACHMENT_LOAD_OP_CLEAR,
                                                             VK_ATTACHMENT_STORE_OP_STORE,
                                                             VK_IMAGE_LAYOUT_GENERAL,
                                                             VK_IMAGE_LAYOUT_GENERAL}};
        const VkAttachmentReference color_reference{1U, VK_IMAGE_LAYOUT_GENERAL};
        const VkAttachmentReference depth_reference{0U, VK_IMAGE_LAYOUT_GENERAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1U;
        subpass.pColorAttachments = &color_reference;
        subpass.pDepthStencilAttachment = &depth_reference;
        VkRenderPassCreateInfo render_pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        render_pass.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        render_pass.pAttachments = attachments.data();
        render_pass.subpassCount = 1U;
        render_pass.pSubpasses = &subpass;
        vk_check(vkCreateRenderPass(replay.device(), &render_pass, nullptr, &result.render_pass),
                 "vkCreateRenderPass(raster)");
        const std::array views{result.depth_framebuffer_view, result.color_framebuffer_view};
        VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = result.render_pass;
        framebuffer.attachmentCount = static_cast<std::uint32_t>(views.size());
        framebuffer.pAttachments = views.data();
        framebuffer.width = width;
        framebuffer.height = height;
        framebuffer.layers = 1U;
        vk_check(vkCreateFramebuffer(replay.device(), &framebuffer, nullptr, &result.framebuffer),
                 "vkCreateFramebuffer(raster)");

        VkCommandBuffer command = replay.make_command_buffer();
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(raster layout)");
        std::array<VkImageMemoryBarrier, 2> barriers{};
        barriers[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = result.depth;
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 1U, 0U, 1U};
        barriers[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].image = result.color;
        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
        barriers[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0U,
                             0U,
                             nullptr,
                             0U,
                             nullptr,
                             static_cast<std::uint32_t>(barriers.size()),
                             barriers.data());
        vk_check(vkEndCommandBuffer(command), "vkEndCommandBuffer(raster layout)");
        replay.submit(command);
        return result;
    }

    Image make_color_image(const Replay& replay,
                           std::span<const std::uint8_t> interleaved,
                           std::uint32_t width,
                           std::uint32_t height)
    {
        constexpr std::uint32_t channels = 3U;
        constexpr std::uint32_t mip_levels = 12U;
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        if (width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 || interleaved.size() != channels * pixels)
        {
            throw std::runtime_error("recovered colorization requires an even-sized RGB image");
        }
        std::vector<std::uint8_t> planar(interleaved.size());
        for (std::size_t pixel = 0U; pixel != pixels; ++pixel)
        {
            for (std::size_t channel = 0U; channel != channels; ++channel)
                planar[channel * pixels + pixel] = interleaved[channels * pixel + channel];
        }
        Buffer staging =
            replay.make_buffer(planar.size(), planar.data(), planar.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        Image result;
        result.device = replay.device();
        VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_R8_UNORM;
        image_info.extent = {width, height, 1U};
        image_info.mipLevels = mip_levels;
        image_info.arrayLayers = channels;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vk_check(vkCreateImage(replay.device(), &image_info, nullptr, &result.image), "vkCreateImage(color)");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(replay.device(), result.image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex =
            replay.memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vk_check(vkAllocateMemory(replay.device(), &allocation, nullptr, &result.memory), "vkAllocateMemory(color)");
        vk_check(vkBindImageMemory(replay.device(), result.image, result.memory, 0U), "vkBindImageMemory(color)");

        VkCommandBuffer command = replay.make_command_buffer();
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(color upload)");
        VkImageMemoryBarrier initial{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        initial.srcAccessMask = 0U;
        initial.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        initial.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        initial.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        initial.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initial.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initial.image = result.image;
        initial.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, mip_levels, 0U, channels};
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0U,
                             0U,
                             nullptr,
                             0U,
                             nullptr,
                             1U,
                             &initial);
        std::array<VkBufferImageCopy, channels> copies{};
        for (std::uint32_t channel = 0U; channel != channels; ++channel)
        {
            copies[channel].bufferOffset = channel * pixels;
            copies[channel].bufferRowLength = width;
            copies[channel].bufferImageHeight = height;
            copies[channel].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, channel, 1U};
            copies[channel].imageExtent = {width, height, 1U};
        }
        vkCmdCopyBufferToImage(command,
                               staging.buffer,
                               result.image,
                               VK_IMAGE_LAYOUT_GENERAL,
                               static_cast<std::uint32_t>(copies.size()),
                               copies.data());
        VkMemoryBarrier transfer_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        transfer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        transfer_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0U,
                             1U,
                             &transfer_barrier,
                             0U,
                             nullptr,
                             0U,
                             nullptr);
        for (std::uint32_t level = 1U; level != mip_levels; ++level)
        {
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1U, 0U, channels};
            blit.srcOffsets[1] = {static_cast<std::int32_t>(std::max(1U, width >> (level - 1U))),
                                  static_cast<std::int32_t>(std::max(1U, height >> (level - 1U))),
                                  1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0U, channels};
            blit.dstOffsets[1] = {static_cast<std::int32_t>(std::max(1U, width >> level)),
                                  static_cast<std::int32_t>(std::max(1U, height >> level)),
                                  1};
            vkCmdBlitImage(command,
                           result.image,
                           VK_IMAGE_LAYOUT_GENERAL,
                           result.image,
                           VK_IMAGE_LAYOUT_GENERAL,
                           1U,
                           &blit,
                           VK_FILTER_LINEAR);
            vkCmdPipelineBarrier(command,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0U,
                                 1U,
                                 &transfer_barrier,
                                 0U,
                                 nullptr,
                                 0U,
                                 nullptr);
        }
        VkMemoryBarrier shader_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        shader_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        shader_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0U,
                             1U,
                             &shader_barrier,
                             0U,
                             nullptr,
                             0U,
                             nullptr);
        vk_check(vkEndCommandBuffer(command), "vkEndCommandBuffer(color upload)");
        replay.submit(command);

        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = result.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        view_info.format = VK_FORMAT_R8_UNORM;
        view_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY};
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, mip_levels, 0U, channels};
        vk_check(vkCreateImageView(replay.device(), &view_info, nullptr, &result.view), "vkCreateImageView(color)");
        VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.minLod = 0.0F;
        sampler_info.maxLod = 1000.0F;
        vk_check(vkCreateSampler(replay.device(), &sampler_info, nullptr, &result.sampler), "vkCreateSampler(color)");
        return result;
    }

    struct Pipeline
    {
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorSetLayout set0_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout debug_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkShaderModule shader = VK_NULL_HANDLE;

        Pipeline() = default;
        Pipeline(const Pipeline&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;
        Pipeline(Pipeline&& other) noexcept
        {
            *this = std::move(other);
        }
        Pipeline& operator=(Pipeline&& other) noexcept
        {
            if (this == &other)
                return *this;
            release();
            device = other.device;
            set0_layout = other.set0_layout;
            debug_layout = other.debug_layout;
            layout = other.layout;
            pipeline = other.pipeline;
            shader = other.shader;
            other.device = VK_NULL_HANDLE;
            other.set0_layout = VK_NULL_HANDLE;
            other.debug_layout = VK_NULL_HANDLE;
            other.layout = VK_NULL_HANDLE;
            other.pipeline = VK_NULL_HANDLE;
            other.shader = VK_NULL_HANDLE;
            return *this;
        }
        ~Pipeline()
        {
            release();
        }

        void release()
        {
            if (pipeline)
                vkDestroyPipeline(device, pipeline, nullptr);
            if (layout)
                vkDestroyPipelineLayout(device, layout, nullptr);
            if (set0_layout)
                vkDestroyDescriptorSetLayout(device, set0_layout, nullptr);
            if (debug_layout)
                vkDestroyDescriptorSetLayout(device, debug_layout, nullptr);
            if (shader)
                vkDestroyShaderModule(device, shader, nullptr);
            set0_layout = VK_NULL_HANDLE;
            debug_layout = VK_NULL_HANDLE;
            layout = VK_NULL_HANDLE;
            pipeline = VK_NULL_HANDLE;
            shader = VK_NULL_HANDLE;
        }
    };

    struct GraphicsPipeline
    {
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorSetLayout set0_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout debug_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkShaderModule vertex_shader = VK_NULL_HANDLE;
        VkShaderModule fragment_shader = VK_NULL_HANDLE;

        GraphicsPipeline() = default;
        GraphicsPipeline(const GraphicsPipeline&) = delete;
        GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
        GraphicsPipeline(GraphicsPipeline&& other) noexcept
            : device(std::exchange(other.device, VK_NULL_HANDLE)),
              set0_layout(std::exchange(other.set0_layout, VK_NULL_HANDLE)),
              debug_layout(std::exchange(other.debug_layout, VK_NULL_HANDLE)),
              layout(std::exchange(other.layout, VK_NULL_HANDLE)),
              pipeline(std::exchange(other.pipeline, VK_NULL_HANDLE)),
              vertex_shader(std::exchange(other.vertex_shader, VK_NULL_HANDLE)),
              fragment_shader(std::exchange(other.fragment_shader, VK_NULL_HANDLE))
        {
        }
        ~GraphicsPipeline()
        {
            if (pipeline)
                vkDestroyPipeline(device, pipeline, nullptr);
            if (layout)
                vkDestroyPipelineLayout(device, layout, nullptr);
            if (set0_layout)
                vkDestroyDescriptorSetLayout(device, set0_layout, nullptr);
            if (debug_layout)
                vkDestroyDescriptorSetLayout(device, debug_layout, nullptr);
            if (vertex_shader)
                vkDestroyShaderModule(device, vertex_shader, nullptr);
            if (fragment_shader)
                vkDestroyShaderModule(device, fragment_shader, nullptr);
        }
    };

    GraphicsPipeline make_graphics_pipeline(const Replay& replay,
                                            const std::filesystem::path& vertex_shader_path,
                                            const std::filesystem::path& fragment_shader_path,
                                            VkRenderPass render_pass)
    {
        GraphicsPipeline result;
        result.device = replay.device();
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        for (std::uint32_t index = 0U; index != bindings.size(); ++index)
        {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1U;
            bindings[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set0.bindingCount = static_cast<std::uint32_t>(bindings.size());
        set0.pBindings = bindings.data();
        vk_check(vkCreateDescriptorSetLayout(replay.device(), &set0, nullptr, &result.set0_layout),
                 "vkCreateDescriptorSetLayout(graphics set0)");
        VkDescriptorSetLayoutBinding debug_binding{};
        debug_binding.binding = 0U;
        debug_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        debug_binding.descriptorCount = 1U;
        debug_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo debug{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        debug.bindingCount = 1U;
        debug.pBindings = &debug_binding;
        vk_check(vkCreateDescriptorSetLayout(replay.device(), &debug, nullptr, &result.debug_layout),
                 "vkCreateDescriptorSetLayout(graphics debug)");
        const std::array layouts{result.set0_layout, result.debug_layout};
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;
        push.size = 8U;
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = static_cast<std::uint32_t>(layouts.size());
        layout.pSetLayouts = layouts.data();
        layout.pushConstantRangeCount = 1U;
        layout.pPushConstantRanges = &push;
        vk_check(vkCreatePipelineLayout(replay.device(), &layout, nullptr, &result.layout),
                 "vkCreatePipelineLayout(graphics)");
        result.vertex_shader = replay.make_shader(vertex_shader_path);
        result.fragment_shader = replay.make_shader(fragment_shader_path);
        const std::array stages{VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                                nullptr,
                                                                0U,
                                                                VK_SHADER_STAGE_VERTEX_BIT,
                                                                result.vertex_shader,
                                                                "main",
                                                                nullptr},
                                VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                                nullptr,
                                                                0U,
                                                                VK_SHADER_STAGE_FRAGMENT_BIT,
                                                                result.fragment_shader,
                                                                "main",
                                                                nullptr}};
        const VkVertexInputBindingDescription vertex_binding{0U, 12U, VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription vertex_attribute{0U, 0U, VK_FORMAT_R32G32B32_SFLOAT, 0U};
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertex_input.vertexBindingDescriptionCount = 1U;
        vertex_input.pVertexBindingDescriptions = &vertex_binding;
        vertex_input.vertexAttributeDescriptionCount = 1U;
        vertex_input.pVertexAttributeDescriptions = &vertex_attribute;
        VkPipelineInputAssemblyStateCreateInfo input_assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1U;
        viewport.scissorCount = 1U;
        VkPipelineRasterizationStateCreateInfo rasterization{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0F;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendAttachmentState attachment{};
        attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1U;
        blend.pAttachments = &attachment;
        const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();
        VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
        pipeline.pStages = stages.data();
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input_assembly;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &rasterization;
        pipeline.pMultisampleState = &multisample;
        pipeline.pDepthStencilState = &depth;
        pipeline.pColorBlendState = &blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = result.layout;
        pipeline.renderPass = render_pass;
        pipeline.subpass = 0U;
        vk_check(vkCreateGraphicsPipelines(replay.device(), VK_NULL_HANDLE, 1U, &pipeline, nullptr, &result.pipeline),
                 "vkCreateGraphicsPipelines");
        return result;
    }

    Pipeline make_pipeline(const Replay& replay,
                           const std::filesystem::path& shader_path,
                           std::span<const VkDescriptorType> descriptor_types,
                           std::uint32_t push_constant_bytes)
    {
        Pipeline result;
        result.device = replay.device();
        std::vector<VkDescriptorSetLayoutBinding> bindings(descriptor_types.size());
        for (std::uint32_t index = 0U; index != descriptor_types.size(); ++index)
        {
            bindings[index].binding = index;
            bindings[index].descriptorType = descriptor_types[index];
            bindings[index].descriptorCount = 1U;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set0.bindingCount = static_cast<std::uint32_t>(bindings.size());
        set0.pBindings = bindings.data();
        vk_check(vkCreateDescriptorSetLayout(replay.device(), &set0, nullptr, &result.set0_layout),
                 "vkCreateDescriptorSetLayout(set0)");
        VkDescriptorSetLayoutBinding debug_binding{};
        debug_binding.binding = 0U;
        debug_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        debug_binding.descriptorCount = 1U;
        debug_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo debug{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        debug.bindingCount = 1U;
        debug.pBindings = &debug_binding;
        vk_check(vkCreateDescriptorSetLayout(replay.device(), &debug, nullptr, &result.debug_layout),
                 "vkCreateDescriptorSetLayout(debug)");
        const std::array layouts{result.set0_layout, result.debug_layout};
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = push_constant_bytes;
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = static_cast<std::uint32_t>(layouts.size());
        layout.pSetLayouts = layouts.data();
        layout.pushConstantRangeCount = 1U;
        layout.pPushConstantRanges = &push;
        vk_check(vkCreatePipelineLayout(replay.device(), &layout, nullptr, &result.layout), "vkCreatePipelineLayout");
        result.shader = replay.make_shader(shader_path);
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = result.shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = stage;
        pipeline.layout = result.layout;
        vk_check(vkCreateComputePipelines(replay.device(), VK_NULL_HANDLE, 1U, &pipeline, nullptr, &result.pipeline),
                 "vkCreateComputePipelines");
        return result;
    }

    Pipeline make_storage_pipeline(const Replay& replay,
                                   const std::filesystem::path& shader_path,
                                   std::uint32_t storage_bindings,
                                   std::uint32_t push_constant_bytes)
    {
        std::vector<VkDescriptorType> types(storage_bindings, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        return make_pipeline(replay, shader_path, types, push_constant_bytes);
    }

    struct DescriptorPool
    {
        VkDevice device = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        ~DescriptorPool()
        {
            if (pool)
                vkDestroyDescriptorPool(device, pool, nullptr);
        }
    };

    VkDescriptorSet allocate_set(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout)
    {
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool;
        allocation.descriptorSetCount = 1U;
        allocation.pSetLayouts = &layout;
        VkDescriptorSet result = VK_NULL_HANDLE;
        vk_check(vkAllocateDescriptorSets(device, &allocation, &result), "vkAllocateDescriptorSets");
        return result;
    }

    void update_storage_set(VkDevice device, VkDescriptorSet set, std::span<const Buffer* const> buffers)
    {
        std::vector<VkDescriptorBufferInfo> infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        for (std::size_t index = 0U; index != buffers.size(); ++index)
        {
            infos[index] = {buffers[index]->buffer, 0U, VK_WHOLE_SIZE};
            writes[index] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[index].dstSet = set;
            writes[index].dstBinding = static_cast<std::uint32_t>(index);
            writes[index].descriptorCount = 1U;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = &infos[index];
        }
        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0U, nullptr);
    }

    void update_projection_set(VkDevice device,
                               VkDescriptorSet set,
                               const Buffer& positions,
                               const Buffer& camera,
                               VkImageView raster_depth,
                               const Buffer& projected)
    {
        const std::array buffer_infos{VkDescriptorBufferInfo{positions.buffer, 0U, VK_WHOLE_SIZE},
                                      VkDescriptorBufferInfo{camera.buffer, 0U, VK_WHOLE_SIZE},
                                      VkDescriptorBufferInfo{projected.buffer, 0U, VK_WHOLE_SIZE}};
        const VkDescriptorImageInfo image_info{VK_NULL_HANDLE, raster_depth, VK_IMAGE_LAYOUT_GENERAL};
        std::array<VkWriteDescriptorSet, 4> writes{};
        for (auto& write : writes)
            write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = set;
        writes[0].dstBinding = 0U;
        writes[0].descriptorCount = 1U;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &buffer_infos[0];
        writes[1].dstSet = set;
        writes[1].dstBinding = 1U;
        writes[1].descriptorCount = 1U;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &buffer_infos[1];
        writes[2].dstSet = set;
        writes[2].dstBinding = 2U;
        writes[2].descriptorCount = 1U;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo = &image_info;
        writes[3].dstSet = set;
        writes[3].dstBinding = 3U;
        writes[3].descriptorCount = 1U;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &buffer_infos[2];
        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0U, nullptr);
    }

    void
    update_color_set(VkDevice device, VkDescriptorSet set, const Image& color, std::span<const Buffer* const> buffers)
    {
        VkDescriptorImageInfo image_info{color.sampler, color.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet image_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        image_write.dstSet = set;
        image_write.dstBinding = 0U;
        image_write.descriptorCount = 1U;
        image_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        image_write.pImageInfo = &image_info;
        std::vector<VkDescriptorBufferInfo> infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size() + 1U);
        writes[0] = image_write;
        for (std::size_t index = 0U; index != buffers.size(); ++index)
        {
            infos[index] = {buffers[index]->buffer, 0U, VK_WHOLE_SIZE};
            writes[index + 1U] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[index + 1U].dstSet = set;
            writes[index + 1U].dstBinding = static_cast<std::uint32_t>(index + 1U);
            writes[index + 1U].descriptorCount = 1U;
            writes[index + 1U].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index + 1U].pBufferInfo = &infos[index];
        }
        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0U, nullptr);
    }

} // namespace

metmodel::RecoveredVertexColorVulkanStats
metmodel::colorize_mesh_recovered_vulkan(Mesh& mesh,
                                         std::span<const Camera> cameras,
                                         const std::filesystem::path& shader_directory,
                                         const RecoveredVertexColorVulkanOptions& options)
{
    const auto check_cancelled = [&]()
    {
        if (options.isCancelled && options.isCancelled())
            throw std::runtime_error("Recovered Vulkan colorization cancelled");
    };
    check_cancelled();
    RecoveredVertexColorVulkanStats stats;
    stats.input_vertices = mesh.vertices.size();
    stats.input_faces = mesh.faces.size();
    if (mesh.vertices.empty() || mesh.faces.empty())
        throw std::invalid_argument("Vulkan vertex colorization requires a non-empty mesh");
    if (mesh.vertices.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
        mesh.faces.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() / 3U))
    {
        throw std::overflow_error("Vulkan vertex colorization exceeds uint32 shader indexing");
    }
    if (!QDir(qt_path(shader_directory)).exists())
        throw std::invalid_argument("Vulkan vertex color shader directory does not exist");

    std::vector<const Camera*> active_cameras;
    for (const Camera& camera : cameras)
    {
        if (!camera.aligned)
            continue;
        if (camera.image.width == 0 || camera.image.height == 0 || camera.image.width > 32768 ||
            camera.image.height > 32768 || camera.image.width % 2 != 0 || camera.image.height % 2 != 0 ||
            camera.image.rgb.size() != 3U * camera.image.width * camera.image.height)
        {
            throw std::invalid_argument(
                "recovered Vulkan colorization currently requires "
                "even-sized interleaved RGB perspective frames up to 32768 pixels per dimension");
        }
        if (!active_cameras.empty() && (camera.image.width != active_cameras.front()->image.width ||
                                        camera.image.height != active_cameras.front()->image.height))
            throw std::invalid_argument("recovered colorization requires one common image resolution");
        active_cameras.push_back(&camera);
    }
    if (active_cameras.empty())
        throw std::invalid_argument("Vulkan vertex colorization has no aligned RGB cameras");
    const auto image_width = static_cast<std::uint32_t>(active_cameras.front()->image.width);
    const auto image_height = static_cast<std::uint32_t>(active_cameras.front()->image.height);
    const auto raster_width = image_width / 2U;
    const auto raster_height = image_height / 2U;

    std::vector<RecoveredVertexColorPosition> positions4(mesh.vertices.size());
    std::vector<float> positions3(mesh.vertices.size() * 3U);
    for (std::size_t index = 0U; index != mesh.vertices.size(); ++index)
    {
        const auto& position = mesh.vertices[index].position;
        positions4[index] = {
            static_cast<float>(position.x), static_cast<float>(position.y), static_cast<float>(position.z), 1.0F};
        std::copy_n(positions4[index].begin(), 3U, positions3.begin() + static_cast<std::ptrdiff_t>(3U * index));
    }
    std::vector<RecoveredVertexColorFace> faces(mesh.faces.size());
    for (std::size_t index = 0U; index != mesh.faces.size(); ++index)
    {
        for (std::size_t corner = 0U; corner != 3U; ++corner)
        {
            const std::size_t vertex = mesh.faces[index].vertices[corner];
            if (vertex >= mesh.vertices.size() || vertex > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::invalid_argument("Vulkan vertex color face index is invalid");
            }
            faces[index][corner] = static_cast<std::uint32_t>(vertex);
        }
    }
    const auto geometry2_values = compute_recovered_vertex_color_geometry_denominator(positions4, faces);

    constexpr std::array<double, 16> ordinary_frame_auxiliary{};
    constexpr std::array<double, 9> identity_image_matrix{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const auto initial_camera_source = make_recovered_vertex_color_vulkan_camera_source(
        *active_cameras.front(), ordinary_frame_auxiliary, identity_image_matrix);
    const auto initial_camera_payload = pack_recovered_vertex_color_vulkan_camera(initial_camera_source);

    const auto gpu_started = std::chrono::steady_clock::now();
    Replay replay(options);
    stats.device_name = replay.device_name();
    const VkBufferUsageFlags mesh_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    Buffer positions = replay.make_buffer(
        positions3.size() * sizeof(float), positions3.data(), positions3.size() * sizeof(float), mesh_usage);
    Buffer face_buffer = replay.make_buffer(
        faces.size() * sizeof(faces.front()), faces.data(), faces.size() * sizeof(faces.front()), mesh_usage);
    Buffer projected = replay.make_buffer(positions4.size() * 2U * sizeof(float));
    Buffer weighted = replay.make_buffer(positions4.size() * 2U * sizeof(float));
    Buffer geometry0 = replay.make_buffer(positions4.size() * sizeof(float));
    Buffer geometry1 = replay.make_buffer(positions4.size() * sizeof(float));
    Buffer geometry2 = replay.make_buffer(
        geometry2_values.size() * sizeof(float), geometry2_values.data(), geometry2_values.size() * sizeof(float));
    Buffer output = replay.make_buffer(positions4.size() * 4U * sizeof(float));
    Buffer debug = replay.make_buffer(16U);
    Buffer camera =
        replay.make_buffer(initial_camera_payload.size(), initial_camera_payload.data(), initial_camera_payload.size());
    const std::array near_far_initial{std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    Buffer near_far = replay.make_buffer(sizeof(near_far_initial), near_far_initial.data(), sizeof(near_far_initial));
    RasterTargets raster = make_raster_targets(replay, raster_width, raster_height);

    Pipeline depth_range = make_storage_pipeline(replay, find_shader(shader_directory, "shader_000_"), 3U, 8U);
    GraphicsPipeline rasterizer = make_graphics_pipeline(replay,
                                                         find_shader(shader_directory, "shader_001_"),
                                                         find_shader(shader_directory, "shader_002_"),
                                                         raster.render_pass);
    const std::array projection_descriptor_types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    Pipeline projection_pipeline =
        make_pipeline(replay, find_shader(shader_directory, "shader_003_"), projection_descriptor_types, 8U);
    Pipeline clear = make_storage_pipeline(replay, find_shader(shader_directory, "shader_004_"), 3U, 4U);
    Pipeline geometry = make_storage_pipeline(replay, find_shader(shader_directory, "shader_005_"), 6U, 16U);
    const std::array color_descriptor_types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    Pipeline color = make_pipeline(replay, find_shader(shader_directory, "shader_006_"), color_descriptor_types, 16U);

    DescriptorPool descriptors;
    descriptors.device = replay.device();
    const std::array pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64U},
                                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2U},
                                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2U}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 12U;
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    vk_check(vkCreateDescriptorPool(replay.device(), &pool_info, nullptr, &descriptors.pool),
             "vkCreateDescriptorPool(production color)");
    const VkDescriptorSet depth_range_set = allocate_set(replay.device(), descriptors.pool, depth_range.set0_layout);
    const VkDescriptorSet depth_range_debug = allocate_set(replay.device(), descriptors.pool, depth_range.debug_layout);
    const VkDescriptorSet rasterizer_set = allocate_set(replay.device(), descriptors.pool, rasterizer.set0_layout);
    const VkDescriptorSet rasterizer_debug = allocate_set(replay.device(), descriptors.pool, rasterizer.debug_layout);
    const VkDescriptorSet projection_set =
        allocate_set(replay.device(), descriptors.pool, projection_pipeline.set0_layout);
    const VkDescriptorSet projection_debug =
        allocate_set(replay.device(), descriptors.pool, projection_pipeline.debug_layout);
    const VkDescriptorSet clear_set = allocate_set(replay.device(), descriptors.pool, clear.set0_layout);
    const VkDescriptorSet clear_debug = allocate_set(replay.device(), descriptors.pool, clear.debug_layout);
    const VkDescriptorSet geometry_set = allocate_set(replay.device(), descriptors.pool, geometry.set0_layout);
    const VkDescriptorSet geometry_debug = allocate_set(replay.device(), descriptors.pool, geometry.debug_layout);
    const VkDescriptorSet color_set = allocate_set(replay.device(), descriptors.pool, color.set0_layout);
    const VkDescriptorSet color_debug = allocate_set(replay.device(), descriptors.pool, color.debug_layout);

    const std::array<const Buffer*, 3> depth_range_buffers{&positions, &camera, &near_far};
    const std::array<const Buffer*, 2> rasterizer_buffers{&camera, &near_far};
    const std::array<const Buffer*, 3> clear_buffers{&weighted, &geometry0, &geometry1};
    const std::array<const Buffer*, 6> geometry_buffers{
        &positions, &face_buffer, &projected, &weighted, &geometry0, &geometry1};
    const std::array<const Buffer*, 1> debug_buffer{&debug};
    update_storage_set(replay.device(), depth_range_set, depth_range_buffers);
    update_storage_set(replay.device(), rasterizer_set, rasterizer_buffers);
    update_projection_set(replay.device(), projection_set, positions, camera, raster.color_storage_view, projected);
    update_storage_set(replay.device(), clear_set, clear_buffers);
    update_storage_set(replay.device(), geometry_set, geometry_buffers);
    update_storage_set(replay.device(), depth_range_debug, debug_buffer);
    update_storage_set(replay.device(), rasterizer_debug, debug_buffer);
    update_storage_set(replay.device(), projection_debug, debug_buffer);
    update_storage_set(replay.device(), clear_debug, debug_buffer);
    update_storage_set(replay.device(), geometry_debug, debug_buffer);
    update_storage_set(replay.device(), color_debug, debug_buffer);

    const std::array<const Buffer*, 6> color_buffers{
        &projected, &weighted, &geometry2, &geometry0, &geometry1, &output};
    const std::uint32_t vertex_count = static_cast<std::uint32_t>(positions4.size());
    const std::uint32_t face_count = static_cast<std::uint32_t>(faces.size());
    struct TwoWordPush
    {
        std::uint32_t first;
        std::uint32_t second;
    };
    struct GeometryPush
    {
        std::array<float, 3> center;
        std::uint32_t face_count;
    };
    struct ColorPush
    {
        std::uint32_t vertex_count;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t channels;
    };
    static_assert(sizeof(TwoWordPush) == 8U);
    static_assert(sizeof(GeometryPush) == 16U);
    static_assert(sizeof(ColorPush) == 16U);

    for (const Camera* const source_camera : active_cameras)
    {
        check_cancelled();
        if (options.progress)
            options.progress(stats.cameras, active_cameras.size());
        const auto camera_source = make_recovered_vertex_color_vulkan_camera_source(
            *source_camera, ordinary_frame_auxiliary, identity_image_matrix);
        const auto camera_payload = pack_recovered_vertex_color_vulkan_camera(camera_source);
        camera.upload(camera_payload.data(), camera_payload.size());
        near_far.upload(near_far_initial.data(), sizeof(near_far_initial));
        Image color_image = make_color_image(replay, source_camera->image.rgb, image_width, image_height);
        update_color_set(replay.device(), color_set, color_image, color_buffers);

        VkCommandBuffer command = replay.make_command_buffer();
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(production geometry)");
        const TwoWordPush depth_range_push{vertex_count, 0U};
        const std::array depth_range_sets{depth_range_set, depth_range_debug};
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, depth_range.pipeline);
        vkCmdBindDescriptorSets(command,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                depth_range.layout,
                                0U,
                                static_cast<std::uint32_t>(depth_range_sets.size()),
                                depth_range_sets.data(),
                                0U,
                                nullptr);
        vkCmdPushConstants(
            command, depth_range.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(depth_range_push), &depth_range_push);
        vkCmdDispatch(command, (vertex_count + 255U) / 256U, 1U, 1U);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                             0U,
                             1U,
                             &barrier,
                             0U,
                             nullptr,
                             0U,
                             nullptr);

        std::array<VkClearValue, 2> clear_values{};
        clear_values[0].depthStencil = {1.0F, 0U};
        VkRenderPassBeginInfo render_begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        render_begin.renderPass = raster.render_pass;
        render_begin.framebuffer = raster.framebuffer;
        render_begin.renderArea.extent = {raster_width, raster_height};
        render_begin.clearValueCount = static_cast<std::uint32_t>(clear_values.size());
        render_begin.pClearValues = clear_values.data();
        vkCmdBeginRenderPass(command, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterizer.pipeline);
        const std::array rasterizer_sets{rasterizer_set, rasterizer_debug};
        vkCmdBindDescriptorSets(command,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                rasterizer.layout,
                                0U,
                                static_cast<std::uint32_t>(rasterizer_sets.size()),
                                rasterizer_sets.data(),
                                0U,
                                nullptr);
        const VkViewport viewport{
            0.0F, 0.0F, static_cast<float>(raster_width), static_cast<float>(raster_height), 0.0F, 1.0F};
        const VkRect2D scissor{{0, 0}, {raster_width, raster_height}};
        vkCmdSetViewport(command, 0U, 1U, &viewport);
        vkCmdSetScissor(command, 0U, 1U, &scissor);
        const VkDeviceSize vertex_offset = 0U;
        vkCmdBindVertexBuffers(command, 0U, 1U, &positions.buffer, &vertex_offset);
        vkCmdBindIndexBuffer(command, face_buffer.buffer, 0U, VK_INDEX_TYPE_UINT32);
        const TwoWordPush rasterizer_push{2U, 0U};
        vkCmdPushConstants(
            command, rasterizer.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0U, sizeof(rasterizer_push), &rasterizer_push);
        vkCmdDrawIndexed(command, face_count * 3U, 1U, 0U, 0, 0U);
        vkCmdEndRenderPass(command);
        VkImageMemoryBarrier raster_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        raster_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        raster_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        raster_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        raster_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        raster_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        raster_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        raster_barrier.image = raster.color;
        raster_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0U,
                             0U,
                             nullptr,
                             0U,
                             nullptr,
                             1U,
                             &raster_barrier);

        const TwoWordPush projection_push{vertex_count, 2U};
        const std::array projection_sets{projection_set, projection_debug};
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, projection_pipeline.pipeline);
        vkCmdBindDescriptorSets(command,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                projection_pipeline.layout,
                                0U,
                                static_cast<std::uint32_t>(projection_sets.size()),
                                projection_sets.data(),
                                0U,
                                nullptr);
        vkCmdPushConstants(command,
                           projection_pipeline.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0U,
                           sizeof(projection_push),
                           &projection_push);
        vkCmdDispatch(command, (vertex_count + 255U) / 256U, 1U, 1U);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0U,
                             1U,
                             &barrier,
                             0U,
                             nullptr,
                             0U,
                             nullptr);

        const std::array clear_sets{clear_set, clear_debug};
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, clear.pipeline);
        vkCmdBindDescriptorSets(command,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                clear.layout,
                                0U,
                                static_cast<std::uint32_t>(clear_sets.size()),
                                clear_sets.data(),
                                0U,
                                nullptr);
        vkCmdPushConstants(command, clear.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(vertex_count), &vertex_count);
        vkCmdDispatch(command, (vertex_count + 255U) / 256U, 1U, 1U);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0U,
                             1U,
                             &barrier,
                             0U,
                             nullptr,
                             0U,
                             nullptr);

        const GeometryPush geometry_push{{static_cast<float>(source_camera->center.x),
                                          static_cast<float>(source_camera->center.y),
                                          static_cast<float>(source_camera->center.z)},
                                         face_count};
        const std::array geometry_sets{geometry_set, geometry_debug};
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, geometry.pipeline);
        vkCmdBindDescriptorSets(command,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                geometry.layout,
                                0U,
                                static_cast<std::uint32_t>(geometry_sets.size()),
                                geometry_sets.data(),
                                0U,
                                nullptr);
        vkCmdPushConstants(
            command, geometry.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(geometry_push), &geometry_push);
        vkCmdDispatch(command, (face_count + 255U) / 256U, 1U, 1U);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0U,
                             1U,
                             &barrier,
                             0U,
                             nullptr,
                             0U,
                             nullptr);
        vk_check(vkEndCommandBuffer(command), "vkEndCommandBuffer(production geometry)");
        replay.submit(command);

        command = replay.make_command_buffer();
        begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(production color)");
        const ColorPush color_push{vertex_count, image_width, image_height, 3U};
        const std::array color_sets{color_set, color_debug};
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, color.pipeline);
        vkCmdBindDescriptorSets(command,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                color.layout,
                                0U,
                                static_cast<std::uint32_t>(color_sets.size()),
                                color_sets.data(),
                                0U,
                                nullptr);
        vkCmdPushConstants(command, color.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(color_push), &color_push);
        vkCmdDispatch(command, (vertex_count + 255U) / 256U, 1U, 1U);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0U,
                             1U,
                             &barrier,
                             0U,
                             nullptr,
                             0U,
                             nullptr);
        vk_check(vkEndCommandBuffer(command), "vkEndCommandBuffer(production color)");
        replay.submit(command);
        ++stats.cameras;
        ++stats.draw_calls;
        stats.compute_dispatches += 5U;
    }

    const auto output_values = output.download<float>(positions4.size() * 4U);
    stats.gpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_started).count();
    std::vector<RecoveredVertexColorAccumulator> accumulator(positions4.size());
    for (std::size_t vertex = 0U; vertex != accumulator.size(); ++vertex)
    {
        std::copy_n(output_values.begin() + static_cast<std::ptrdiff_t>(4U * vertex), 4U, accumulator[vertex].begin());
        stats.directly_colored_vertices += static_cast<std::size_t>(accumulator[vertex][3] > 0.01F);
    }

    const auto finalize_started = std::chrono::steady_clock::now();
    check_cancelled();
    const auto distance = extrapolate_recovered_vertex_color_accumulator(mesh, accumulator);
    for (std::size_t vertex = 0U; vertex != accumulator.size(); ++vertex)
    {
        if (distance[vertex] > 0)
            ++stats.extrapolated_vertices;
        if (distance[vertex] < 0)
            ++stats.uncolored_vertices;
    }
    assign_recovered_vertex_colors_from_normalized_accumulator(mesh, accumulator);
    stats.finalize_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - finalize_started).count();
    if (options.progress)
        options.progress(stats.cameras, active_cameras.size());
    return stats;
}
