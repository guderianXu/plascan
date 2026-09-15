#include "DepthTsdfInternals.h"
namespace xjw::mesh::tsdf_detail
{
    using namespace tsdf_detail;

    bool checkedMultiply(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t* result)
    {
        if (!result || (rhs > 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs))
        {
            return false;
        }
        *result = lhs * rhs;
        return true;
    }

    std::uint64_t availablePhysicalMemoryBytes()
    {
#ifdef _WIN32
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status))
        {
            return static_cast<std::uint64_t>(status.ullAvailPhys);
        }
#elif defined(__APPLE__)
        vm_statistics64_data_t statistics{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        vm_size_t page_size = 0;
        const mach_port_t host = mach_host_self();
        const kern_return_t page_result = host_page_size(host, &page_size);
        const kern_return_t statistics_result =
            host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&statistics), &count);
        mach_port_deallocate(mach_task_self(), host);
        if (page_result == KERN_SUCCESS && statistics_result == KERN_SUCCESS)
        {
            const std::uint64_t available_pages = static_cast<std::uint64_t>(statistics.free_count) +
                                                  static_cast<std::uint64_t>(statistics.inactive_count) +
                                                  static_cast<std::uint64_t>(statistics.speculative_count) +
                                                  static_cast<std::uint64_t>(statistics.purgeable_count);
            return available_pages * static_cast<std::uint64_t>(page_size);
        }
#elif defined(__linux__)
        struct sysinfo status{};
        if (sysinfo(&status) == 0)
        {
            return static_cast<std::uint64_t>(status.freeram) * static_cast<std::uint64_t>(status.mem_unit);
        }
#endif
        return 0;
    }
} // namespace xjw::mesh::tsdf_detail
