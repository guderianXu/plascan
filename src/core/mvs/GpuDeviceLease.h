#pragma once

#include <QLockFile>

#include <memory>
#include <string>
#include <vector>

class QString;

namespace xjw
{
namespace mvs
{

struct GpuDeviceDescriptor
{
    std::string physicalIdentity;
    std::string displayName;
};

class GpuDeviceLeaseSet
{
public:
    GpuDeviceLeaseSet() = default;
    GpuDeviceLeaseSet(const GpuDeviceLeaseSet &) = delete;
    GpuDeviceLeaseSet &operator=(const GpuDeviceLeaseSet &) = delete;
    ~GpuDeviceLeaseSet();

    bool acquire(const std::vector<GpuDeviceDescriptor> &devices, QString *errorMessage);
    bool empty() const noexcept;

private:
    std::vector<std::unique_ptr<QLockFile>> _locks;
};

std::string fallbackGpuPhysicalIdentity(const std::string &vendor,
                                        const std::string &name,
                                        int deviceIndex);

} // namespace mvs
} // namespace xjw
