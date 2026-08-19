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

/// Normalizes a device display name for conservative cross-API matching when
/// one runtime cannot expose a PCI identity.
std::string normalizedGpuDeviceName(const std::string &name);

/// Returns true when an OpenCL vendor string identifies NVIDIA. Auto mode uses
/// this to avoid admitting an unstable duplicate of an active CUDA device;
/// explicit OpenCL-only processing remains allowed.
bool isNvidiaOpenClVendor(const std::string &openClVendor);

/// Returns true when an OpenCL interface cannot expose a PCI identity but is
/// supplied by NVIDIA while CUDA is already selected. In that case treating
/// it as a separate physical GPU could create two execution lanes and two
/// process leases for the same device, so Auto scheduling must skip it.
bool shouldSkipUnstableOpenClCudaAlias(const std::string &openClVendor,
                                       const std::string &physicalIdentity,
                                       bool cudaSelected);

} // namespace mvs
} // namespace xjw
