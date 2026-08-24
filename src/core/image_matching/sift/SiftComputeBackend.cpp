#include "SiftComputeBackend.h"

#include <stdexcept>

namespace xjw::image_matching
{

#if defined(PLASCAN_HAS_CUDA_SIFT)
    bool isCudaSiftBackendAvailable(int deviceIndex);
    QString cudaSiftDeviceName(int deviceIndex);
    SiftRawFeatures extractCudaSift(const SiftExtractionRequest& request);
    std::vector<SiftNearestMatch>
    matchCudaSift(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, int deviceIndex);
    SiftBidirectionalMatches
    matchCudaSiftBidirectionally(const cv::Mat& descriptors0, const cv::Mat& descriptors1, int deviceIndex);
    void releaseCudaSiftThreadWorkspaces();
#endif

#if defined(PLASCAN_HAS_OPENCL_SIFT)
    bool isOpenClSiftBackendAvailable(int deviceIndex);
    QString openClSiftDeviceName(int deviceIndex);
    SiftRawFeatures extractOpenClSift(const SiftExtractionRequest& request);
    std::vector<SiftNearestMatch>
    matchOpenClSift(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, int deviceIndex);
#endif

#if defined(PLASCAN_HAS_METAL_SIFT)
    bool isMetalSiftBackendAvailable(int deviceIndex);
    QString metalSiftDeviceName(int deviceIndex);
    SiftRawFeatures extractMetalSift(const SiftExtractionRequest& request);
    std::vector<SiftNearestMatch>
    matchMetalSift(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, int deviceIndex);
#endif

    const char* siftBackendName(SiftComputeBackend backend)
    {
        switch (backend)
        {
        case SiftComputeBackend::Automatic:
            return "auto";
        case SiftComputeBackend::Cpu:
            return "cpu";
        case SiftComputeBackend::Cuda:
            return "cuda";
        case SiftComputeBackend::OpenCl:
            return "opencl";
        case SiftComputeBackend::Metal:
            return "metal";
        }
        return "unknown";
    }

    QString siftBackendDisplayName(SiftComputeBackend backend)
    {
        switch (backend)
        {
        case SiftComputeBackend::Automatic:
            return QStringLiteral("自动");
        case SiftComputeBackend::Cpu:
            return QStringLiteral("OpenCV CPU");
        case SiftComputeBackend::Cuda:
            return QStringLiteral("CUDA");
        case SiftComputeBackend::OpenCl:
            return QStringLiteral("OpenCL");
        case SiftComputeBackend::Metal:
            return QStringLiteral("Metal");
        }
        return QStringLiteral("未知");
    }

    QString siftBackendDeviceName(SiftComputeBackend backend, int deviceIndex)
    {
        switch (backend)
        {
        case SiftComputeBackend::Cuda:
#if defined(PLASCAN_HAS_CUDA_SIFT)
            return cudaSiftDeviceName(deviceIndex);
#else
            break;
#endif
        case SiftComputeBackend::OpenCl:
#if defined(PLASCAN_HAS_OPENCL_SIFT)
            return openClSiftDeviceName(deviceIndex);
#else
            break;
#endif
        case SiftComputeBackend::Metal:
#if defined(PLASCAN_HAS_METAL_SIFT)
            return metalSiftDeviceName(deviceIndex);
#else
            break;
#endif
        case SiftComputeBackend::Automatic:
        case SiftComputeBackend::Cpu:
            break;
        }
        return QString();
    }

    QString siftBackendRuntimeDisplayName(SiftComputeBackend backend, int deviceIndex)
    {
        const QString backendName = siftBackendDisplayName(backend);
        const QString deviceName = siftBackendDeviceName(backend, deviceIndex).trimmed();
        return deviceName.isEmpty() ? backendName : QStringLiteral("%1 · %2").arg(backendName, deviceName);
    }

    bool isSiftBackendAvailable(SiftComputeBackend backend, int deviceIndex)
    {
        switch (backend)
        {
        case SiftComputeBackend::Automatic:
        case SiftComputeBackend::Cpu:
            return true;
        case SiftComputeBackend::Cuda:
#if defined(PLASCAN_HAS_CUDA_SIFT)
            return isCudaSiftBackendAvailable(deviceIndex);
#else
            return false;
#endif
        case SiftComputeBackend::OpenCl:
#if defined(PLASCAN_HAS_OPENCL_SIFT)
            return isOpenClSiftBackendAvailable(deviceIndex);
#else
            return false;
#endif
        case SiftComputeBackend::Metal:
#if defined(PLASCAN_HAS_METAL_SIFT)
            return isMetalSiftBackendAvailable(deviceIndex);
#else
            return false;
#endif
        }
        return false;
    }

    SiftComputeBackend resolveSiftBackend(SiftComputeBackend requested, int deviceIndex)
    {
        if (requested != SiftComputeBackend::Automatic)
        {
            if (!isSiftBackendAvailable(requested, deviceIndex))
            {
                throw std::runtime_error(std::string("Requested SIFT backend is unavailable: ") +
                                         siftBackendName(requested));
            }
            return requested;
        }

        constexpr SiftComputeBackend priority[] = {
            SiftComputeBackend::Cuda, SiftComputeBackend::Metal, SiftComputeBackend::OpenCl, SiftComputeBackend::Cpu};
        for (const SiftComputeBackend candidate : priority)
        {
            if (isSiftBackendAvailable(candidate, deviceIndex))
            {
                return candidate;
            }
        }
        return SiftComputeBackend::Cpu;
    }

    SiftRawFeatures extractSiftOnGpu(SiftComputeBackend backend, const SiftExtractionRequest& request)
    {
        switch (backend)
        {
        case SiftComputeBackend::Cuda:
#if defined(PLASCAN_HAS_CUDA_SIFT)
            return extractCudaSift(request);
#endif
            break;
        case SiftComputeBackend::OpenCl:
#if defined(PLASCAN_HAS_OPENCL_SIFT)
            return extractOpenClSift(request);
#endif
            break;
        case SiftComputeBackend::Metal:
#if defined(PLASCAN_HAS_METAL_SIFT)
            return extractMetalSift(request);
#endif
            break;
        case SiftComputeBackend::Automatic:
        case SiftComputeBackend::Cpu:
            break;
        }
        throw std::runtime_error(std::string("SIFT GPU extraction backend is unavailable: ") +
                                 siftBackendName(backend));
    }

    std::vector<SiftNearestMatch> matchSiftOnGpu(SiftComputeBackend backend,
                                                 const cv::Mat& queryDescriptors,
                                                 const cv::Mat& trainDescriptors,
                                                 int deviceIndex)
    {
        switch (backend)
        {
        case SiftComputeBackend::Cuda:
#if defined(PLASCAN_HAS_CUDA_SIFT)
            return matchCudaSift(queryDescriptors, trainDescriptors, deviceIndex);
#endif
            break;
        case SiftComputeBackend::OpenCl:
#if defined(PLASCAN_HAS_OPENCL_SIFT)
            return matchOpenClSift(queryDescriptors, trainDescriptors, deviceIndex);
#endif
            break;
        case SiftComputeBackend::Metal:
#if defined(PLASCAN_HAS_METAL_SIFT)
            return matchMetalSift(queryDescriptors, trainDescriptors, deviceIndex);
#endif
            break;
        case SiftComputeBackend::Automatic:
        case SiftComputeBackend::Cpu:
            break;
        }
        throw std::runtime_error(std::string("SIFT GPU matching backend is unavailable: ") + siftBackendName(backend));
    }

    SiftBidirectionalMatches matchSiftBidirectionallyOnGpu(SiftComputeBackend backend,
                                                           const cv::Mat& descriptors0,
                                                           const cv::Mat& descriptors1,
                                                           int deviceIndex)
    {
        if (backend == SiftComputeBackend::Cuda)
        {
#if defined(PLASCAN_HAS_CUDA_SIFT)
            return matchCudaSiftBidirectionally(descriptors0, descriptors1, deviceIndex);
#endif
        }

        SiftBidirectionalMatches result;
        result.forward = matchSiftOnGpu(backend, descriptors0, descriptors1, deviceIndex);
        result.reverse = matchSiftOnGpu(backend, descriptors1, descriptors0, deviceIndex);
        return result;
    }

    void releaseSiftGpuThreadWorkspaces()
    {
#if defined(PLASCAN_HAS_CUDA_SIFT)
        releaseCudaSiftThreadWorkspaces();
#endif
    }

} // namespace xjw::image_matching
