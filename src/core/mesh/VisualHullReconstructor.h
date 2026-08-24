#pragma once

#include "FramePinholeCamera.h"
#include "MeshTypes.h"

#include <opencv2/core.hpp>

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

    enum class VisualHullComputeBackend
    {
        // Auto may fall back in CUDA -> OpenCL -> CPU order. Explicit device
        // selections are strict and report an error when unavailable.
        Auto,
        Cpu,
        Cuda,
        OpenCL
    };

    struct VisualHullExecutionInfo
    {
        // Auto denotes "not resolved" in actualBackend when execution fails
        // before a backend completes.
        VisualHullComputeBackend requestedBackend = VisualHullComputeBackend::Auto;
        VisualHullComputeBackend actualBackend = VisualHullComputeBackend::Auto;
        int requestedDeviceIndex = -1;
        int actualDeviceIndex = -1;
        bool usedFallback = false;
        std::string fallbackReason;
    };

    struct VisualHullView
    {
        xjw::FramePinholeCamera camera;
        cv::Mat silhouetteMask;
        cv::Mat depthMap;
        cv::Mat colorImage;
        std::string imagePath;
    };

    struct VisualHullConfig
    {
        std::array<float, 3> boundsMin{-1.0f, -1.0f, -1.0f};
        std::array<float, 3> boundsMax{1.0f, 1.0f, 1.0f};
        int resolution = 96;
        int minimumVisibleViews = 4;
        int allowedSilhouetteViolations = 1;
        bool enableDepthFreeSpaceCarving = false;
        int minimumDepthFreeSpaceViolations = 2;
        float relativeDepthTolerance = 0.01f;
        bool closeVolumeBoundary = true;
        int topologyClosingIterations = 0;
        bool useContinuousSilhouetteField = false;
        int smoothingIterations = 2;
        float smoothingLambda = 0.18f;
        int workerCount = 0;
        VisualHullComputeBackend computeBackend = VisualHullComputeBackend::Auto;
        int computeDeviceIndex = -1;
        // Number of sampled Z layers per GPU dispatch. Smaller slabs improve
        // cancellation latency and reduce the chance of a display-driver TDR.
        int gpuSlabDepth = 16;
        std::function<bool()> isCancelled;
        std::function<void(const std::string&, float)> progressFn;
        std::function<void(const VisualHullExecutionInfo&)> executionInfoFn;
    };

    struct MeshConnectivityStats
    {
        struct Component
        {
            std::size_t faceCount = 0;
            std::array<float, 3> boundsMin{};
            std::array<float, 3> boundsMax{};
            double diagonal = 0.0;
        };

        int componentCount = 0;
        std::size_t largestComponentFaceCount = 0;
        double largestComponentFaceRatio = 0.0;
        std::vector<std::size_t> componentFaceCounts;
        std::vector<Component> components;
    };

    class VisualHullReconstructor
    {
    public:
        static bool reconstruct(const std::vector<VisualHullView>& views,
                                const VisualHullConfig& config,
                                TriMesh* mesh,
                                std::string* errorMessage = nullptr,
                                VisualHullExecutionInfo* executionInfo = nullptr);
        static MeshConnectivityStats analyzeConnectivity(const TriMesh& mesh);
        static bool requiresSilhouetteOnlyRetry(const MeshConnectivityStats& stats,
                                                double minimumLargestComponentFaceRatio,
                                                int maximumConnectedComponents);
        static bool retainLargestConnectedComponent(TriMesh* mesh);
    };

} // namespace xjw::mesh
