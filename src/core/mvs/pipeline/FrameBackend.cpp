#include "MvsPipelineInternals.h"

namespace xjw::mvs::pipeline_detail
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    bool estimatePatchMatchWithAdaptiveCuda(const char* stageLabel,
                                            int refIdx,
                                            const cv::Mat& refGray,
                                            const std::vector<cv::Mat>& srcGrays,
                                            const FramePinholeCamera& refCam,
                                            const std::vector<FramePinholeCamera>& srcCams,
                                            float zNear,
                                            float zFar,
                                            const PatchMatchConfig& config,
                                            cv::Mat& depthOut,
                                            cv::Mat* confOut,
                                            std::string* errorMsg,
                                            const cv::Mat* hintDepth,
                                            const cv::Mat* hintRadius,
                                            const cv::Mat* referenceValidMask,
                                            const std::vector<cv::Mat>* sourceValidMasks,
                                            const PatchMatchAuxiliaryInput* auxiliaryInput,
                                            PatchMatchAuxiliaryOutput* auxiliaryOutput)
    {
        const bool tryCuda = (config.backend == PatchMatchBackend::Cuda || config.backend == PatchMatchBackend::Auto) &&
                             PatchMatchDepthEstimator::isCudaAvailable();
        if (!tryCuda)
        {
            return PatchMatchDepthEstimator::estimate(refGray,
                                                      srcGrays,
                                                      refCam,
                                                      srcCams,
                                                      zNear,
                                                      zFar,
                                                      config,
                                                      depthOut,
                                                      confOut,
                                                      errorMsg,
                                                      hintDepth,
                                                      hintRadius,
                                                      referenceValidMask,
                                                      sourceValidMasks,
                                                      auxiliaryInput,
                                                      auxiliaryOutput);
        }

        constexpr int kMaxCudaAttempts = 4;
        PatchMatchConfig attemptConfig = config;
        attemptConfig.cudaFallbackToCpu = false;
        std::string lastCudaError;

        for (int attempt = 0; attempt < kMaxCudaAttempts; ++attempt)
        {
            std::string attemptError;
            if (PatchMatchDepthEstimator::estimate(refGray,
                                                   srcGrays,
                                                   refCam,
                                                   srcCams,
                                                   zNear,
                                                   zFar,
                                                   attemptConfig,
                                                   depthOut,
                                                   confOut,
                                                   &attemptError,
                                                   hintDepth,
                                                   hintRadius,
                                                   referenceValidMask,
                                                   sourceValidMasks,
                                                   auxiliaryInput,
                                                   auxiliaryOutput))
            {
                if (attemptConfig.downsampleFactor != config.downsampleFactor)
                {
                    LOG_INFO("[MVS][帧 %d][PatchMatch] %s CUDA 重试成功: ds=%d iterations=%d patch=%d",
                             refIdx,
                             stageLabel,
                             attemptConfig.downsampleFactor,
                             attemptConfig.numIterations,
                             attemptConfig.patchHalf * 2 + 1);
                }
                if (errorMsg)
                {
                    errorMsg->clear();
                }
                return true;
            }

            lastCudaError = attemptError;
            if (config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed))
            {
                if (errorMsg)
                {
                    *errorMsg = attemptError.empty() ? std::string("PatchMatch cancelled") : attemptError;
                }
                return false;
            }

            PatchMatchDepthEstimator::cleanupGpuImageCache();

            if (!isCudaMemoryFailure(attemptError) || attemptConfig.downsampleFactor >= 12)
            {
                break;
            }

            PatchMatchConfig nextConfig =
                MvsPipelineService::nextCudaRetryPatchMatchConfig(attemptConfig, refGray.cols, refGray.rows);
            if (nextConfig.downsampleFactor <= attemptConfig.downsampleFactor)
            {
                break;
            }

            LOG_WARN("[MVS][帧 %d][PatchMatch] %s CUDA 显存不足，ds=%d -> %d 后重试: %s",
                     refIdx,
                     stageLabel,
                     attemptConfig.downsampleFactor,
                     nextConfig.downsampleFactor,
                     attemptError.c_str());

            attemptConfig = nextConfig;
            attemptConfig.cudaFallbackToCpu = false;
        }

        if (!config.cudaFallbackToCpu)
        {
            LOG_ERROR("[MVS][帧 %d][PatchMatch] %s CUDA 重试失败，配置禁止回退 CPU: %s",
                      refIdx,
                      stageLabel,
                      lastCudaError.empty() ? "未知 CUDA 错误" : lastCudaError.c_str());
            if (errorMsg)
            {
                *errorMsg = lastCudaError.empty() ? std::string("CUDA PatchMatch retries exhausted") : lastCudaError;
            }
            return false;
        }

        LOG_WARN("[MVS][帧 %d][PatchMatch] %s CUDA 重试失败，按配置回退 CPU: %s",
                 refIdx,
                 stageLabel,
                 lastCudaError.empty() ? "未知 CUDA 错误" : lastCudaError.c_str());

        PatchMatchConfig cpuConfig = config;
        cpuConfig.backend = PatchMatchBackend::Cpu;
        cpuConfig.cudaFallbackToCpu = false;
        const bool cpuOk = PatchMatchDepthEstimator::estimate(refGray,
                                                              srcGrays,
                                                              refCam,
                                                              srcCams,
                                                              zNear,
                                                              zFar,
                                                              cpuConfig,
                                                              depthOut,
                                                              confOut,
                                                              errorMsg,
                                                              hintDepth,
                                                              hintRadius,
                                                              referenceValidMask,
                                                              sourceValidMasks,
                                                              auxiliaryInput,
                                                              auxiliaryOutput);
        if (!cpuOk && errorMsg && errorMsg->empty())
        {
            *errorMsg = lastCudaError;
        }
        return cpuOk;
    }
} // namespace xjw::mvs::pipeline_detail
