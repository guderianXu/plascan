#include "DenseMatchService.h"
#include "BlockMatcher.h"
#include "DenseMatchBackend.h"
#include "SgmMatcher.h"
#include "opencv/OpenCVSgbmWrapper.h"
#include "SubpixelRefiner.h"
#include "DisparityValidator.h"
#include "io/PathIO.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xjw::dense_match
{

    namespace
    {

        constexpr int kMaximumDenseMatchKernelSize = 255;

        class DenseMatchExecutionError : public std::runtime_error
        {
        public:
            using std::runtime_error::runtime_error;
        };

        std::string formatDenseMatchFailure(const cv::Size imageSize,
                                            const DenseMatchConfig& config,
                                            const char* category,
                                            const char* detail)
        {
            std::ostringstream message;
            message << category << ": size=" << imageSize.width << 'x' << imageSize.height << " disparity=["
                    << config.minDisparity << ',' << config.maxDisparity << ')';
            if (detail != nullptr && detail[0] != '\0')
            {
                message << ": " << detail;
            }
            return message.str();
        }

        void setReverseDisparityRange(const DenseMatchConfig& forwardConfig, DenseMatchConfig& reverseConfig)
        {
            const std::int64_t reverseMinimum = 1 - static_cast<std::int64_t>(forwardConfig.maxDisparity);
            const std::int64_t reverseMaximum = 1 - static_cast<std::int64_t>(forwardConfig.minDisparity);
            if (reverseMinimum < std::numeric_limits<int>::min() || reverseMinimum > std::numeric_limits<int>::max() ||
                reverseMaximum < std::numeric_limits<int>::min() || reverseMaximum > std::numeric_limits<int>::max())
            {
                throw std::overflow_error("Dense matching reverse disparity range cannot be represented by int");
            }
            reverseConfig.minDisparity = static_cast<int>(reverseMinimum);
            reverseConfig.maxDisparity = static_cast<int>(reverseMaximum);
        }

        std::string validateDenseMatchKernels(const DenseMatchConfig& config)
        {
            if (config.corrKernelW <= 0 || config.corrKernelH <= 0 ||
                config.corrKernelW > kMaximumDenseMatchKernelSize ||
                config.corrKernelH > kMaximumDenseMatchKernelSize || config.corrKernelW % 2 == 0 ||
                config.corrKernelH % 2 == 0)
            {
                return "correlation kernels must be positive odd values no larger than 255";
            }
            if (config.medianFilterSize < 0 || config.medianFilterSize > kMaximumDenseMatchKernelSize ||
                (config.medianFilterSize > 0 && config.medianFilterSize % 2 == 0))
            {
                return "median filter size must be zero or a positive odd value no larger than 255";
            }
            return {};
        }

        void resetExecutionReport(DenseMatchExecutionReport& report, const DenseMatchConfig& config)
        {
            report = {};
            report.requestedBackend = config.computeBackend;
            report.actualBackend = config.computeBackend == DenseMatchComputeBackend::Automatic
                                       ? DenseMatchComputeBackend::Cpu
                                       : config.computeBackend;
            report.deviceIndex = config.computeBackend == DenseMatchComputeBackend::Cuda     ? config.cudaDevice
                                 : config.computeBackend == DenseMatchComputeBackend::OpenCl ? config.openClDevice
                                                                                             : -1;
        }

        int backendDeviceIndex(DenseMatchComputeBackend backend, const DenseMatchConfig& config)
        {
            if (backend == DenseMatchComputeBackend::Cuda)
            {
                return config.cudaDevice;
            }
            if (backend == DenseMatchComputeBackend::OpenCl)
            {
                return config.openClDevice;
            }
            return -1;
        }

        std::string backendAttemptLabel(DenseMatchComputeBackend backend, const DenseMatchConfig& config)
        {
            std::string label = denseMatchComputeBackendName(backend);
            const int device_index = backendDeviceIndex(backend, config);
            if (device_index >= 0)
            {
                label += " device " + std::to_string(device_index);
            }
            return label;
        }

        void appendFallbackReason(std::string& reasons, std::string reason)
        {
            if (!reasons.empty())
            {
                reasons += "; ";
            }
            reasons += std::move(reason);
        }

    } // namespace

    DenseMatchService::DenseMatchService(const DenseMatchConfig& cfg) : _config(cfg)
    {
    }

    const std::string& DenseMatchService::lastError() const
    {
        return _lastError;
    }

    const DenseMatchExecutionReport& DenseMatchService::executionReport() const
    {
        return _executionReport;
    }

    DisparityResult DenseMatchService::fail(std::string message)
    {
        _lastError = std::move(message);
        fprintf(stderr, "[DenseMatch] ERROR: %s\n", _lastError.c_str());
        return {};
    }

    DisparityResult DenseMatchService::process()
    {
        _lastError.clear();
        resetExecutionReport(_executionReport, _config);
        _left.release();
        _right.release();
        try
        {
            _left = xjw::common::io::readImage(_config.leftImagePath, cv::IMREAD_GRAYSCALE);
            _right = xjw::common::io::readImage(_config.rightImagePath, cv::IMREAD_GRAYSCALE);
        }
        catch (const std::bad_alloc& error)
        {
            return fail(
                formatDenseMatchFailure(_left.size(), _config, "Dense matching host allocation failed", error.what()));
        }
        catch (const cv::Exception& error)
        {
            return fail(
                formatDenseMatchFailure(_left.size(), _config, "Dense matching image load failed", error.what()));
        }
        catch (const std::exception& error)
        {
            return fail(
                formatDenseMatchFailure(_left.size(), _config, "Dense matching image load failed", error.what()));
        }
        if (_left.empty() || _right.empty())
        {
            return fail(formatDenseMatchFailure(
                _left.size(), _config, "Dense matching image load failed", "one or both input images are empty"));
        }
        return process(_left, _right);
    }

    DisparityResult DenseMatchService::process(const cv::Mat& left, const cv::Mat& right)
    {
        _lastError.clear();
        resetExecutionReport(_executionReport, _config);
        try
        {
            return processImpl(left, right);
        }
        catch (const DenseMatchExecutionError& error)
        {
            return fail(error.what());
        }
        catch (const std::bad_alloc& error)
        {
            return fail(
                formatDenseMatchFailure(left.size(), _config, "Dense matching host allocation failed", error.what()));
        }
        catch (const cv::Exception& error)
        {
            const char* category = error.code == cv::Error::GpuApiCallError
                                       ? "Dense matching CUDA execution or allocation failed"
                                   : error.code == cv::Error::StsNoMem ? "Dense matching OpenCV allocation failed"
                                                                       : "Dense matching OpenCV operation failed";
            return fail(formatDenseMatchFailure(left.size(), _config, category, error.what()));
        }
        catch (const std::length_error& error)
        {
            return fail(formatDenseMatchFailure(
                left.size(), _config, "Dense matching layout is not representable", error.what()));
        }
        catch (const std::overflow_error& error)
        {
            return fail(formatDenseMatchFailure(left.size(), _config, "Dense matching layout overflow", error.what()));
        }
        catch (const std::exception& error)
        {
            return fail(formatDenseMatchFailure(left.size(), _config, "Dense matching execution failed", error.what()));
        }
    }

    DisparityResult DenseMatchService::processImpl(const cv::Mat& left, const cv::Mat& right)
    {
        if (left.empty() || right.empty() || left.size() != right.size() || left.type() != CV_8UC1 ||
            right.type() != CV_8UC1 || _config.maxDisparity <= _config.minDisparity)
        {
            return fail(
                formatDenseMatchFailure(left.size(), _config, "Dense matching input validation failed", nullptr));
        }
        const std::string kernelValidationError = validateDenseMatchKernels(_config);
        if (!kernelValidationError.empty())
        {
            return fail(formatDenseMatchFailure(
                left.size(), _config, "Dense matching configuration is invalid", kernelValidationError.c_str()));
        }
        const std::int64_t disparityCount =
            static_cast<std::int64_t>(_config.maxDisparity) - static_cast<std::int64_t>(_config.minDisparity);
        if (_config.algorithm == StereoAlgorithm::OpenCV_SGBM &&
            (disparityCount > std::numeric_limits<int>::max() || disparityCount % 16 != 0))
        {
            return fail(formatDenseMatchFailure(
                left.size(),
                _config,
                "Dense matching configuration is invalid",
                "OpenCV SGBM requires an int-representable positive multiple-of-16 disparity count"));
        }

        std::vector<DenseMatchComputeBackend> backend_candidates;
        std::string fallback_reason;
        if (_config.algorithm == StereoAlgorithm::OpenCV_SGBM)
        {
            if (_config.computeBackend == DenseMatchComputeBackend::Cuda ||
                _config.computeBackend == DenseMatchComputeBackend::OpenCl)
            {
                return fail(formatDenseMatchFailure(
                    left.size(),
                    _config,
                    "Dense matching configuration is invalid",
                    "OpenCV SGBM is CPU-only and cannot use an explicitly requested GPU backend"));
            }
            backend_candidates.push_back(DenseMatchComputeBackend::Cpu);
        }
        else if (_config.computeBackend != DenseMatchComputeBackend::Automatic)
        {
            backend_candidates.push_back(resolveDenseMatchComputeBackend(_config));
        }
        else if (!_config.useCuda)
        {
            backend_candidates.push_back(DenseMatchComputeBackend::Cpu);
        }
        else
        {
            if (isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend::Cuda, _config.cudaDevice))
            {
                backend_candidates.push_back(DenseMatchComputeBackend::Cuda);
            }
            else
            {
                appendFallbackReason(fallback_reason,
                                     backendAttemptLabel(DenseMatchComputeBackend::Cuda, _config) + " unavailable");
            }

            if (isDenseMatchComputeBackendAvailable(DenseMatchComputeBackend::OpenCl, _config.openClDevice))
            {
                backend_candidates.push_back(DenseMatchComputeBackend::OpenCl);
            }
            else
            {
                appendFallbackReason(fallback_reason,
                                     backendAttemptLabel(DenseMatchComputeBackend::OpenCl, _config) + " unavailable");
            }
            backend_candidates.push_back(DenseMatchComputeBackend::Cpu);
        }

        auto t0 = std::chrono::steady_clock::now();
        const bool needs_reverse = _config.enableLRCheck && _config.lrCheckThreshold > 0.0f;
        DenseMatchConfig reverse_config = _config;
        if (needs_reverse)
        {
            setReverseDisparityRange(_config, reverse_config);
        }

        DisparityResult result;
        DisparityResult reverse;
        std::string backend_error;
        const bool backend_succeeded = detail::runDenseMatchBackendAttempts(
            _config.computeBackend,
            backend_candidates,
            std::move(fallback_reason),
            [&](const DenseMatchComputeBackend backend, std::string* error_message)
            {
                DenseMatchConfig attempt_config = _config;
                attempt_config.computeBackend = backend;
                attempt_config.useCuda = backend != DenseMatchComputeBackend::Cpu;
                reverse_config.computeBackend = backend;
                reverse_config.useCuda = attempt_config.useCuda;

                _executionReport.deviceIndex = backendDeviceIndex(backend, _config);

                fprintf(stdout,
                        "[DenseMatch] size=%dx%d algo=%d cost=%d disp=[%d,%d) kernel=%dx%d "
                        "backend=%s cudaDevice=%d openClDevice=%d threads=%d\n",
                        left.cols,
                        left.rows,
                        static_cast<int>(_config.algorithm),
                        static_cast<int>(_config.costFunc),
                        _config.minDisparity,
                        _config.maxDisparity,
                        _config.corrKernelW,
                        _config.corrKernelH,
                        denseMatchComputeBackendName(backend),
                        _config.cudaDevice,
                        _config.openClDevice,
                        _config.numThreads);

                try
                {
                    result = computeRawDisparity(left, right, attempt_config);
                    if (result.disparity.empty())
                    {
                        throw DenseMatchExecutionError(formatDenseMatchFailure(left.size(),
                                                                               attempt_config,
                                                                               "Dense matching execution failed",
                                                                               "backend returned no result"));
                    }
                    if (needs_reverse)
                    {
                        reverse = computeRawDisparity(right, left, reverse_config);
                        if (reverse.disparity.empty())
                        {
                            throw DenseMatchExecutionError(
                                formatDenseMatchFailure(right.size(),
                                                        reverse_config,
                                                        "Dense matching execution failed",
                                                        "reverse backend returned no result"));
                        }
                    }
                    return true;
                }
                catch (const std::exception& error)
                {
                    *error_message = backendAttemptLabel(backend, _config) + " failed: " + error.what();
                    if (_config.computeBackend == DenseMatchComputeBackend::Automatic &&
                        backend != DenseMatchComputeBackend::Cpu)
                    {
                        fprintf(stderr, "[DenseMatch] WARNING: %s; trying next backend\n", error_message->c_str());
                    }
                    return false;
                }
            },
            &_executionReport,
            &backend_error);
        if (!backend_succeeded)
        {
            throw DenseMatchExecutionError(backend_error);
        }

        auto t1 = std::chrono::steady_clock::now();
        fprintf(stdout, "[DenseMatch] matching: %.0f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

        DisparityValidator validator(_config);
        result = validator.validate(result.disparity, result.confidence, result.validMask);
        if (needs_reverse && !result.disparity.empty())
        {
            reverse = validator.validate(reverse.disparity, reverse.confidence, reverse.validMask);
            if (!reverse.disparity.empty() && reverse.disparity.size() == result.disparity.size())
            {
                const cv::Mat lrMask = validator.checkLRConsistency(
                    result.disparity, reverse.disparity, result.validMask, reverse.validMask);
                if (result.validMask.empty())
                {
                    result.validMask = lrMask;
                }
                else
                {
                    cv::bitwise_and(result.validMask, lrMask, result.validMask);
                }
            }
        }
        validator.applyImageSupportMask(result, left, right);

        auto t2 = std::chrono::steady_clock::now();
        fprintf(stdout,
                "[DenseMatch] validate: %.0f ms, total: %.0f ms\n",
                std::chrono::duration<double, std::milli>(t2 - t1).count(),
                std::chrono::duration<double, std::milli>(t2 - t0).count());

        return result;
    }

    DisparityResult DenseMatchService::computeRawDisparity(const cv::Mat& left,
                                                           const cv::Mat& right,
                                                           const DenseMatchConfig& config) const
    {
        try
        {
            if (config.algorithm == StereoAlgorithm::BlockMatch)
            {
                BlockMatcher bm(config);
                return bm.compute(left, right);
            }
            if (config.algorithm == StereoAlgorithm::OpenCV_SGBM)
            {
                OpenCVSgbmWrapper sgbm(config);
                return sgbm.compute(left, right);
            }

            SgmMatcher sgm(config);
            return sgm.compute(left, right);
        }
        catch (const std::bad_alloc& error)
        {
            throw DenseMatchExecutionError(
                formatDenseMatchFailure(left.size(), config, "Dense matching host allocation failed", error.what()));
        }
        catch (const cv::Exception& error)
        {
            const char* category = error.code == cv::Error::GpuApiCallError
                                       ? "Dense matching CUDA execution or allocation failed"
                                   : error.code == cv::Error::StsNoMem ? "Dense matching OpenCV allocation failed"
                                                                       : "Dense matching OpenCV operation failed";
            throw DenseMatchExecutionError(formatDenseMatchFailure(left.size(), config, category, error.what()));
        }
        catch (const std::length_error& error)
        {
            throw DenseMatchExecutionError(formatDenseMatchFailure(
                left.size(), config, "Dense matching layout is not representable", error.what()));
        }
        catch (const std::overflow_error& error)
        {
            throw DenseMatchExecutionError(
                formatDenseMatchFailure(left.size(), config, "Dense matching layout overflow", error.what()));
        }
    }

    bool DenseMatchService::saveDisparity(const DisparityResult& result, const std::string& filepath)
    {
        if (result.disparity.empty())
        {
            return false;
        }
        return xjw::common::io::writeImage(filepath, result.disparity);
    }

} // namespace xjw::dense_match
