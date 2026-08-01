#include "TensorRtLightGlueMatcher.h"

#include "LightGluePostprocessor.h"
#ifdef PLASCAN_TENSORRT_CUDA_POSTPROCESS
#include "TensorRtLightGluePostprocessor.h"
#endif
#include "../FeatureSet.h"
#include "io/PathIO.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xjw::image_matching
{
namespace
{

constexpr const char *kKeypoints0 = "keypoints0";
constexpr const char *kDescriptors0 = "descriptors0";
constexpr const char *kImageSize0 = "image_size0";
constexpr const char *kKeypoints1 = "keypoints1";
constexpr const char *kDescriptors1 = "descriptors1";
constexpr const char *kImageSize1 = "image_size1";
constexpr const char *kValid0 = "valid0";
constexpr const char *kValid1 = "valid1";
constexpr const char *kSimilarity = "similarity";
constexpr const char *kMatchability0 = "matchability0";
constexpr const char *kMatchability1 = "matchability1";

class TensorRtLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *message) noexcept override
    {
        if (severity <= Severity::kERROR && message)
        {
            try
            {
                _lastError = message;
            }
            catch (...)
            {
            }
        }
    }

    const std::string &lastError() const
    {
        return _lastError;
    }

private:
    std::string _lastError;
};

template<typename T>
struct TensorRtDeleter
{
    void operator()(T *object) const noexcept
    {
        delete object;
    }
};

template<typename T>
using TensorRtPtr = std::unique_ptr<T, TensorRtDeleter<T>>;

void checkCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("[TensorRtLightGlueMatcher] ") + operation + ": " +
            cudaGetErrorString(status));
    }
}

class CudaStream
{
public:
    CudaStream() = default;

    ~CudaStream()
    {
        if (_stream)
        {
            cudaStreamDestroy(_stream);
        }
    }

    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    void create(unsigned int flags)
    {
        if (_stream)
        {
            throw std::logic_error("CUDA stream has already been created");
        }
        checkCuda(cudaStreamCreateWithFlags(&_stream, flags), "cudaStreamCreateWithFlags");
    }

    operator cudaStream_t() const
    {
        return _stream;
    }

private:
    cudaStream_t _stream = nullptr;
};

std::pair<float, float> inferImageSize(const xjw::image_matching::FeatureSet &feature)
{
    if (feature.imageWidth > 0 && feature.imageHeight > 0)
    {
        return {static_cast<float>(feature.imageWidth),
                static_cast<float>(feature.imageHeight)};
    }

    float maxX = 1.0f;
    float maxY = 1.0f;
    for (const cv::KeyPoint &keypoint : feature.keypoints)
    {
        maxX = std::max(maxX, keypoint.pt.x);
        maxY = std::max(maxY, keypoint.pt.y);
    }
    return {maxX * 1.1f + 1.0f, maxY * 1.1f + 1.0f};
}

std::vector<float> makeSiftKeypoints(const xjw::image_matching::FeatureSet &feature)
{
    std::vector<float> values(static_cast<std::size_t>(feature.size()) * 4U, 0.0f);
    for (int index = 0; index < feature.size(); ++index)
    {
        const cv::KeyPoint &keypoint = feature.keypoints[static_cast<std::size_t>(index)];
        const std::size_t offset = static_cast<std::size_t>(index) * 4U;
        values[offset] = keypoint.pt.x;
        values[offset + 1U] = keypoint.pt.y;
        values[offset + 2U] = std::max(1.0f, keypoint.size);
        values[offset + 3U] = std::isfinite(keypoint.angle) && keypoint.angle >= 0.0f
            ? keypoint.angle * static_cast<float>(CV_PI / 180.0)
            : 0.0f;
    }
    return values;
}

class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    ~DeviceBuffer()
    {
        if (_data)
        {
            cudaFree(_data);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void ensureSize(std::size_t bytes)
    {
        if (bytes <= _capacity)
        {
            return;
        }
        if (_data)
        {
            checkCuda(cudaFree(_data), "cudaFree");
            _data = nullptr;
            _capacity = 0;
        }
        checkCuda(cudaMalloc(&_data, bytes), "cudaMalloc");
        _capacity = bytes;
    }

    void *data()
    {
        return _data;
    }

private:
    void *_data = nullptr;
    std::size_t _capacity = 0;
};

} // namespace

class TensorRtLightGlueMatcher::Impl
{
public:
    explicit Impl(const TensorRtLightGlueConfig &config)
        : _config(config)
    {
        if (_config.enginePath.empty())
        {
            throw std::invalid_argument("[TensorRtLightGlueMatcher] TensorRT engine path is empty");
        }

        checkCuda(cudaSetDevice(std::max(0, _config.cudaDevice)), "cudaSetDevice");
        _stream.create(cudaStreamNonBlocking);

        std::ifstream input = xjw::common::io::openInputFile(_config.enginePath);
        if (!input)
        {
            throw std::runtime_error(
                "[TensorRtLightGlueMatcher] Cannot open TensorRT engine: " + _config.enginePath);
        }
        input.seekg(0, std::ios::end);
        const std::streamoff size = input.tellg();
        input.seekg(0, std::ios::beg);
        if (size <= 0)
        {
            throw std::runtime_error("[TensorRtLightGlueMatcher] TensorRT engine is empty");
        }
        std::vector<char> bytes(static_cast<std::size_t>(size));
        input.read(bytes.data(), static_cast<std::streamsize>(size));
        if (!input)
        {
            throw std::runtime_error("[TensorRtLightGlueMatcher] Failed to read TensorRT engine");
        }

        _runtime.reset(nvinfer1::createInferRuntime(_logger));
        if (!_runtime)
        {
            throwTensorRtError("Cannot create TensorRT runtime");
        }
        _engine.reset(_runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!_engine)
        {
            throwTensorRtError("Cannot deserialize TensorRT engine");
        }
        _context.reset(_engine->createExecutionContext());
        if (!_context)
        {
            throwTensorRtError("Cannot create TensorRT execution context");
        }

        validateTensor(kKeypoints0, nvinfer1::TensorIOMode::kINPUT,
                       nvinfer1::DataType::kFLOAT);
        validateTensor(kDescriptors0, nvinfer1::TensorIOMode::kINPUT,
                       nvinfer1::DataType::kFLOAT);
        validateTensor(kImageSize0, nvinfer1::TensorIOMode::kINPUT,
                       nvinfer1::DataType::kFLOAT);
        validateTensor(kKeypoints1, nvinfer1::TensorIOMode::kINPUT,
                       nvinfer1::DataType::kFLOAT);
        validateTensor(kDescriptors1, nvinfer1::TensorIOMode::kINPUT,
                       nvinfer1::DataType::kFLOAT);
        validateTensor(kImageSize1, nvinfer1::TensorIOMode::kINPUT,
                       nvinfer1::DataType::kFLOAT);
        validateTensor(kValid0, nvinfer1::TensorIOMode::kINPUT,
                       nvinfer1::DataType::kBOOL);
        validateTensor(kValid1, nvinfer1::TensorIOMode::kINPUT,
                       nvinfer1::DataType::kBOOL);
        validateTensor(kSimilarity, nvinfer1::TensorIOMode::kOUTPUT,
                       nvinfer1::DataType::kFLOAT);
        validateTensor(kMatchability0, nvinfer1::TensorIOMode::kOUTPUT,
                       nvinfer1::DataType::kFLOAT);
        validateTensor(kMatchability1, nvinfer1::TensorIOMode::kOUTPUT,
                       nvinfer1::DataType::kFLOAT);

        const nvinfer1::Dims keypointShape = _engine->getTensorShape(kKeypoints0);
        if (keypointShape.nbDims != 3 || keypointShape.d[0] != 1 ||
            keypointShape.d[1] <= 0 || keypointShape.d[2] != 4)
        {
            throw std::runtime_error(
                "[TensorRtLightGlueMatcher] Engine must use a fixed [1,K,4] keypoint bucket");
        }
        _bucketKeypoints = keypointShape.d[1];
    }

    ~Impl()
    {
        _context.reset();
        _engine.reset();
        _runtime.reset();
    }

    MatchResult match(const xjw::image_matching::FeatureSet &feature0,
                      const xjw::image_matching::FeatureSet &feature1)
    {
        checkCuda(cudaSetDevice(std::max(0, _config.cudaDevice)), "cudaSetDevice");
        if (feature0.empty() || feature1.empty())
        {
            MatchResult empty;
            empty.sourceAlgorithm = "lightglue";
            empty.matches0.assign(static_cast<std::size_t>(feature0.size()), -1);
            empty.matches1.assign(static_cast<std::size_t>(feature1.size()), -1);
            empty.matchingScores0.assign(static_cast<std::size_t>(feature0.size()), 0.0f);
            empty.matchingScores1.assign(static_cast<std::size_t>(feature1.size()), 0.0f);
            return empty;
        }
        validateFeatures(feature0, "left");
        validateFeatures(feature1, "right");

        const int count0 = feature0.size();
        const int count1 = feature1.size();
        if (count0 > _bucketKeypoints || count1 > _bucketKeypoints)
        {
            throw std::runtime_error(
                "[TensorRtLightGlueMatcher] Feature count exceeds the fixed TensorRT bucket " +
                std::to_string(_bucketKeypoints) + ": " + std::to_string(count0) + ", " +
                std::to_string(count1));
        }

        std::vector<float> keypoints0 = makeSiftKeypoints(feature0);
        std::vector<float> keypoints1 = makeSiftKeypoints(feature1);
        keypoints0.resize(static_cast<std::size_t>(_bucketKeypoints) * 4U, 0.0f);
        keypoints1.resize(static_cast<std::size_t>(_bucketKeypoints) * 4U, 0.0f);
        const auto [width0, height0] = inferImageSize(feature0);
        const auto [width1, height1] = inferImageSize(feature1);
        const float imageSize0[2] = {width0, height0};
        const float imageSize1[2] = {width1, height1};

        cv::Mat descriptors0 = feature0.descriptors.isContinuous()
            ? feature0.descriptors
            : feature0.descriptors.clone();
        cv::Mat descriptors1 = feature1.descriptors.isContinuous()
            ? feature1.descriptors
            : feature1.descriptors.clone();

        const std::size_t bucket = static_cast<std::size_t>(_bucketKeypoints);
        std::vector<float> paddedDescriptors0(bucket * 128U, 0.0f);
        std::vector<float> paddedDescriptors1(bucket * 128U, 0.0f);
        std::memcpy(paddedDescriptors0.data(), descriptors0.ptr<float>(),
                    descriptors0.total() * descriptors0.elemSize());
        std::memcpy(paddedDescriptors1.data(), descriptors1.ptr<float>(),
                    descriptors1.total() * descriptors1.elemSize());
        std::vector<std::uint8_t> valid0(bucket, 0U);
        std::vector<std::uint8_t> valid1(bucket, 0U);
        std::fill_n(valid0.begin(), count0, static_cast<std::uint8_t>(1U));
        std::fill_n(valid1.begin(), count1, static_cast<std::uint8_t>(1U));

        const nvinfer1::Dims similarityShape = _context->getTensorShape(kSimilarity);
        const nvinfer1::Dims matchability0Shape = _context->getTensorShape(kMatchability0);
        const nvinfer1::Dims matchability1Shape = _context->getTensorShape(kMatchability1);
        if (similarityShape.nbDims != 3 || similarityShape.d[0] != 1 ||
            similarityShape.d[1] != _bucketKeypoints ||
            similarityShape.d[2] != _bucketKeypoints ||
            matchability0Shape.nbDims != 2 || matchability0Shape.d[0] != 1 ||
            matchability0Shape.d[1] != _bucketKeypoints ||
            matchability1Shape.nbDims != 2 || matchability1Shape.d[0] != 1 ||
            matchability1Shape.d[1] != _bucketKeypoints)
        {
            throwTensorRtError("TensorRT returned invalid LightGlue core output shapes");
        }

        const std::size_t keypoints0Bytes = keypoints0.size() * sizeof(float);
        const std::size_t keypoints1Bytes = keypoints1.size() * sizeof(float);
        const std::size_t descriptors0Bytes = paddedDescriptors0.size() * sizeof(float);
        const std::size_t descriptors1Bytes = paddedDescriptors1.size() * sizeof(float);
        const std::size_t imageSizeBytes = 2U * sizeof(float);
        const std::size_t validBytes = bucket * sizeof(std::uint8_t);
        const std::size_t similarityBytes = bucket * bucket * sizeof(float);
        const std::size_t matchabilityBytes = bucket * sizeof(float);

        _keypoints0.ensureSize(keypoints0Bytes);
        _descriptors0.ensureSize(descriptors0Bytes);
        _imageSize0.ensureSize(imageSizeBytes);
        _keypoints1.ensureSize(keypoints1Bytes);
        _descriptors1.ensureSize(descriptors1Bytes);
        _imageSize1.ensureSize(imageSizeBytes);
        _valid0.ensureSize(validBytes);
        _valid1.ensureSize(validBytes);
        _similarity.ensureSize(similarityBytes);
        _matchability0.ensureSize(matchabilityBytes);
        _matchability1.ensureSize(matchabilityBytes);
#ifdef PLASCAN_TENSORRT_CUDA_POSTPROCESS
        _rowConstants.ensureSize(matchabilityBytes);
        _columnConstants.ensureSize(matchabilityBytes);
        _bestColumns.ensureSize(bucket * sizeof(int));
        _bestRows.ensureSize(bucket * sizeof(int));
        _bestRowScores.ensureSize(matchabilityBytes);
        _outputMatches0.ensureSize(bucket * sizeof(int));
        _outputMatches1.ensureSize(bucket * sizeof(int));
        _outputScores0.ensureSize(matchabilityBytes);
        _outputScores1.ensureSize(matchabilityBytes);
#endif

        copyToDevice(_keypoints0, keypoints0.data(), keypoints0Bytes);
        copyToDevice(_descriptors0, paddedDescriptors0.data(), descriptors0Bytes);
        copyToDevice(_imageSize0, imageSize0, imageSizeBytes);
        copyToDevice(_keypoints1, keypoints1.data(), keypoints1Bytes);
        copyToDevice(_descriptors1, paddedDescriptors1.data(), descriptors1Bytes);
        copyToDevice(_imageSize1, imageSize1, imageSizeBytes);
        copyToDevice(_valid0, valid0.data(), validBytes);
        copyToDevice(_valid1, valid1.data(), validBytes);

        setTensorAddress(kKeypoints0, _keypoints0.data());
        setTensorAddress(kDescriptors0, _descriptors0.data());
        setTensorAddress(kImageSize0, _imageSize0.data());
        setTensorAddress(kKeypoints1, _keypoints1.data());
        setTensorAddress(kDescriptors1, _descriptors1.data());
        setTensorAddress(kImageSize1, _imageSize1.data());
        setTensorAddress(kValid0, _valid0.data());
        setTensorAddress(kValid1, _valid1.data());
        setTensorAddress(kSimilarity, _similarity.data());
        setTensorAddress(kMatchability0, _matchability0.data());
        setTensorAddress(kMatchability1, _matchability1.data());

        if (!_context->enqueueV3(_stream))
        {
            throwTensorRtError("TensorRT LightGlue inference failed");
        }

#ifdef PLASCAN_TENSORRT_CUDA_POSTPROCESS
        TensorRtLightGluePostprocessBuffers postprocessBuffers;
        postprocessBuffers.rowConstants = static_cast<float *>(_rowConstants.data());
        postprocessBuffers.columnConstants = static_cast<float *>(_columnConstants.data());
        postprocessBuffers.bestColumns = static_cast<int *>(_bestColumns.data());
        postprocessBuffers.bestRows = static_cast<int *>(_bestRows.data());
        postprocessBuffers.bestRowScores = static_cast<float *>(_bestRowScores.data());
        postprocessBuffers.matches0 = static_cast<int *>(_outputMatches0.data());
        postprocessBuffers.matches1 = static_cast<int *>(_outputMatches1.data());
        postprocessBuffers.matchingScores0 = static_cast<float *>(_outputScores0.data());
        postprocessBuffers.matchingScores1 = static_cast<float *>(_outputScores1.data());
        checkCuda(
            launchTensorRtLightGluePostprocess(
                static_cast<const float *>(_similarity.data()),
                _bucketKeypoints,
                static_cast<const float *>(_matchability0.data()),
                static_cast<const float *>(_matchability1.data()),
                count0,
                count1,
                _config.scoreThreshold,
                postprocessBuffers,
                _stream),
            "launchTensorRtLightGluePostprocess");

        MatchResult result;
        result.sourceAlgorithm = "lightglue";
        result.matches0.resize(static_cast<std::size_t>(count0));
        result.matches1.resize(static_cast<std::size_t>(count1));
        result.matchingScores0.resize(static_cast<std::size_t>(count0));
        result.matchingScores1.resize(static_cast<std::size_t>(count1));
        checkCuda(cudaMemcpyAsync(result.matches0.data(),
                                  _outputMatches0.data(),
                                  static_cast<std::size_t>(count0) * sizeof(int),
                                  cudaMemcpyDeviceToHost,
                                  _stream),
                  "cudaMemcpyAsync(D2H matches0)");
        checkCuda(cudaMemcpyAsync(result.matches1.data(),
                                  _outputMatches1.data(),
                                  static_cast<std::size_t>(count1) * sizeof(int),
                                  cudaMemcpyDeviceToHost,
                                  _stream),
                  "cudaMemcpyAsync(D2H matches1)");
        checkCuda(cudaMemcpyAsync(result.matchingScores0.data(),
                                  _outputScores0.data(),
                                  static_cast<std::size_t>(count0) * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  _stream),
                  "cudaMemcpyAsync(D2H matchingScores0)");
        checkCuda(cudaMemcpyAsync(result.matchingScores1.data(),
                                  _outputScores1.data(),
                                  static_cast<std::size_t>(count1) * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  _stream),
                  "cudaMemcpyAsync(D2H matchingScores1)");
        checkCuda(cudaStreamSynchronize(_stream), "cudaStreamSynchronize");

        result.cvMatches.reserve(static_cast<std::size_t>(std::min(count0, count1)));
        for (int row = 0; row < count0; ++row)
        {
            const int column = result.matches0[static_cast<std::size_t>(row)];
            if (column >= 0)
            {
                const float confidence = result.matchingScores0[static_cast<std::size_t>(row)];
                result.cvMatches.emplace_back(row, column, 1.0f - confidence);
            }
        }
        result.numMatches = static_cast<int>(result.cvMatches.size());
        return result;
#else
        std::vector<float> hostSimilarity(bucket * bucket);
        std::vector<float> hostMatchability0(bucket);
        std::vector<float> hostMatchability1(bucket);
        checkCuda(cudaMemcpyAsync(hostSimilarity.data(),
                                  _similarity.data(),
                                  similarityBytes,
                                  cudaMemcpyDeviceToHost,
                                  _stream),
                  "cudaMemcpyAsync(D2H similarity)");
        checkCuda(cudaMemcpyAsync(hostMatchability0.data(),
                                  _matchability0.data(),
                                  matchabilityBytes,
                                  cudaMemcpyDeviceToHost,
                                  _stream),
                  "cudaMemcpyAsync(D2H matchability0)");
        checkCuda(cudaMemcpyAsync(hostMatchability1.data(),
                                  _matchability1.data(),
                                  matchabilityBytes,
                                  cudaMemcpyDeviceToHost,
                                  _stream),
                  "cudaMemcpyAsync(D2H matchability1)");
        checkCuda(cudaStreamSynchronize(_stream), "cudaStreamSynchronize");

        return postprocessLightGlueCoreOutputs(hostSimilarity.data(),
                                               _bucketKeypoints,
                                               _bucketKeypoints,
                                               hostMatchability0.data(),
                                               hostMatchability1.data(),
                                               count0,
                                               count1,
                                               _config.scoreThreshold,
                                               "lightglue");
#endif
    }

    int bucketKeypoints() const
    {
        return _bucketKeypoints;
    }

private:
    [[noreturn]] void throwTensorRtError(const std::string &message) const
    {
        std::string detail = message;
        if (!_logger.lastError().empty())
        {
            detail += ": " + _logger.lastError();
        }
        throw std::runtime_error("[TensorRtLightGlueMatcher] " + detail);
    }

    void validateTensor(const char *name,
                        nvinfer1::TensorIOMode expectedMode,
                        nvinfer1::DataType expectedType) const
    {
        if (_engine->getTensorIOMode(name) != expectedMode)
        {
            throw std::runtime_error(
                std::string("[TensorRtLightGlueMatcher] Missing or invalid engine tensor: ") + name);
        }
        if (_engine->getTensorDataType(name) != expectedType)
        {
            throw std::runtime_error(
                std::string("[TensorRtLightGlueMatcher] Engine tensor has an invalid data type: ") +
                name);
        }
    }

    static void validateFeatures(const xjw::image_matching::FeatureSet &feature,
                                 const char *side)
    {
        if (feature.sourceAlgorithm != "sift")
        {
            throw std::invalid_argument(
                std::string("[TensorRtLightGlueMatcher] ") + side +
                " feature set is not SIFT");
        }
        if (feature.descriptors.type() != CV_32F ||
            feature.descriptors.rows != feature.size() ||
            feature.descriptors.cols != 128)
        {
            throw std::invalid_argument(
                std::string("[TensorRtLightGlueMatcher] ") + side +
                " descriptors must be contiguous-compatible CV_32F [N,128]");
        }
    }

    void setTensorAddress(const char *name, void *address)
    {
        if (!_context->setTensorAddress(name, address))
        {
            throwTensorRtError(std::string("Cannot bind TensorRT tensor: ") + name);
        }
    }

    void copyToDevice(DeviceBuffer &destination, const void *source, std::size_t bytes)
    {
        checkCuda(cudaMemcpyAsync(destination.data(),
                                  source,
                                  bytes,
                                  cudaMemcpyHostToDevice,
                                  _stream),
                  "cudaMemcpyAsync(H2D)");
    }

    TensorRtLightGlueConfig _config;
    TensorRtLogger _logger;
    TensorRtPtr<nvinfer1::IRuntime> _runtime;
    TensorRtPtr<nvinfer1::ICudaEngine> _engine;
    TensorRtPtr<nvinfer1::IExecutionContext> _context;
    CudaStream _stream;
    std::int32_t _bucketKeypoints = 0;
    DeviceBuffer _keypoints0;
    DeviceBuffer _descriptors0;
    DeviceBuffer _imageSize0;
    DeviceBuffer _keypoints1;
    DeviceBuffer _descriptors1;
    DeviceBuffer _imageSize1;
    DeviceBuffer _valid0;
    DeviceBuffer _valid1;
    DeviceBuffer _similarity;
    DeviceBuffer _matchability0;
    DeviceBuffer _matchability1;
#ifdef PLASCAN_TENSORRT_CUDA_POSTPROCESS
    DeviceBuffer _rowConstants;
    DeviceBuffer _columnConstants;
    DeviceBuffer _bestColumns;
    DeviceBuffer _bestRows;
    DeviceBuffer _bestRowScores;
    DeviceBuffer _outputMatches0;
    DeviceBuffer _outputMatches1;
    DeviceBuffer _outputScores0;
    DeviceBuffer _outputScores1;
#endif
};

TensorRtLightGlueMatcher::TensorRtLightGlueMatcher(const TensorRtLightGlueConfig &config)
    : _impl(std::make_unique<Impl>(config))
{
}

TensorRtLightGlueMatcher::~TensorRtLightGlueMatcher() = default;

MatchResult TensorRtLightGlueMatcher::match(
    const xjw::image_matching::FeatureSet &feat0,
    const xjw::image_matching::FeatureSet &feat1)
{
    return _impl->match(feat0, feat1);
}

std::string TensorRtLightGlueMatcher::algorithmName() const
{
    return "lightglue";
}

bool TensorRtLightGlueMatcher::isLoaded() const
{
    return static_cast<bool>(_impl);
}

int TensorRtLightGlueMatcher::bucketKeypoints() const
{
    return _impl ? _impl->bucketKeypoints() : 0;
}

} // namespace xjw::image_matching
