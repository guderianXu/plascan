#include "SiftComputeBackend.h"

#include "SiftMetalKernels.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace xjw::image_matching
{
namespace
{

struct MetalCandidate
{
    float x = 0.0f;
    float y = 0.0f;
    float size = 0.0f;
    float angle = 0.0f;
    float response = 0.0f;
    std::uint32_t octave = 0;
    std::uint32_t layer = 0;
    std::uint32_t padding = 0;
};

struct MetalNearestMatch
{
    std::int32_t index = -1;
    float similarity = 0.0f;
    float ambiguity = 1.0f;
    float padding = 0.0f;
};

static_assert(sizeof(MetalCandidate) == 32);
static_assert(sizeof(MetalNearestMatch) == 16);

std::runtime_error metalError(const char* operation, NSError* error = nil)
{
    std::string message = std::string("SIFT Metal backend ") + operation;
    if (error)
    {
        message += ": ";
        message += [[error localizedDescription] UTF8String];
    }
    return std::runtime_error(message);
}

class MetalRuntime
{
public:
    MetalRuntime()
    {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device)
        {
            throw metalError("could not create the default device");
        }
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:kSiftMetalSource];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        _library = [_device newLibraryWithSource:source options:options error:&error];
        if (!_library)
        {
            throw metalError("kernel compilation failed", error);
        }
        _queue = [_device newCommandQueue];
        if (!_queue)
        {
            throw metalError("could not create a command queue");
        }
    }

    id<MTLDevice> device() const
    {
        return _device;
    }

    id<MTLCommandQueue> queue() const
    {
        return _queue;
    }

    id<MTLComputePipelineState> pipeline(const char* name)
    {
        const std::string key(name);
        const auto existing = _pipelines.find(key);
        if (existing != _pipelines.end())
        {
            return existing->second;
        }
        NSString* functionName = [NSString stringWithUTF8String:name];
        id<MTLFunction> function = [_library newFunctionWithName:functionName];
        if (!function)
        {
            throw metalError((key + " kernel was not found").c_str());
        }
        NSError* error = nil;
        id<MTLComputePipelineState> result = [_device newComputePipelineStateWithFunction:function error:&error];
        if (!result)
        {
            throw metalError((key + " pipeline creation failed").c_str(), error);
        }
        _pipelines.emplace(key, result);
        return result;
    }

private:
    id<MTLDevice> _device = nil;
    id<MTLLibrary> _library = nil;
    id<MTLCommandQueue> _queue = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> _pipelines;
};

MetalRuntime& metalRuntime()
{
    static MetalRuntime runtime;
    return runtime;
}

id<MTLBuffer> makeBuffer(MetalRuntime& runtime, std::size_t bytes)
{
    id<MTLBuffer> buffer = [runtime.device() newBufferWithLength:std::max<std::size_t>(bytes, 4U)
                                                       options:MTLResourceStorageModeShared];
    if (!buffer)
    {
        throw metalError("buffer allocation failed");
    }
    return buffer;
}

void dispatch1d(id<MTLComputeCommandEncoder> encoder,
                id<MTLComputePipelineState> pipeline,
                std::size_t count)
{
    [encoder setComputePipelineState:pipeline];
    const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256U);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

void dispatch2d(id<MTLComputeCommandEncoder> encoder,
                id<MTLComputePipelineState> pipeline,
                std::size_t width,
                std::size_t height)
{
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake(width, height, 1)
       threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

void complete(id<MTLCommandBuffer> commandBuffer, const char* operation)
{
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError)
    {
        throw metalError(operation, commandBuffer.error);
    }
}

id<MTLBuffer> convertImage(MetalRuntime& runtime, const cv::Mat& image)
{
    cv::Mat contiguous = image.isContinuous() ? image : image.clone();
    const std::size_t pixelCount = contiguous.total();
    id<MTLBuffer> input = makeBuffer(runtime, pixelCount);
    id<MTLBuffer> output = makeBuffer(runtime, pixelCount * sizeof(float));
    std::memcpy(input.contents, contiguous.data, pixelCount);

    id<MTLCommandBuffer> commandBuffer = [runtime.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> pipeline = runtime.pipeline("convert_u8");
    const std::uint32_t count = static_cast<std::uint32_t>(pixelCount);
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    dispatch1d(encoder, pipeline, pixelCount);
    [encoder endEncoding];
    complete(commandBuffer, "image upload failed");
    return output;
}

id<MTLBuffer> gaussianBlur(MetalRuntime& runtime,
                           id<MTLBuffer> input,
                           std::uint32_t width,
                           std::uint32_t height,
                           float sigma)
{
    const std::size_t bytes = static_cast<std::size_t>(width) * height * sizeof(float);
    id<MTLBuffer> temporary = makeBuffer(runtime, bytes);
    id<MTLBuffer> output = makeBuffer(runtime, bytes);
    const int radius = std::clamp(static_cast<int>(std::ceil(3.0f * sigma)), 1, 12);

    id<MTLCommandBuffer> commandBuffer = [runtime.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> horizontal = runtime.pipeline("gaussian_horizontal");
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:temporary offset:0 atIndex:1];
    [encoder setBytes:&width length:sizeof(width) atIndex:2];
    [encoder setBytes:&height length:sizeof(height) atIndex:3];
    [encoder setBytes:&sigma length:sizeof(sigma) atIndex:4];
    [encoder setBytes:&radius length:sizeof(radius) atIndex:5];
    dispatch2d(encoder, horizontal, width, height);

    id<MTLComputePipelineState> vertical = runtime.pipeline("gaussian_vertical");
    [encoder setBuffer:temporary offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&width length:sizeof(width) atIndex:2];
    [encoder setBytes:&height length:sizeof(height) atIndex:3];
    [encoder setBytes:&sigma length:sizeof(sigma) atIndex:4];
    [encoder setBytes:&radius length:sizeof(radius) atIndex:5];
    dispatch2d(encoder, vertical, width, height);
    [encoder endEncoding];
    complete(commandBuffer, "Gaussian pyramid construction failed");
    return output;
}

id<MTLBuffer> difference(MetalRuntime& runtime,
                         id<MTLBuffer> low,
                         id<MTLBuffer> high,
                         std::uint32_t pixelCount)
{
    id<MTLBuffer> output = makeBuffer(runtime, static_cast<std::size_t>(pixelCount) * sizeof(float));
    id<MTLCommandBuffer> commandBuffer = [runtime.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> pipeline = runtime.pipeline("difference");
    [encoder setBuffer:low offset:0 atIndex:0];
    [encoder setBuffer:high offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&pixelCount length:sizeof(pixelCount) atIndex:3];
    dispatch1d(encoder, pipeline, pixelCount);
    [encoder endEncoding];
    complete(commandBuffer, "difference-of-Gaussians construction failed");
    return output;
}

id<MTLBuffer> downsample(MetalRuntime& runtime,
                         id<MTLBuffer> input,
                         std::uint32_t inputWidth,
                         std::uint32_t outputWidth,
                         std::uint32_t outputHeight)
{
    id<MTLBuffer> output = makeBuffer(
        runtime, static_cast<std::size_t>(outputWidth) * outputHeight * sizeof(float));
    id<MTLCommandBuffer> commandBuffer = [runtime.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> pipeline = runtime.pipeline("downsample_half");
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&inputWidth length:sizeof(inputWidth) atIndex:2];
    [encoder setBytes:&outputWidth length:sizeof(outputWidth) atIndex:3];
    [encoder setBytes:&outputHeight length:sizeof(outputHeight) atIndex:4];
    dispatch2d(encoder, pipeline, outputWidth, outputHeight);
    [encoder endEncoding];
    complete(commandBuffer, "octave downsampling failed");
    return output;
}

void appendLayerFeatures(MetalRuntime& runtime,
                         id<MTLBuffer> previousDog,
                         id<MTLBuffer> currentDog,
                         id<MTLBuffer> nextDog,
                         id<MTLBuffer> gaussian,
                         std::uint32_t width,
                         std::uint32_t height,
                         std::uint32_t octave,
                         std::uint32_t layer,
                         float threshold,
                         std::uint32_t capacity,
                         SiftRawFeatures* result)
{
    id<MTLBuffer> candidates = makeBuffer(runtime, static_cast<std::size_t>(capacity) * sizeof(MetalCandidate));
    id<MTLBuffer> descriptors = makeBuffer(
        runtime, static_cast<std::size_t>(capacity) * 128U * sizeof(float));
    id<MTLBuffer> count = makeBuffer(runtime, sizeof(std::uint32_t));
    *static_cast<std::uint32_t*>(count.contents) = 0;

    id<MTLCommandBuffer> commandBuffer = [runtime.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> detect = runtime.pipeline("detect_extrema");
    [encoder setBuffer:previousDog offset:0 atIndex:0];
    [encoder setBuffer:currentDog offset:0 atIndex:1];
    [encoder setBuffer:nextDog offset:0 atIndex:2];
    [encoder setBuffer:candidates offset:0 atIndex:3];
    [encoder setBuffer:count offset:0 atIndex:4];
    [encoder setBytes:&width length:sizeof(width) atIndex:5];
    [encoder setBytes:&height length:sizeof(height) atIndex:6];
    [encoder setBytes:&octave length:sizeof(octave) atIndex:7];
    [encoder setBytes:&layer length:sizeof(layer) atIndex:8];
    [encoder setBytes:&threshold length:sizeof(threshold) atIndex:9];
    [encoder setBytes:&capacity length:sizeof(capacity) atIndex:10];
    dispatch2d(encoder, detect, width, height);

    id<MTLComputePipelineState> orientation = runtime.pipeline("assign_orientation");
    [encoder setBuffer:gaussian offset:0 atIndex:0];
    [encoder setBuffer:candidates offset:0 atIndex:1];
    [encoder setBuffer:count offset:0 atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    [encoder setBytes:&height length:sizeof(height) atIndex:4];
    [encoder setBytes:&capacity length:sizeof(capacity) atIndex:5];
    dispatch1d(encoder, orientation, capacity);

    id<MTLComputePipelineState> descriptor = runtime.pipeline("make_descriptor");
    [encoder setBuffer:gaussian offset:0 atIndex:0];
    [encoder setBuffer:candidates offset:0 atIndex:1];
    [encoder setBuffer:count offset:0 atIndex:2];
    [encoder setBuffer:descriptors offset:0 atIndex:3];
    [encoder setBytes:&width length:sizeof(width) atIndex:4];
    [encoder setBytes:&height length:sizeof(height) atIndex:5];
    [encoder setBytes:&capacity length:sizeof(capacity) atIndex:6];
    dispatch1d(encoder, descriptor, capacity);
    [encoder endEncoding];
    complete(commandBuffer, "keypoint detection or descriptor construction failed");

    const std::uint32_t featureCount = std::min(*static_cast<std::uint32_t*>(count.contents), capacity);
    if (featureCount == 0)
    {
        return;
    }
    const auto* candidateData = static_cast<const MetalCandidate*>(candidates.contents);
    const auto* descriptorData = static_cast<const float*>(descriptors.contents);
    const float octaveScale = std::ldexp(1.0f, static_cast<int>(octave));
    const int firstOutputRow = result->descriptors.rows;
    cv::Mat expanded(firstOutputRow + static_cast<int>(featureCount), 128, CV_32F);
    if (firstOutputRow > 0)
    {
        result->descriptors.copyTo(expanded.rowRange(0, firstOutputRow));
    }
    for (std::uint32_t index = 0; index < featureCount; ++index)
    {
        const MetalCandidate& candidate = candidateData[index];
        cv::KeyPoint keypoint;
        keypoint.pt = cv::Point2f(candidate.x * octaveScale, candidate.y * octaveScale);
        keypoint.size = candidate.size * octaveScale;
        keypoint.angle = candidate.angle;
        keypoint.response = candidate.response;
        keypoint.octave = static_cast<int>(candidate.octave);
        result->keypoints.push_back(keypoint);
        std::copy(descriptorData + static_cast<std::size_t>(index) * 128U,
                  descriptorData + static_cast<std::size_t>(index + 1U) * 128U,
                  expanded.ptr<float>(firstOutputRow + static_cast<int>(index)));
    }
    result->descriptors = std::move(expanded);
}

} // namespace

bool isMetalSiftBackendAvailable(int deviceIndex)
{
    if (deviceIndex != 0)
    {
        return false;
    }
    @autoreleasepool
    {
        try
        {
            return metalRuntime().device() != nil;
        }
        catch (...)
        {
            return false;
        }
    }
}

SiftRawFeatures extractMetalSift(const SiftExtractionRequest& request)
{
    @autoreleasepool
    {
        if (request.deviceIndex != 0)
        {
            throw std::invalid_argument("SIFT Metal backend currently exposes device index 0 only");
        }
        if (request.image.empty() || request.image.type() != CV_8U)
        {
            throw std::invalid_argument("SIFT Metal backend requires a CV_8U grayscale image");
        }
        MetalRuntime& runtime = metalRuntime();
        constexpr int scalesPerOctave = 3;
        constexpr int gaussianLevelCount = scalesPerOctave + 3;
        constexpr int dogLevelCount = gaussianLevelCount - 1;
        const std::uint32_t capacity = static_cast<std::uint32_t>(
            std::clamp(request.maximumFeatures > 0 ? request.maximumFeatures : 32768, 1024, 100000));
        const float threshold = std::clamp(request.contrastThreshold / scalesPerOctave, 0.0001f, 0.1f);

        std::uint32_t width = static_cast<std::uint32_t>(request.image.cols);
        std::uint32_t height = static_cast<std::uint32_t>(request.image.rows);
        id<MTLBuffer> octaveBase = convertImage(runtime, request.image);
        SiftRawFeatures result;
        for (std::uint32_t octave = 0;
             octave < 6U && std::min(width, height) >= 32U && result.keypoints.size() < capacity;
             ++octave)
        {
            std::vector<id<MTLBuffer>> gaussianLevels;
            gaussianLevels.reserve(gaussianLevelCount);
            if (octave == 0U)
            {
                constexpr float sourceSigma = 0.5f;
                constexpr float baseSigma = 1.6f;
                const float incrementalSigma =
                    std::sqrt(baseSigma * baseSigma - sourceSigma * sourceSigma);
                gaussianLevels.push_back(gaussianBlur(
                    runtime, octaveBase, width, height, incrementalSigma));
            }
            else
            {
                // 第 3 个尺度下采样后已经是下一 octave 的 sigma=1.6 基准层。
                gaussianLevels.push_back(octaveBase);
            }
            float previousSigma = 1.6f;
            for (int level = 1; level < gaussianLevelCount; ++level)
            {
                const float sigma = 1.6f * std::pow(2.0f, static_cast<float>(level) / scalesPerOctave);
                const float incrementalSigma =
                    std::sqrt(std::max(0.01f, sigma * sigma - previousSigma * previousSigma));
                gaussianLevels.push_back(
                    gaussianBlur(runtime, gaussianLevels.back(), width, height, incrementalSigma));
                previousSigma = sigma;
            }

            const std::uint32_t pixelCount = width * height;
            std::vector<id<MTLBuffer>> dogLevels;
            dogLevels.reserve(dogLevelCount);
            for (int level = 0; level < dogLevelCount; ++level)
            {
                dogLevels.push_back(
                    difference(runtime, gaussianLevels[level], gaussianLevels[level + 1], pixelCount));
            }
            for (std::uint32_t layer = 1; layer + 1 < dogLevels.size(); ++layer)
            {
                const std::uint32_t remaining =
                    capacity - static_cast<std::uint32_t>(result.keypoints.size());
                if (remaining == 0U)
                {
                    break;
                }
                appendLayerFeatures(runtime,
                                    dogLevels[layer - 1],
                                    dogLevels[layer],
                                    dogLevels[layer + 1],
                                    gaussianLevels[layer],
                                    width,
                                    height,
                                    octave,
                                    layer,
                                    threshold,
                                    remaining,
                                    &result);
            }

            const std::uint32_t nextWidth = width / 2U;
            const std::uint32_t nextHeight = height / 2U;
            if (std::min(nextWidth, nextHeight) < 16U)
            {
                break;
            }
            octaveBase = downsample(runtime,
                                    gaussianLevels[scalesPerOctave],
                                    width,
                                    nextWidth,
                                    nextHeight);
            width = nextWidth;
            height = nextHeight;
        }
        return result;
    }
}

std::vector<SiftNearestMatch> matchMetalSift(const cv::Mat& queryDescriptors,
                                             const cv::Mat& trainDescriptors,
                                             int deviceIndex)
{
    @autoreleasepool
    {
        if (deviceIndex != 0)
        {
            throw std::invalid_argument("SIFT Metal backend currently exposes device index 0 only");
        }
        if (queryDescriptors.type() != CV_32F || trainDescriptors.type() != CV_32F ||
            queryDescriptors.cols != 128 || trainDescriptors.cols != 128)
        {
            throw std::invalid_argument("SIFT Metal matcher requires CV_32F descriptors with 128 columns");
        }
        if (queryDescriptors.empty() || trainDescriptors.empty())
        {
            return std::vector<SiftNearestMatch>(static_cast<std::size_t>(queryDescriptors.rows));
        }
        MetalRuntime& runtime = metalRuntime();
        const cv::Mat query = queryDescriptors.isContinuous() ? queryDescriptors : queryDescriptors.clone();
        const cv::Mat train = trainDescriptors.isContinuous() ? trainDescriptors : trainDescriptors.clone();
        const std::size_t queryBytes = query.total() * query.elemSize();
        const std::size_t trainBytes = train.total() * train.elemSize();
        id<MTLBuffer> queryBuffer = makeBuffer(runtime, queryBytes);
        id<MTLBuffer> trainBuffer = makeBuffer(runtime, trainBytes);
        id<MTLBuffer> outputBuffer = makeBuffer(
            runtime, static_cast<std::size_t>(query.rows) * sizeof(MetalNearestMatch));
        std::memcpy(queryBuffer.contents, query.data, queryBytes);
        std::memcpy(trainBuffer.contents, train.data, trainBytes);

        const std::uint32_t queryCount = static_cast<std::uint32_t>(query.rows);
        const std::uint32_t trainCount = static_cast<std::uint32_t>(train.rows);
        id<MTLCommandBuffer> commandBuffer = [runtime.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        id<MTLComputePipelineState> pipeline = runtime.pipeline("nearest_match");
        [encoder setBuffer:queryBuffer offset:0 atIndex:0];
        [encoder setBuffer:trainBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBytes:&queryCount length:sizeof(queryCount) atIndex:3];
        [encoder setBytes:&trainCount length:sizeof(trainCount) atIndex:4];
        dispatch1d(encoder, pipeline, queryCount);
        [encoder endEncoding];
        complete(commandBuffer, "descriptor matching failed");

        const auto* output = static_cast<const MetalNearestMatch*>(outputBuffer.contents);
        std::vector<SiftNearestMatch> result(static_cast<std::size_t>(query.rows));
        for (int index = 0; index < query.rows; ++index)
        {
            result[static_cast<std::size_t>(index)] = {
                output[index].index, output[index].similarity, output[index].ambiguity};
        }
        return result;
    }
}

} // namespace xjw::image_matching
