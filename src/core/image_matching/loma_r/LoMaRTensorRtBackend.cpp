#include "LoMaRTensorRtBackend.h"

#include "inference/tensorrt/TensorRtSession.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xjw::image_matching
{
namespace
{

constexpr const char *kImage = "image";
constexpr const char *kKeypoints = "keypoints";
constexpr const char *kKeypointScores = "keypoint_scores";
constexpr const char *kDescriptors = "descriptors";
constexpr const char *kKeypoints0 = "keypoints0";
constexpr const char *kKeypoints1 = "keypoints1";
constexpr const char *kDescriptors0 = "descriptors0";
constexpr const char *kDescriptors1 = "descriptors1";
constexpr const char *kValid0 = "valid0";
constexpr const char *kValid1 = "valid1";
constexpr const char *kScores = "scores";

bool hasFixedShape(const nvinfer1::Dims &shape, std::initializer_list<int> expected)
{
    if (shape.nbDims != static_cast<int>(expected.size()))
    {
        return false;
    }
    int index = 0;
    for (const int value : expected)
    {
        if (shape.d[index++] != value)
        {
            return false;
        }
    }
    return true;
}

cv::Mat makeBgrImage(const ImageFeatureInput &input)
{
    if (!input.colorImage.empty())
    {
        if (input.colorImage.channels() == 3)
        {
            return input.colorImage;
        }
        cv::Mat bgr;
        cv::cvtColor(input.colorImage, bgr,
                     input.colorImage.channels() == 4 ? cv::COLOR_BGRA2BGR
                                                       : cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (input.grayImage.empty())
    {
        throw std::invalid_argument("LoMa-R received an empty image");
    }
    cv::Mat bgr;
    cv::cvtColor(input.grayImage, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

std::vector<float> preprocessRgb(const cv::Mat &bgr,
                                 const cv::Mat &validMask,
                                 int width,
                                 int height)
{
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    if (!validMask.empty())
    {
        cv::Mat resizedMask;
        cv::resize(validMask, resizedMask, rgb.size(), 0.0, 0.0, cv::INTER_NEAREST);
        rgb.setTo(cv::Scalar(0.0f, 0.0f, 0.0f), resizedMask == 0);
    }

    std::vector<float> chw(static_cast<std::size_t>(3 * width * height));
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    const std::size_t plane = static_cast<std::size_t>(width * height);
    for (int channel = 0; channel < 3; ++channel)
    {
        std::memcpy(chw.data() + static_cast<std::size_t>(channel) * plane,
                    channels[static_cast<std::size_t>(channel)].ptr<float>(),
                    plane * sizeof(float));
    }
    return chw;
}

bool pointAllowed(const cv::Mat &validMask, float x, float y)
{
    if (validMask.empty())
    {
        return true;
    }
    const int px = std::clamp(static_cast<int>(std::lround(x)), 0, validMask.cols - 1);
    const int py = std::clamp(static_cast<int>(std::lround(y)), 0, validMask.rows - 1);
    return validMask.at<std::uint8_t>(py, px) != 0;
}

void validateFeatureSet(const FeatureSet &features, int descriptorDimension)
{
    if (!features.isConsistent() || features.sourceAlgorithm != "loma_r" ||
        features.descriptors.type() != CV_32F ||
        features.descriptors.cols != descriptorDimension)
    {
        throw std::invalid_argument("LoMa-R requires CV_32F DeDoDe-G feature sets");
    }
}

} // namespace

class LoMaRTensorRtBackend::Impl
{
public:
    explicit Impl(LoMaRTensorRtConfig config)
        : _config(std::move(config))
    {
        validateConfiguration();
    }

    FeatureSet extract(const ImageFeatureInput &input) const
    {
        const cv::Mat bgr = makeBgrImage(input);
        std::vector<float> image = preprocessRgb(bgr, input.validMask,
                                                 _config.inputWidth,
                                                 _config.inputHeight);
        const std::size_t keypointCount = static_cast<std::size_t>(_config.featureKeypointCount);
        const std::size_t descriptorDimension =
            static_cast<std::size_t>(_config.descriptorDimension);
        std::vector<float> normalizedKeypoints(keypointCount * 2U);
        std::vector<float> keypointScores(keypointCount);
        std::vector<float> descriptors(keypointCount * descriptorDimension);
        featureSession().execute({
            {kImage, image.data(), image.size() * sizeof(float), true},
            {kKeypoints, normalizedKeypoints.data(), normalizedKeypoints.size() * sizeof(float), false},
            {kKeypointScores, keypointScores.data(), keypointScores.size() * sizeof(float), false},
            {kDescriptors, descriptors.data(), descriptors.size() * sizeof(float), false}});

        const int sourceWidth = bgr.cols;
        const int sourceHeight = bgr.rows;
        const int requested = _config.maxKeypoints > 0
            ? std::min(_config.maxKeypoints, _config.featureKeypointCount)
            : _config.featureKeypointCount;
        FeatureSet result;
        result.imageWidth = input.originalWidth > 0 ? input.originalWidth : sourceWidth;
        result.imageHeight = input.originalHeight > 0 ? input.originalHeight : sourceHeight;
        result.sourceAlgorithm = "loma_r";
        result.keypoints.reserve(static_cast<std::size_t>(requested));
        result.scores.reserve(static_cast<std::size_t>(requested));
        std::vector<int> kept;
        kept.reserve(static_cast<std::size_t>(requested));
        for (int index = 0; index < _config.featureKeypointCount && static_cast<int>(kept.size()) < requested; ++index)
        {
            const float nx = normalizedKeypoints[static_cast<std::size_t>(index) * 2U];
            const float ny = normalizedKeypoints[static_cast<std::size_t>(index) * 2U + 1U];
            const float x = (nx + 1.0f) * 0.5f * static_cast<float>(sourceWidth);
            const float y = (ny + 1.0f) * 0.5f * static_cast<float>(sourceHeight);
            if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0f || y < 0.0f || x >= sourceWidth ||
                y >= sourceHeight || !pointAllowed(input.validMask, x, y))
            {
                continue;
            }
            cv::KeyPoint keypoint;
            keypoint.pt.x = static_cast<float>(x * input.coordinateScale + input.coordinateOffsetX);
            keypoint.pt.y = static_cast<float>(y * input.coordinateScale + input.coordinateOffsetY);
            keypoint.response = keypointScores[static_cast<std::size_t>(index)];
            keypoint.size = static_cast<float>(input.coordinateScale);
            keypoint.angle = -1.0f;
            result.keypoints.push_back(keypoint);
            result.scores.push_back(keypoint.response);
            kept.push_back(index);
        }

        result.descriptors.create(static_cast<int>(kept.size()), _config.descriptorDimension, CV_32F);
        for (int row = 0; row < static_cast<int>(kept.size()); ++row)
        {
            const float* source = descriptors.data() +
                                  static_cast<std::size_t>(kept[static_cast<std::size_t>(row)]) * descriptorDimension;
            std::memcpy(result.descriptors.ptr<float>(row), source,
                        descriptorDimension * sizeof(float));
        }
        return result;
    }

    MatchResult match(const FeatureSet &features0, const FeatureSet &features1)
    {
        validateFeatureSet(features0, _config.descriptorDimension);
        validateFeatureSet(features1, _config.descriptorDimension);
        const int count0 = std::min(features0.size(), _config.keypointCount);
        const int count1 = std::min(features1.size(), _config.keypointCount);
        const std::size_t bucket = static_cast<std::size_t>(_config.keypointCount);
        const std::size_t dimension = static_cast<std::size_t>(_config.descriptorDimension);
        std::vector<float> keypoints0(bucket * 2U, 0.0f);
        std::vector<float> keypoints1(bucket * 2U, 0.0f);
        std::vector<float> descriptors0(bucket * dimension, 0.0f);
        std::vector<float> descriptors1(bucket * dimension, 0.0f);
        std::vector<std::uint8_t> valid0(bucket, 0U);
        std::vector<std::uint8_t> valid1(bucket, 0U);
        fillMatcherInputs(features0, count0, keypoints0, descriptors0, valid0);
        fillMatcherInputs(features1, count1, keypoints1, descriptors1, valid1);
        std::vector<float> scores(bucket * bucket, 0.0f);
        matcherSession().execute({
            {kKeypoints0, keypoints0.data(), keypoints0.size() * sizeof(float), true},
            {kKeypoints1, keypoints1.data(), keypoints1.size() * sizeof(float), true},
            {kDescriptors0, descriptors0.data(), descriptors0.size() * sizeof(float), true},
            {kDescriptors1, descriptors1.data(), descriptors1.size() * sizeof(float), true},
            {kValid0, valid0.data(), valid0.size(), true},
            {kValid1, valid1.data(), valid1.size(), true},
            {kScores, scores.data(), scores.size() * sizeof(float), false}});
        return filterMutualMatches(scores, count0, count1, features0.size(), features1.size());
    }

private:
    void validateConfiguration() const
    {
        if (_config.inputWidth <= 0 || _config.inputHeight <= 0 ||
            _config.keypointCount <= 0 ||
            _config.featureKeypointCount < _config.keypointCount ||
            _config.descriptorDimension <= 0)
        {
            throw std::invalid_argument("LoMa-R TensorRT package metadata is invalid");
        }
    }

    inference::TensorRtSession &featureSession() const
    {
        if (!_feature)
        {
            _feature = std::make_unique<inference::TensorRtSession>(
                _config.featureEnginePath.toStdString(), _config.cudaDevice);
            _feature->validateTensor(kImage, nvinfer1::TensorIOMode::kINPUT,
                                     nvinfer1::DataType::kFLOAT);
            _feature->validateTensor(kKeypoints, nvinfer1::TensorIOMode::kOUTPUT,
                                     nvinfer1::DataType::kFLOAT);
            _feature->validateTensor(kKeypointScores, nvinfer1::TensorIOMode::kOUTPUT,
                                     nvinfer1::DataType::kFLOAT);
            _feature->validateTensor(kDescriptors, nvinfer1::TensorIOMode::kOUTPUT,
                                     nvinfer1::DataType::kFLOAT);
        }
        if (!hasFixedShape(_feature->tensorShape(kImage),
                           {1, 3, _config.inputHeight, _config.inputWidth}) ||
            !hasFixedShape(_feature->tensorShape(kKeypoints),
                           {1, _config.featureKeypointCount, 2}) ||
            !hasFixedShape(_feature->tensorShape(kDescriptors),
                           {1, _config.featureKeypointCount, _config.descriptorDimension}))
        {
            throw std::runtime_error("LoMa-R feature engine shapes do not match the manifest");
        }
        return *_feature;
    }

    inference::TensorRtSession &matcherSession()
    {
        if (!_matcher)
        {
            _matcher = std::make_unique<inference::TensorRtSession>(
                _config.matcherEnginePath.toStdString(), _config.cudaDevice);
            _matcher->validateTensor(kKeypoints0, nvinfer1::TensorIOMode::kINPUT,
                                     nvinfer1::DataType::kFLOAT);
            _matcher->validateTensor(kKeypoints1, nvinfer1::TensorIOMode::kINPUT,
                                     nvinfer1::DataType::kFLOAT);
            _matcher->validateTensor(kDescriptors0, nvinfer1::TensorIOMode::kINPUT,
                                     nvinfer1::DataType::kFLOAT);
            _matcher->validateTensor(kDescriptors1, nvinfer1::TensorIOMode::kINPUT,
                                     nvinfer1::DataType::kFLOAT);
            _matcher->validateTensor(kValid0, nvinfer1::TensorIOMode::kINPUT,
                                     nvinfer1::DataType::kBOOL);
            _matcher->validateTensor(kValid1, nvinfer1::TensorIOMode::kINPUT,
                                     nvinfer1::DataType::kBOOL);
            _matcher->validateTensor(kScores, nvinfer1::TensorIOMode::kOUTPUT,
                                     nvinfer1::DataType::kFLOAT);
            const nvinfer1::Dims3 keypoints{1, _config.keypointCount, 2};
            const nvinfer1::Dims3 descriptors{
                1, _config.keypointCount, _config.descriptorDimension};
            const nvinfer1::Dims2 valid{1, _config.keypointCount};
            _matcher->setInputShape(kKeypoints0, keypoints);
            _matcher->setInputShape(kKeypoints1, keypoints);
            _matcher->setInputShape(kDescriptors0, descriptors);
            _matcher->setInputShape(kDescriptors1, descriptors);
            _matcher->setInputShape(kValid0, valid);
            _matcher->setInputShape(kValid1, valid);
        }
        if (!hasFixedShape(_matcher->tensorShape(kScores),
                           {1, _config.keypointCount, _config.keypointCount}))
        {
            throw std::runtime_error("LoMa-R matcher engine shape does not match the manifest");
        }
        return *_matcher;
    }

    void fillMatcherInputs(const FeatureSet &features,
                           int count,
                           std::vector<float> &keypoints,
                           std::vector<float> &descriptors,
                           std::vector<std::uint8_t> &valid) const
    {
        for (int index = 0; index < count; ++index)
        {
            const cv::Point2f point = features.keypoints[static_cast<std::size_t>(index)].pt;
            keypoints[static_cast<std::size_t>(index) * 2U] =
                2.0f * point.x / static_cast<float>(features.imageWidth) - 1.0f;
            keypoints[static_cast<std::size_t>(index) * 2U + 1U] =
                2.0f * point.y / static_cast<float>(features.imageHeight) - 1.0f;
            std::memcpy(descriptors.data() + static_cast<std::size_t>(index) *
                            static_cast<std::size_t>(_config.descriptorDimension),
                        features.descriptors.ptr<float>(index),
                        static_cast<std::size_t>(_config.descriptorDimension) * sizeof(float));
            valid[static_cast<std::size_t>(index)] = 1U;
        }
    }

    MatchResult filterMutualMatches(const std::vector<float> &scores,
                                    int count0,
                                    int count1,
                                    int fullCount0,
                                    int fullCount1) const
    {
        MatchResult result;
        result.sourceAlgorithm = "loma_r";
        result.matches0.assign(static_cast<std::size_t>(fullCount0), -1);
        result.matches1.assign(static_cast<std::size_t>(fullCount1), -1);
        result.matchingScores0.assign(static_cast<std::size_t>(fullCount0), 0.0f);
        result.matchingScores1.assign(static_cast<std::size_t>(fullCount1), 0.0f);
        const int bucket = _config.keypointCount;
        std::vector<int> bestColumn(static_cast<std::size_t>(count0), -1);
        std::vector<int> bestRow(static_cast<std::size_t>(count1), -1);
        for (int row = 0; row < count0; ++row)
        {
            float best = -1.0f;
            for (int column = 0; column < count1; ++column)
            {
                const float value = scores[static_cast<std::size_t>(row * bucket + column)];
                if (value > best)
                {
                    best = value;
                    bestColumn[static_cast<std::size_t>(row)] = column;
                }
            }
        }
        for (int column = 0; column < count1; ++column)
        {
            float best = -1.0f;
            for (int row = 0; row < count0; ++row)
            {
                const float value = scores[static_cast<std::size_t>(row * bucket + column)];
                if (value > best)
                {
                    best = value;
                    bestRow[static_cast<std::size_t>(column)] = row;
                }
            }
        }
        for (int row = 0; row < count0; ++row)
        {
            const int column = bestColumn[static_cast<std::size_t>(row)];
            if (column < 0 || bestRow[static_cast<std::size_t>(column)] != row)
            {
                continue;
            }
            const float confidence = scores[static_cast<std::size_t>(row * bucket + column)];
            if (confidence <= _config.matchThreshold)
            {
                continue;
            }
            result.matches0[static_cast<std::size_t>(row)] = column;
            result.matches1[static_cast<std::size_t>(column)] = row;
            result.matchingScores0[static_cast<std::size_t>(row)] = confidence;
            result.matchingScores1[static_cast<std::size_t>(column)] = confidence;
            result.cvMatches.emplace_back(row, column, 1.0f - confidence);
        }
        result.numMatches = static_cast<int>(result.cvMatches.size());
        return result;
    }

    LoMaRTensorRtConfig _config;
    mutable std::unique_ptr<inference::TensorRtSession> _feature;
    std::unique_ptr<inference::TensorRtSession> _matcher;
};

LoMaRTensorRtBackend::LoMaRTensorRtBackend(LoMaRTensorRtConfig config)
    : _impl(std::make_unique<Impl>(std::move(config)))
{
}

LoMaRTensorRtBackend::~LoMaRTensorRtBackend() = default;

FeatureSet LoMaRTensorRtBackend::extract(const ImageFeatureInput &input) const
{
    return _impl->extract(input);
}

MatchResult LoMaRTensorRtBackend::match(const FeatureSet &features0,
                                        const FeatureSet &features1)
{
    return _impl->match(features0, features1);
}

} // namespace xjw::image_matching
