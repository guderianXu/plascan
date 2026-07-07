#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

#include "Sam21MaskGenerator.h"

#include "MaskGenerator.h"

#include <torch/script.h>
#include <torch/torch.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace xjw::mask
{
namespace
{

bool torchCudaAvailable()
{
#if defined(PLASCAN_TORCH_HAS_CUDA)
    return torch::cuda::is_available();
#else
    return false;
#endif
}

std::string formatDeviceLabel(const torch::Device &device)
{
    if (device.is_cuda())
    {
        return "cuda:" + std::to_string(device.index());
    }
    return "cpu";
}

void requireExistingModel(const std::string &path, const char *role)
{
    if (path.empty() || !std::filesystem::exists(std::filesystem::path(path)))
    {
        throw std::runtime_error(std::string("SAM2.1 ") + role + " TorchScript 模型不存在: " + path);
    }
}

torch::Tensor makeImageTensor(const cv::Mat &image, int inputSize, const torch::Device &device)
{
    if (image.empty())
    {
        throw std::runtime_error("SAM2.1 输入图像为空");
    }
    if (inputSize <= 0)
    {
        throw std::runtime_error("SAM2.1 输入尺寸必须大于 0");
    }

    cv::Mat rgb;
    if (image.channels() == 1)
    {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    }
    else if (image.channels() == 3)
    {
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
    }
    else
    {
        throw std::runtime_error("SAM2.1 仅支持 1/3/4 通道图像");
    }

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(inputSize, inputSize), 0.0, 0.0, cv::INTER_AREA);

    cv::Mat rgbFloat;
    resized.convertTo(rgbFloat, CV_32FC3, 1.0 / 255.0);

    auto tensor = torch::from_blob(rgbFloat.data,
                                   {1, inputSize, inputSize, 3},
                                   torch::TensorOptions().dtype(torch::kFloat32))
                      .clone()
                      .permute({0, 3, 1, 2});

    const auto mean = torch::tensor({0.485f, 0.456f, 0.406f},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                          .view({1, 3, 1, 1});
    const auto std = torch::tensor({0.229f, 0.224f, 0.225f},
                                   torch::TensorOptions().dtype(torch::kFloat32))
                         .view({1, 3, 1, 1});
    tensor = (tensor - mean) / std;
    return tensor.to(device);
}

torch::Tensor makePointCoordsTensor(const Sam21Prompt &prompt,
                                    const cv::Size &imageSize,
                                    int inputSize,
                                    const torch::Device &device)
{
    if (prompt.points.empty())
    {
        return torch::empty({1, 0, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    std::vector<float> coords;
    coords.reserve(prompt.points.size() * 2);
    const float sx = static_cast<float>(inputSize) / static_cast<float>(std::max(1, imageSize.width));
    const float sy = static_cast<float>(inputSize) / static_cast<float>(std::max(1, imageSize.height));
    for (const Sam21PointPrompt &point : prompt.points)
    {
        coords.push_back(point.x * sx);
        coords.push_back(point.y * sy);
    }

    return torch::from_blob(coords.data(),
                            {1, static_cast<long long>(prompt.points.size()), 2},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

torch::Tensor makePointLabelsTensor(const Sam21Prompt &prompt, const torch::Device &device)
{
    if (prompt.points.empty())
    {
        return torch::empty({1, 0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }

    std::vector<int> labels;
    labels.reserve(prompt.points.size());
    for (const Sam21PointPrompt &point : prompt.points)
    {
        labels.push_back(point.label != 0 ? 1 : 0);
    }

    return torch::from_blob(labels.data(),
                            {1, static_cast<long long>(prompt.points.size())},
                            torch::TensorOptions().dtype(torch::kInt32))
        .clone()
        .to(device);
}

torch::Tensor makeBoxTensor(const Sam21Prompt &prompt,
                            const cv::Size &imageSize,
                            int inputSize,
                            const torch::Device &device)
{
    std::array<float, 4> box = {0.0f, 0.0f, 0.0f, 0.0f};
    if (prompt.hasBox)
    {
        const float sx = static_cast<float>(inputSize) / static_cast<float>(std::max(1, imageSize.width));
        const float sy = static_cast<float>(inputSize) / static_cast<float>(std::max(1, imageSize.height));
        box = {
            prompt.box.x * sx,
            prompt.box.y * sy,
            (prompt.box.x + prompt.box.width) * sx,
            (prompt.box.y + prompt.box.height) * sy
        };
    }

    return torch::from_blob(box.data(), {1, 4}, torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

torch::Tensor makeMaskInputTensor(const Sam21Prompt &prompt, const torch::Device &device)
{
    if (prompt.maskInput.empty())
    {
        return torch::zeros({1, 1, 256, 256}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    cv::Mat gray;
    if (prompt.maskInput.channels() == 1)
    {
        gray = prompt.maskInput;
    }
    else
    {
        cv::cvtColor(prompt.maskInput, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat foreground = gray == 0;
    cv::Mat resized;
    cv::resize(foreground, resized, cv::Size(256, 256), 0.0, 0.0, cv::INTER_NEAREST);
    resized.convertTo(resized, CV_32FC1, 1.0 / 255.0);

    return torch::from_blob(resized.data, {1, 1, 256, 256}, torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

torch::Tensor makeFlagTensor(bool value, const torch::Device &device)
{
    return torch::tensor({value ? 1 : 0}, torch::TensorOptions().dtype(torch::kInt64).device(device));
}

cv::Mat toGray8(const cv::Mat &image)
{
    if (image.empty())
    {
        return cv::Mat();
    }

    cv::Mat gray;
    if (image.channels() == 1)
    {
        gray = image;
    }
    else if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        return cv::Mat();
    }

    if (gray.depth() == CV_8U)
    {
        return gray.clone();
    }

    cv::Mat normalized;
    cv::normalize(gray, normalized, 0, 255, cv::NORM_MINMAX);
    cv::Mat gray8;
    normalized.convertTo(gray8, CV_8U);
    return gray8;
}

bool hasDarkBorder(const cv::Mat &gray)
{
    if (gray.empty())
    {
        return false;
    }

    const int band = std::max(2, std::min(gray.cols, gray.rows) / 40);
    int borderPixels = 0;
    int darkPixels = 0;
    for (int y = 0; y < gray.rows; ++y)
    {
        const bool borderRow = y < band || y >= gray.rows - band;
        for (int x = 0; x < gray.cols; ++x)
        {
            if (!borderRow && x >= band && x < gray.cols - band)
            {
                continue;
            }
            ++borderPixels;
            if (gray.at<uchar>(y, x) <= 12)
            {
                ++darkPixels;
            }
        }
    }

    return borderPixels > 0 && static_cast<double>(darkPixels) / static_cast<double>(borderPixels) > 0.85;
}

cv::Rect2f paddedRect(const cv::Rect &rect, const cv::Size &size, float paddingRatio)
{
    const float padding = std::max(2.0f,
                                   std::max(rect.width, rect.height) * std::max(0.0f, paddingRatio));
    const float x0 = std::max(0.0f, static_cast<float>(rect.x) - padding);
    const float y0 = std::max(0.0f, static_cast<float>(rect.y) - padding);
    const float x1 = std::min(static_cast<float>(size.width), static_cast<float>(rect.x + rect.width) + padding);
    const float y1 = std::min(static_cast<float>(size.height), static_cast<float>(rect.y + rect.height) + padding);
    return {x0, y0, std::max(1.0f, x1 - x0), std::max(1.0f, y1 - y0)};
}

cv::Mat tensorMaskToPlaScanMask(const torch::Tensor &maskLogits,
                                const torch::Tensor &ious,
                                const cv::Size &outputSize,
                                double threshold,
                                double *score)
{
    torch::Tensor logits = maskLogits;
    if (logits.dim() != 4)
    {
        throw std::runtime_error("SAM2.1 decoder 输出 mask 维度不正确，应为 [B,M,H,W]");
    }

    int64_t bestIndex = 0;
    if (ious.defined() && ious.numel() > 0)
    {
        const torch::Tensor flatIous = ious.reshape({-1}).to(torch::kCPU);
        bestIndex = flatIous.argmax().item<int64_t>();
        if (score)
        {
            *score = flatIous.index({bestIndex}).item<double>();
        }
    }

    const int64_t masksPerImage = logits.size(1);
    bestIndex = std::clamp<int64_t>(bestIndex, 0, std::max<int64_t>(0, masksPerImage - 1));
    torch::Tensor selected = logits.index({0, bestIndex}).to(torch::kCPU).to(torch::kFloat32).contiguous();

    const int rows = static_cast<int>(selected.size(0));
    const int cols = static_cast<int>(selected.size(1));
    cv::Mat logitMat(rows, cols, CV_32FC1, selected.data_ptr<float>());
    cv::Mat resized;
    cv::resize(logitMat, resized, outputSize, 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat foreground = resized > threshold;
    cv::Mat mask(outputSize, CV_8UC1, cv::Scalar(255));
    mask.setTo(cv::Scalar(0), foreground);
    return mask;
}

} // namespace

bool Sam21Prompt::empty() const
{
    return points.empty() && !hasBox && maskInput.empty();
}

Sam21Prompt Sam21Prompt::fullImageBox(const cv::Size &imageSize)
{
    Sam21Prompt prompt;
    prompt.hasBox = true;
    prompt.box = cv::Rect2f(0.0f,
                            0.0f,
                            static_cast<float>(std::max(1, imageSize.width)),
                            static_cast<float>(std::max(1, imageSize.height)));
    return prompt;
}

Sam21Prompt Sam21Prompt::autoBox(const cv::Mat &image, float paddingRatio)
{
    if (image.empty())
    {
        return {};
    }

    const cv::Size imageSize = image.size();
    const cv::Mat gray = toGray8(image);
    if (!hasDarkBorder(gray))
    {
        return fullImageBox(imageSize);
    }

    MaskGenerationOptions options;
    options.method = MaskGenerationMethod::BlackBackground;
    options.threshold = -1.0;
    options.minComponentArea = std::max(64, imageSize.area() / 10000);
    options.morphologyRadius = 2;
    options.keepLargestComponent = true;

    const cv::Mat mask = generateMask(image, options);
    const std::vector<std::vector<cv::Point>> contours = extractMaskContours(mask, true);
    if (contours.empty())
    {
        return fullImageBox(imageSize);
    }

    const cv::Rect bounds = cv::boundingRect(contours.front());
    const double areaRatio = static_cast<double>(bounds.area()) / static_cast<double>(std::max(1, imageSize.area()));
    if (areaRatio < 0.01 || areaRatio > 0.85)
    {
        return fullImageBox(imageSize);
    }

    Sam21Prompt prompt;
    prompt.hasBox = true;
    prompt.box = paddedRect(bounds, imageSize, paddingRatio);
    return prompt;
}

std::string sam21VariantToken(Sam21ModelVariant variant)
{
    switch (variant)
    {
    case Sam21ModelVariant::Tiny:
        return "tiny";
    case Sam21ModelVariant::Small:
        return "small";
    case Sam21ModelVariant::BasePlus:
        return "base_plus";
    case Sam21ModelVariant::Large:
        return "large";
    }
    return "tiny";
}

Sam21ModelVariant sam21VariantFromToken(const std::string &token)
{
    if (token == "small")
    {
        return Sam21ModelVariant::Small;
    }
    if (token == "base_plus" || token == "base-plus")
    {
        return Sam21ModelVariant::BasePlus;
    }
    if (token == "large")
    {
        return Sam21ModelVariant::Large;
    }
    return Sam21ModelVariant::Tiny;
}

Sam21TorchScriptModelNames sam21TorchScriptModelNames(Sam21ModelVariant variant, bool useCuda)
{
    const std::string token = sam21VariantToken(variant);
    const std::string device = useCuda ? "cuda" : "cpu";
    return {
        "sam21_hiera_" + token + "_encoder_" + device + ".pt",
        "sam21_hiera_" + token + "_decoder_" + device + ".pt"
    };
}

class Sam21MaskGenerator::Impl
{
public:
    explicit Impl(const Sam21MaskGeneratorConfig &config)
        : _config(config)
    {
        loadModels();
    }

    Sam21MaskResult generate(const cv::Mat &image, const Sam21Prompt &prompt)
    {
        if (prompt.empty())
        {
            throw std::runtime_error("SAM2.1 需要点、框或已有蒙版提示");
        }

        torch::NoGradGuard noGrad;
        const torch::Tensor input = makeImageTensor(image, _config.inputSize, _device);
        const auto encoderOutput = _encoder.forward({input}).toTuple();
        if (!encoderOutput || encoderOutput->elements().size() < 3)
        {
            throw std::runtime_error("SAM2.1 encoder 输出应为 image_embed, high_res_0, high_res_1");
        }

        const torch::Tensor imageEmbed = encoderOutput->elements()[0].toTensor();
        const torch::Tensor highRes0 = encoderOutput->elements()[1].toTensor();
        const torch::Tensor highRes1 = encoderOutput->elements()[2].toTensor();
        const torch::Tensor pointCoords = makePointCoordsTensor(prompt, image.size(), _config.inputSize, _device);
        const torch::Tensor pointLabels = makePointLabelsTensor(prompt, _device);
        const torch::Tensor box = makeBoxTensor(prompt, image.size(), _config.inputSize, _device);
        const torch::Tensor maskInput = makeMaskInputTensor(prompt, _device);

        std::vector<torch::jit::IValue> inputs;
        inputs.reserve(10);
        inputs.emplace_back(imageEmbed);
        inputs.emplace_back(highRes0);
        inputs.emplace_back(highRes1);
        inputs.emplace_back(pointCoords);
        inputs.emplace_back(pointLabels);
        inputs.emplace_back(box);
        inputs.emplace_back(makeFlagTensor(prompt.hasBox, _device));
        inputs.emplace_back(maskInput);
        inputs.emplace_back(makeFlagTensor(!prompt.maskInput.empty(), _device));
        inputs.emplace_back(makeFlagTensor(_config.multimaskOutput, _device));

        const auto decoderOutput = _decoder.forward(inputs).toTuple();
        if (!decoderOutput || decoderOutput->elements().size() < 2)
        {
            throw std::runtime_error("SAM2.1 decoder 输出应至少包含 mask_logits 和 iou_predictions");
        }

        double score = 0.0;
        cv::Mat mask = tensorMaskToPlaScanMask(decoderOutput->elements()[0].toTensor(),
                                               decoderOutput->elements()[1].toTensor(),
                                               image.size(),
                                               _config.maskThreshold,
                                               &score);
        return {mask, _device.is_cuda(), formatDeviceLabel(_device), _warning, score};
    }

    bool usedCuda() const
    {
        return _device.is_cuda();
    }

    std::string deviceLabel() const
    {
        return formatDeviceLabel(_device);
    }

private:
    void loadModels()
    {
        const bool cudaRequested = _config.useCuda;
        const bool cudaAvailable = cudaRequested && torchCudaAvailable();
        _device = cudaAvailable ? torch::Device(torch::kCUDA, std::max(0, _config.cudaDevice))
                                : torch::Device(torch::kCPU);
        if (cudaRequested && !cudaAvailable)
        {
            if (!_config.allowDeviceFallback)
            {
                throw std::runtime_error("SAM2.1 请求 CUDA，但当前 LibTorch/CUDA 不可用");
            }
            _warning = "SAM2.1 CUDA 不可用，已回退 CPU";
        }

        std::string encoderPath = _config.encoderModelPath;
        std::string decoderPath = _config.decoderModelPath;
        if (!_device.is_cuda() && !_config.cpuEncoderModelPath.empty() && !_config.cpuDecoderModelPath.empty())
        {
            encoderPath = _config.cpuEncoderModelPath;
            decoderPath = _config.cpuDecoderModelPath;
        }

        try
        {
            requireExistingModel(encoderPath, "encoder");
            requireExistingModel(decoderPath, "decoder");
            _encoder = torch::jit::load(encoderPath, _device);
            _decoder = torch::jit::load(decoderPath, _device);
            _encoder.eval();
            _decoder.eval();
        }
        catch (const c10::Error &error)
        {
            if (!_device.is_cuda() || !_config.allowDeviceFallback)
            {
                throw;
            }
            _device = torch::Device(torch::kCPU);
            _warning = std::string("SAM2.1 CUDA 模型加载失败，已回退 CPU: ") + error.what_without_backtrace();
            const std::string fallbackEncoder = _config.cpuEncoderModelPath.empty()
                                                    ? _config.encoderModelPath
                                                    : _config.cpuEncoderModelPath;
            const std::string fallbackDecoder = _config.cpuDecoderModelPath.empty()
                                                    ? _config.decoderModelPath
                                                    : _config.cpuDecoderModelPath;
            requireExistingModel(fallbackEncoder, "CPU encoder");
            requireExistingModel(fallbackDecoder, "CPU decoder");
            _encoder = torch::jit::load(fallbackEncoder, _device);
            _decoder = torch::jit::load(fallbackDecoder, _device);
            _encoder.eval();
            _decoder.eval();
        }
    }

    Sam21MaskGeneratorConfig _config;
    torch::Device _device{torch::kCPU};
    torch::jit::script::Module _encoder;
    torch::jit::script::Module _decoder;
    std::string _warning;
};

Sam21MaskGenerator::Sam21MaskGenerator(const Sam21MaskGeneratorConfig &config)
    : _impl(std::make_unique<Impl>(config))
{
}

Sam21MaskGenerator::~Sam21MaskGenerator() = default;

Sam21MaskResult Sam21MaskGenerator::generate(const cv::Mat &image, const Sam21Prompt &prompt)
{
    return _impl->generate(image, prompt);
}

bool Sam21MaskGenerator::usedCuda() const
{
    return _impl && _impl->usedCuda();
}

std::string Sam21MaskGenerator::deviceLabel() const
{
    return _impl ? _impl->deviceLabel() : std::string("cpu");
}

} // namespace xjw::mask
