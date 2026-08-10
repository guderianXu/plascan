#include "ProjectMaskInferenceAdapter.h"

#include "birefnet/BiRefNetMaskGenerator.h"
#include "model/BiRefNetModelCatalog.h"
#include "model/ModelFileResolver.h"
#include "model/U2NetModelCatalog.h"
#include "u2net/U2NetMaskGenerator.h"

#include <QDir>
#include <QStandardPaths>

#include <algorithm>
#include <exception>
#include <optional>
#include <utility>

namespace xjw::gui::project
{
namespace
{

std::string nativeUtf8Path(const QString& path)
{
    const QByteArray bytes = QDir::toNativeSeparators(path).toUtf8();
    return std::string(bytes.constData(), static_cast<std::size_t>(bytes.size()));
}

QString aiBackendToken(const QJsonObject& settings)
{
    QString token = settings.value(QStringLiteral("ai_backend")).toString().trimmed();
    if (token.isEmpty())
    {
        token = settings.value(QStringLiteral("u2net_backend")).toString().trimmed();
    }
    if (token.isEmpty())
    {
        token = settings.value(QStringLiteral("u2net_device"))
                    .toString(QStringLiteral("auto"))
                    .trimmed();
    }
    return token == QLatin1String("cuda") ? QStringLiteral("tensorrt") : token;
}

double aiThreshold(const QJsonObject& settings)
{
    const QJsonValue value = settings.value(QStringLiteral("ai_mask_threshold"));
    const double threshold = value.isDouble()
                                 ? value.toDouble()
                                 : settings.value(QStringLiteral("u2net_mask_threshold")).toDouble(0.5);
    return std::clamp(threshold, 0.01, 0.99);
}

QString engineCacheDirectory(const QString& modelDirectory)
{
    return QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation))
        .filePath(QStringLiteral("models/%1/engines").arg(modelDirectory));
}

std::optional<xjw::mask::U2NetMaskGeneratorConfig>
u2netConfig(const QJsonObject& settings,
            ProjectMaskInferenceAdapter::StatusCallback callback,
            QString* error)
{
    const xjw::common::model::ModelFileResolver resolver;
    const auto status = xjw::common::model::u2netModelStatus(resolver);
    if (!status.isInstalled)
    {
        if (error)
        {
            *error = QStringLiteral("未找到 U2Net ONNX 模型：U2Net_v1.onnx。"
                                    "请放到 PLASCAN_MODEL_DIR 或 resources/models。");
        }
        return std::nullopt;
    }

    xjw::mask::U2NetMaskGeneratorConfig config;
    config.modelPath = nativeUtf8Path(status.modelPath);
    config.backend = xjw::mask::parseU2NetBackendType(aiBackendToken(settings).toStdString())
                         .value_or(xjw::mask::U2NetBackendType::Auto);
    const QJsonValue genericFallback = settings.value(QStringLiteral("ai_allow_fallback"));
    config.allowDeviceFallback = genericFallback.isBool()
                                     ? genericFallback.toBool()
                                     : settings.value(QStringLiteral("u2net_allow_fallback")).toBool(false);
    config.inputSize = xjw::mask::kU2NetModelInputSize;
    config.foregroundThreshold = static_cast<float>(aiThreshold(settings));
    config.morphologyRadius = 1;
    config.minComponentArea = 64;
    config.keepLargestComponent = true;
    config.preferFp16 = true;
    config.engineCacheDirectory = engineCacheDirectory(QStringLiteral("u2net")).toStdString();
    config.statusCallback = std::move(callback);
    return config;
}

std::optional<xjw::mask::BiRefNetMaskGeneratorConfig>
biRefNetConfig(const QJsonObject& settings,
               ProjectMaskInferenceAdapter::StatusCallback callback,
               QString* error)
{
    const xjw::common::model::ModelFileResolver resolver;
    const auto status = xjw::common::model::biRefNetDynamicModelStatus(resolver);
    if (!status.isInstalled)
    {
        if (error)
        {
            *error = QStringLiteral("未找到 BiRefNet Dynamic ONNX 模型："
                                    "birefnet_dynamic/BiRefNet_dynamic_1024.onnx。"
                                    "请在生成蒙版对话框下载，或放到 PLASCAN_MODEL_DIR。");
        }
        return std::nullopt;
    }

    const auto backend = xjw::mask::parseBiRefNetBackendType(aiBackendToken(settings).toStdString());
    if (!backend)
    {
        if (error)
        {
            *error = QStringLiteral("BiRefNet Dynamic 只支持 TensorRT GPU，不能选择 OpenCV CPU。");
        }
        return std::nullopt;
    }

    xjw::mask::BiRefNetMaskGeneratorConfig config;
    config.modelPath = nativeUtf8Path(status.modelPath);
    config.backend = *backend;
    config.inputSize = xjw::mask::kBiRefNetDynamicInputSize;
    config.foregroundThreshold = static_cast<float>(aiThreshold(settings));
    config.morphologyRadius = 0;
    config.minComponentArea = 64;
    config.keepLargestComponent = false;
    config.preferFp16 = true;
    config.engineCacheDirectory = engineCacheDirectory(QStringLiteral("birefnet_dynamic")).toStdString();
    config.statusCallback = std::move(callback);
    return config;
}

} // namespace

ProjectMaskInferenceAdapter::~ProjectMaskInferenceAdapter() = default;

std::unique_ptr<ProjectMaskInferenceAdapter>
ProjectMaskInferenceAdapter::create(const QString& method,
                                    const QJsonObject& settings,
                                    StatusCallback statusCallback,
                                    QString* error)
{
    auto adapter = std::unique_ptr<ProjectMaskInferenceAdapter>(new ProjectMaskInferenceAdapter);
    adapter->_method = method;
    try
    {
        if (method == QLatin1String("u2net"))
        {
            auto config = u2netConfig(settings, std::move(statusCallback), error);
            if (!config)
            {
                return {};
            }
            adapter->_u2net = std::make_unique<xjw::mask::U2NetMaskGenerator>(*config);
            adapter->_metadata.modelId = QStringLiteral("u2net_v1");
            adapter->_metadata.modelFileName = QStringLiteral("U2Net_v1.onnx");
            adapter->_metadata.modelSha256 = QString::fromStdString(adapter->_u2net->modelSha256());
            adapter->_metadata.inputSize = xjw::mask::kU2NetModelInputSize;
        }
        else if (method == QLatin1String("birefnet_dynamic"))
        {
            auto config = biRefNetConfig(settings, std::move(statusCallback), error);
            if (!config)
            {
                return {};
            }
            adapter->_biRefNet = std::make_unique<xjw::mask::BiRefNetMaskGenerator>(*config);
            adapter->_metadata.modelId = QStringLiteral("birefnet_dynamic_1024");
            adapter->_metadata.modelFileName = QStringLiteral("BiRefNet_dynamic_1024.onnx");
            adapter->_metadata.modelSha256 = QString::fromStdString(adapter->_biRefNet->modelSha256());
            adapter->_metadata.inputSize = xjw::mask::kBiRefNetDynamicInputSize;
        }
        else
        {
            if (error)
            {
                *error = QStringLiteral("未知 AI 蒙版模型：%1").arg(method);
            }
            return {};
        }
        adapter->_metadata = adapter->metadata();
        return adapter;
    }
    catch (const std::exception& exception)
    {
        if (error)
        {
            *error = QStringLiteral("%1 模型加载失败：%2")
                         .arg(adapter->_metadata.modelFileName.isEmpty() ? method
                                                                        : adapter->_metadata.modelFileName,
                              QString::fromUtf8(exception.what()));
        }
        return {};
    }
}

ProjectMaskInferenceResult ProjectMaskInferenceAdapter::generate(const cv::Mat& image)
{
    ProjectMaskInferenceResult result = metadata();
    if (_u2net)
    {
        const xjw::mask::U2NetMaskResult generated = _u2net->generate(image);
        result.mask = generated.mask;
        result.backend = QString::fromStdString(xjw::mask::u2netBackendTypeToken(generated.actualBackend));
        result.device = QString::fromStdString(generated.deviceLabel);
        result.precision =
            QString::fromStdString(xjw::mask::u2netInferencePrecisionToken(generated.precision));
        result.environment = QString::fromStdString(generated.environmentSummary);
        result.fallbackReason = QString::fromStdString(generated.fallbackReason);
        result.enginePath = QString::fromStdString(generated.enginePath);
        result.engineReused = generated.engineReused;
        return result;
    }

    const xjw::mask::BiRefNetMaskResult generated = _biRefNet->generate(image);
    result.mask = generated.mask;
    result.backend = QString::fromStdString(xjw::mask::biRefNetBackendTypeToken(generated.actualBackend));
    result.device = QString::fromStdString(generated.deviceLabel);
    result.precision =
        QString::fromStdString(xjw::mask::biRefNetInferencePrecisionToken(generated.precision));
    result.environment = QString::fromStdString(generated.environmentSummary);
    result.enginePath = QString::fromStdString(generated.enginePath);
    result.engineReused = generated.engineReused;
    return result;
}

ProjectMaskInferenceResult ProjectMaskInferenceAdapter::metadata() const
{
    ProjectMaskInferenceResult result = _metadata;
    if (_u2net)
    {
        result.backend = QString::fromStdString(xjw::mask::u2netBackendTypeToken(_u2net->actualBackend()));
        result.device = QString::fromStdString(_u2net->deviceLabel());
        result.precision = QString::fromStdString(xjw::mask::u2netInferencePrecisionToken(_u2net->precision()));
        result.environment = QString::fromStdString(_u2net->environmentSummary());
        result.fallbackReason = QString::fromStdString(_u2net->fallbackReason());
        result.enginePath = QString::fromStdString(_u2net->enginePath());
        result.engineReused = _u2net->engineReused();
    }
    else if (_biRefNet)
    {
        result.backend =
            QString::fromStdString(xjw::mask::biRefNetBackendTypeToken(_biRefNet->actualBackend()));
        result.device = QString::fromStdString(_biRefNet->deviceLabel());
        result.precision =
            QString::fromStdString(xjw::mask::biRefNetInferencePrecisionToken(_biRefNet->precision()));
        result.environment = QString::fromStdString(_biRefNet->environmentSummary());
        result.enginePath = QString::fromStdString(_biRefNet->enginePath());
        result.engineReused = _biRefNet->engineReused();
    }
    return result;
}

} // namespace xjw::gui::project
