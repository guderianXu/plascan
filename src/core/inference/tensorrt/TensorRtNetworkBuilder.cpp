#include "TensorRtNetworkBuilder.h"

#include "TensorRtTypeConversions.h"

#include <algorithm>
#include <string>
#include <unordered_set>

namespace xjw::inference::detail
{
    namespace
    {

        const TensorRtInputShape* findInputShape(const std::vector<TensorRtInputShape>& inputShapes,
                                                 const QString& name)
        {
            const auto iterator = std::lower_bound(inputShapes.begin(),
                                                   inputShapes.end(),
                                                   name,
                                                   [](const TensorRtInputShape& shape, const QString& candidate)
                                                   { return shape.name < candidate; });
            return iterator != inputShapes.end() && iterator->name == name ? &*iterator : nullptr;
        }

        bool hasDynamicDimension(const nvinfer1::Dims& dimensions)
        {
            for (int index = 0; index < dimensions.nbDims; ++index)
            {
                if (dimensions.d[index] < 0)
                {
                    return true;
                }
            }
            return false;
        }

        QString validateRequestedShape(const QString& name,
                                       const nvinfer1::Dims& modelShape,
                                       const TensorRtInputShape& requested,
                                       nvinfer1::Dims* nativeShape)
        {
            if (!toNativeDims(requested.dimensions, nativeShape) || modelShape.nbDims != nativeShape->nbDims)
            {
                return QStringLiteral("TensorRT 输入 %1 的维数与 ONNX 不匹配").arg(name);
            }
            for (int index = 0; index < modelShape.nbDims; ++index)
            {
                if (modelShape.d[index] > 0 && modelShape.d[index] != nativeShape->d[index])
                {
                    return QStringLiteral("TensorRT 输入 %1 的静态维度 %2 不匹配：ONNX=%3，请求=%4")
                        .arg(name)
                        .arg(index)
                        .arg(modelShape.d[index])
                        .arg(nativeShape->d[index]);
                }
            }
            return QString();
        }

    } // namespace

    std::vector<TensorRtInputShape> normalizeInputShapes(const TensorRtEngineBuildRequest& request)
    {
        std::vector<TensorRtInputShape> result = request.inputShapes;
        if (result.empty() && request.fixedKeypointCount > 0)
        {
            const std::int64_t count = request.fixedKeypointCount;
            result = {{QStringLiteral("keypoints0"), {1, count, 2}},
                      {QStringLiteral("keypoints1"), {1, count, 2}},
                      {QStringLiteral("descriptors0"), {1, count, 256}},
                      {QStringLiteral("descriptors1"), {1, count, 256}},
                      {QStringLiteral("valid0"), {1, count}},
                      {QStringLiteral("valid1"), {1, count}}};
        }
        std::sort(result.begin(),
                  result.end(),
                  [](const TensorRtInputShape& left, const TensorRtInputShape& right)
                  { return left.name < right.name; });
        return result;
    }

    QString validateInputShapes(const std::vector<TensorRtInputShape>& inputShapes)
    {
        QString previous_name;
        for (const TensorRtInputShape& input : inputShapes)
        {
            if (input.name.trimmed().isEmpty())
            {
                return QStringLiteral("TensorRT 输入名称不能为空");
            }
            if (input.name == previous_name)
            {
                return QStringLiteral("TensorRT 输入形状重复：%1").arg(input.name);
            }
            for (const std::int64_t dimension : input.dimensions)
            {
                if (dimension <= 0)
                {
                    return QStringLiteral("TensorRT 输入 %1 包含非正维度").arg(input.name);
                }
            }
            previous_name = input.name;
        }
        return QString();
    }

    QString configureInputProfile(nvinfer1::IBuilder& builder,
                                  nvinfer1::INetworkDefinition& network,
                                  nvinfer1::IBuilderConfig& config,
                                  const std::vector<TensorRtInputShape>& inputShapes)
    {
        std::unordered_set<std::string> model_inputs;
        bool needs_profile = false;
        for (int index = 0; index < network.getNbInputs(); ++index)
        {
            nvinfer1::ITensor* input = network.getInput(index);
            if (!input || !input->getName())
            {
                return QStringLiteral("TensorRT ONNX 包含无效输入");
            }
            const QString name = QString::fromUtf8(input->getName());
            model_inputs.insert(name.toStdString());
            const TensorRtInputShape* requested = findInputShape(inputShapes, name);
            const nvinfer1::Dims model_shape = input->getDimensions();
            if (requested)
            {
                nvinfer1::Dims native_shape{};
                const QString error = validateRequestedShape(name, model_shape, *requested, &native_shape);
                if (!error.isEmpty())
                {
                    return error;
                }
            }
            if (hasDynamicDimension(model_shape))
            {
                if (!requested)
                {
                    return QStringLiteral("动态 TensorRT 输入 %1 缺少固定 inputShapes").arg(name);
                }
                if (input->isShapeTensor())
                {
                    return QStringLiteral("TensorRT shape tensor %1 暂不支持固定 profile").arg(name);
                }
                needs_profile = true;
            }
        }

        for (const TensorRtInputShape& requested : inputShapes)
        {
            if (!model_inputs.contains(requested.name.toStdString()))
            {
                return QStringLiteral("ONNX 不包含请求的 TensorRT 输入：%1").arg(requested.name);
            }
        }
        if (!needs_profile)
        {
            return QString();
        }

        nvinfer1::IOptimizationProfile* profile = builder.createOptimizationProfile();
        if (!profile)
        {
            return QStringLiteral("无法创建 TensorRT optimization profile");
        }
        for (int index = 0; index < network.getNbInputs(); ++index)
        {
            nvinfer1::ITensor* input = network.getInput(index);
            if (!hasDynamicDimension(input->getDimensions()))
            {
                continue;
            }
            const TensorRtInputShape* requested = findInputShape(inputShapes, QString::fromUtf8(input->getName()));
            nvinfer1::Dims shape{};
            toNativeDims(requested->dimensions, &shape);
            const bool accepted = profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, shape) &&
                                  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, shape) &&
                                  profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, shape);
            if (!accepted)
            {
                return QStringLiteral("TensorRT 拒绝输入 %1 的固定 optimization profile")
                    .arg(QString::fromUtf8(input->getName()));
            }
        }
        if (config.addOptimizationProfile(profile) < 0)
        {
            return QStringLiteral("TensorRT 无法添加 optimization profile");
        }
        return QString();
    }

    QString validateRequiredOutputs(nvinfer1::INetworkDefinition& network, const QStringList& requiredOutputNames)
    {
        for (const QString& required : requiredOutputNames)
        {
            bool found = false;
            for (int index = 0; index < network.getNbOutputs(); ++index)
            {
                const nvinfer1::ITensor* output = network.getOutput(index);
                if (output && output->getName() && required == QString::fromUtf8(output->getName()))
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                return QStringLiteral("ONNX 不包含要求的 TensorRT 输出：%1").arg(required);
            }
        }
        return QString();
    }

    std::vector<TensorRtTensorInfo> inspectEngine(const void* serializedData,
                                                  std::size_t serializedSize,
                                                  const std::vector<TensorRtInputShape>& inputShapes,
                                                  TensorRtBuildLogger& logger,
                                                  QString* errorMessage)
    {
        TensorRtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        TensorRtPtr<nvinfer1::ICudaEngine> engine(
            runtime ? runtime->deserializeCudaEngine(serializedData, serializedSize) : nullptr);
        TensorRtPtr<nvinfer1::IExecutionContext> context(engine ? engine->createExecutionContext() : nullptr);
        if (!runtime || !engine || !context)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("TensorRT 无法校验刚构建的 engine：%1").arg(logger.errors());
            }
            return {};
        }

        for (const TensorRtInputShape& requested : inputShapes)
        {
            const QByteArray name = requested.name.toUtf8();
            const nvinfer1::Dims engine_shape = engine->getTensorShape(name.constData());
            if (!hasDynamicDimension(engine_shape))
            {
                continue;
            }
            nvinfer1::Dims shape{};
            if (!toNativeDims(requested.dimensions, &shape) || !context->setInputShape(name.constData(), shape))
            {
                if (errorMessage)
                {
                    *errorMessage = QStringLiteral("TensorRT engine 拒绝输入形状：%1").arg(requested.name);
                }
                return {};
            }
        }

        std::vector<TensorRtTensorInfo> result;
        result.reserve(static_cast<std::size_t>(engine->getNbIOTensors()));
        for (int index = 0; index < engine->getNbIOTensors(); ++index)
        {
            const char* name = engine->getIOTensorName(index);
            if (!name)
            {
                continue;
            }
            TensorRtTensorInfo tensor;
            tensor.name = QString::fromUtf8(name);
            tensor.mode = fromNativeMode(engine->getTensorIOMode(name));
            tensor.dataType = fromNativeDataType(engine->getTensorDataType(name));
            tensor.dimensions = fromNativeDims(context->getTensorShape(name));
            result.push_back(std::move(tensor));
        }
        return result;
    }

} // namespace xjw::inference::detail
