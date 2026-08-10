#pragma once

#include "TensorRtBuildSupport.h"
#include "TensorRtEngineBuilder.h"

#include <NvInfer.h>

#include <cstddef>
#include <vector>

namespace xjw::inference::detail
{

    std::vector<TensorRtInputShape> normalizeInputShapes(const TensorRtEngineBuildRequest& request);
    QString validateInputShapes(const std::vector<TensorRtInputShape>& inputShapes);
    QString configureInputProfile(nvinfer1::IBuilder& builder,
                                  nvinfer1::INetworkDefinition& network,
                                  nvinfer1::IBuilderConfig& config,
                                  const std::vector<TensorRtInputShape>& inputShapes);
    QString validateRequiredOutputs(nvinfer1::INetworkDefinition& network, const QStringList& requiredOutputNames);

    std::vector<TensorRtTensorInfo> inspectEngine(const void* serializedData,
                                                  std::size_t serializedSize,
                                                  const std::vector<TensorRtInputShape>& inputShapes,
                                                  TensorRtBuildLogger& logger,
                                                  QString* errorMessage);

} // namespace xjw::inference::detail
