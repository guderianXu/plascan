#include "TensorRtEngineBuilder.h"

namespace xjw::image_matching
{

TensorRtEngineBuildResult ensureTensorRtEngine(
    const TensorRtEngineBuildRequest &request)
{
    TensorRtEngineBuildResult result;
    result.errorMessage = QStringLiteral(
        "当前 PlaScan 构建未启用 TensorRT，无法将 ONNX 编译为本机 engine：%1")
                              .arg(request.onnxPath);
    return result;
}

} // namespace xjw::image_matching
