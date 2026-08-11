#include "TensorRtEngineBuilder.h"

namespace xjw::inference
{

    QString tensorRtBuildPrecisionName(TensorRtBuildPrecision precision)
    {
        return precision == TensorRtBuildPrecision::Fp16 ? QStringLiteral("fp16") : QStringLiteral("fp32");
    }

    TensorRtEngineBuildResult ensureTensorRtEngine(const TensorRtEngineBuildRequest& request)
    {
        TensorRtEngineBuildResult result;
        result.precision = request.precision;
        result.errorMessage =
            QStringLiteral("当前 PlaScan 构建未启用 TensorRT，无法将 ONNX 编译为本机 engine：%1").arg(request.onnxPath);
        if (request.progressCallback)
        {
            request.progressCallback({result.errorMessage, 0, 1});
        }
        return result;
    }

} // namespace xjw::inference
