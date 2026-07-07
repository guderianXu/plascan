#include "model/U2NetModelCatalog.h"

namespace xjw::common::model
{

U2NetModelSpec u2netModelSpec()
{
    return {
        QStringLiteral("u2net_v1"),
        QStringLiteral("U2Net v1"),
        QStringList{
            QStringLiteral("U2Net_v1.onnx"),
            QStringLiteral("u2net_v1.onnx"),
        },
    };
}

U2NetModelStatus u2netModelStatus(const TorchScriptModelResolver &resolver)
{
    U2NetModelStatus status;
    status.spec = u2netModelSpec();

    QString pickedName;
    status.modelPath = resolver.findFirstModel(status.spec.modelFileNames, &pickedName);
    status.isInstalled = !status.modelPath.isEmpty();

    if (status.isInstalled)
    {
        status.label = QStringLiteral("已安装");
        status.detail = QStringLiteral("模型文件：%1").arg(pickedName);
        return status;
    }

    status.label = QStringLiteral("未安装");
    status.missingFiles = status.spec.modelFileNames;
    status.detail = QStringLiteral("缺少：%1；请放到 PLASCAN_MODEL_DIR 或 resources/models。")
        .arg(status.spec.modelFileNames.front());
    return status;
}

} // namespace xjw::common::model
