#include "model/U2NetModelCatalog.h"

#include <QDir>

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

U2NetModelStatus u2netModelStatus(const ModelFileResolver &resolver)
{
    U2NetModelStatus status;
    status.spec = u2netModelSpec();

    QString pickedName;
    status.modelPath = resolver.findFirstModel(status.spec.modelFileNames, &pickedName);
    status.isInstalled = !status.modelPath.isEmpty();

    if (status.isInstalled)
    {
        status.label = QStringLiteral("已安装");
        status.detail = QStringLiteral("模型文件：%1\n路径：%2")
            .arg(pickedName, QDir::toNativeSeparators(status.modelPath));
        return status;
    }

    status.label = QStringLiteral("未安装");
    status.missingFiles = status.spec.modelFileNames;
    const ModelInstallLocation location = resolver.installLocation();
    status.detail = QStringLiteral("缺少：%1\n下载位置（%2）：%3")
        .arg(status.spec.modelFileNames.front(),
             location.label,
             QDir::toNativeSeparators(location.directory));
    return status;
}

} // namespace xjw::common::model
