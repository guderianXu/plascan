#include "model/BiRefNetModelCatalog.h"

#include <QDir>
#include <QFileInfo>

namespace xjw::common::model
{

BiRefNetModelSpec biRefNetDynamicModelSpec()
{
    return {
        QStringLiteral("birefnet_dynamic_1024"),
        QStringLiteral("BiRefNet Dynamic（1024 TensorRT）"),
        QStringList{
            QStringLiteral("birefnet_dynamic/BiRefNet_dynamic_1024.onnx"),
            QStringLiteral("BiRefNet_dynamic_1024.onnx"),
        },
        QStringLiteral("BiRefNet_dynamic_1024.provenance.json"),
    };
}

BiRefNetModelStatus biRefNetDynamicModelStatus(const ModelFileResolver& resolver)
{
    BiRefNetModelStatus status;
    status.spec = biRefNetDynamicModelSpec();

    QString pickedName;
    status.modelPath = resolver.findFirstModel(status.spec.modelFileNames, &pickedName);
    if (!status.modelPath.isEmpty())
    {
        status.provenancePath = QFileInfo(status.modelPath)
                                    .dir()
                                    .filePath(status.spec.provenanceFileName);
        const QFileInfo provenance(status.provenancePath);
        if (!provenance.exists() || !provenance.isFile())
        {
            status.missingFiles.append(status.spec.provenanceFileName);
        }
    }
    else
    {
        status.missingFiles.append(status.spec.modelFileNames.front());
    }
    status.isInstalled = !status.modelPath.isEmpty() && status.missingFiles.isEmpty();
    if (status.isInstalled)
    {
        status.label = QStringLiteral("已安装");
        status.detail = QStringLiteral("模型文件：%1\n来源清单：%2\n路径：%3")
                            .arg(pickedName,
                                 status.spec.provenanceFileName,
                                 QDir::toNativeSeparators(status.modelPath));
        return status;
    }

    status.label = QStringLiteral("未安装");
    const ModelInstallLocation location = resolver.installLocation();
    status.detail = QStringLiteral("缺少：%1\n下载位置（%2）：%3")
                        .arg(status.missingFiles.join(QStringLiteral("、")),
                             location.label,
                             QDir::toNativeSeparators(location.directory));
    return status;
}

} // namespace xjw::common::model
