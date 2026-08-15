// GUI“工作流程 > 生成模型”的无界面入口。
#include "cli_common.h"
#include "CliJsonIO.h"

#include "ModelWorkflowService.h"

#include <QCoreApplication>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

#include <cstdio>
#include <string>

namespace
{

QJsonObject readSettingsObject(const QString &path, const QString &settings_key)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        cli::fatal(QStringLiteral("无法打开设置 JSON: %1").arg(path).toStdString(), cli::EXIT_IO_ERR);
    }

    QJsonParseError parse_error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !document.isObject())
    {
        cli::fatal(QStringLiteral("设置 JSON 无效: %1 (%2)")
                       .arg(path, parse_error.errorString())
                       .toStdString(),
                   cli::EXIT_IO_ERR);
    }

    const QJsonObject root = document.object();
    if (settings_key.isEmpty())
    {
        return root;
    }
    const QJsonValue value = root.value(settings_key);
    if (!value.isObject())
    {
        cli::fatal(QStringLiteral("设置 JSON 中缺少对象: %1").arg(settings_key).toStdString(),
                   cli::EXIT_IO_ERR);
    }
    return value.toObject();
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication qt_application(argc, argv);
    CLI::App app{"PlaScan GUI 等价模型生成工具"};

    std::string source_data;
    std::string point_cloud;
    std::string depth_map_dir;
    std::string dense_cloud;
    std::string sparse_scaffold;
    std::string sparse_points_json;
    std::string output_dir;
    std::string settings_json;
    std::string settings_key = "generate_model";

    app.add_option("--source-data", source_data, "源数据: point_cloud 或 depth_maps");
    app.add_option("--point-cloud", point_cloud, "点云源 PLY 路径");
    app.add_option("--depth-map-dir", depth_map_dir, "深度图输出目录");
    app.add_option("--dense-cloud", dense_cloud, "仅显式 poisson_legacy 模式使用的密集点云 PLY");
    app.add_option("--sparse-scaffold", sparse_scaffold,
                   "环拍深度补全使用的 SfM 稀疏骨架 PLY");
    app.add_option("--sparse-points-json", sparse_points_json,
                   "稀疏骨架逐点质量元数据 JSON");
    app.add_option("--output-dir", output_dir, "模型输出根目录")->required();
    app.add_option("--settings-json", settings_json, "GUI 设置 JSON 文件")->required();
    app.add_option("--settings-key", settings_key, "设置对象键；空字符串表示 JSON 根对象");

    CLI11_PARSE(app, argc, argv);

    QJsonObject settings = readSettingsObject(QString::fromUtf8(settings_json),
                                              QString::fromUtf8(settings_key));
    QString source_data_qt = QString::fromUtf8(source_data).trimmed();
    if (source_data_qt.isEmpty())
    {
        source_data_qt = settings.value(QStringLiteral("source_data"))
                             .toString(QStringLiteral("point_cloud"));
    }
    if (source_data_qt != QStringLiteral("point_cloud") &&
        source_data_qt != QStringLiteral("depth_maps"))
    {
        cli::fatal("--source-data 仅支持 point_cloud 或 depth_maps", cli::EXIT_ARG_ERR);
    }
    if (source_data_qt == QStringLiteral("point_cloud") && point_cloud.empty())
    {
        cli::fatal("point_cloud 模式缺少 --point-cloud", cli::EXIT_ARG_ERR);
    }
    if (source_data_qt == QStringLiteral("depth_maps") && depth_map_dir.empty())
    {
        cli::fatal("depth_maps 模式缺少 --depth-map-dir", cli::EXIT_ARG_ERR);
    }
    if (sparse_scaffold.empty() != sparse_points_json.empty())
    {
        cli::fatal("--sparse-scaffold 与 --sparse-points-json 必须成对提供",
                   cli::EXIT_ARG_ERR);
    }

    settings[QStringLiteral("source_data")] = source_data_qt;
    if (source_data_qt == QStringLiteral("depth_maps") &&
        !settings.contains(QStringLiteral("reconstruction_mode")))
    {
        settings[QStringLiteral("reconstruction_mode")] = QStringLiteral("depth_tsdf");
    }
    const QString reconstruction_mode =
        settings.value(QStringLiteral("reconstruction_mode")).toString();
    xjw::mesh::workflow::ModelBuildRequest request;
    request.sourceData = source_data_qt;
    request.requestedSourcePath = source_data_qt == QStringLiteral("depth_maps")
        ? QString::fromUtf8(depth_map_dir)
        : QString::fromUtf8(point_cloud);
    request.sourcePointCloudPath = source_data_qt == QStringLiteral("depth_maps")
        ? (reconstruction_mode == QStringLiteral("poisson_legacy")
               ? QString::fromUtf8(dense_cloud)
               : QString())
        : QString::fromUtf8(point_cloud);
    request.depthMapSourcePath = QString::fromUtf8(depth_map_dir);
    request.sparseScaffoldPointCloudPath = QString::fromUtf8(sparse_scaffold);
    request.sparseScaffoldPointsPath = QString::fromUtf8(sparse_points_json);
    request.outputRoot = QString::fromUtf8(output_dir);
    request.settings = settings;
    request.progress = [](const QString &stage, int percent)
    {
        const QByteArray message = QStringLiteral("[%1%] %2\n").arg(percent).arg(stage).toUtf8();
        std::fwrite(message.constData(), 1, static_cast<std::size_t>(message.size()), stderr);
        std::fflush(stderr);
    };

    const xjw::mesh::workflow::WorkflowResult result =
        xjw::mesh::workflow::buildModel(request);
    QJsonObject output = result.payload;
    output[QStringLiteral("ok")] = result.ok;
    if (!result.ok)
    {
        output[QStringLiteral("error")] = result.errorMessage;
        xjw::cli::writeJson(stderr, output);
        return cli::EXIT_ALGO_ERR;
    }

    xjw::cli::writeJson(stdout, output);
    return cli::EXIT_OK;
}
