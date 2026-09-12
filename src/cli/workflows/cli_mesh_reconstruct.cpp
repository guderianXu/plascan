// GUI“工作流程 > 生成模型”的无界面入口。
#include "cli_common.h"
#include "CliJsonIO.h"

#include "ModelWorkflowService.h"

#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>
#include <QtGlobal>

#include <cmath>
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
    cli::configureApp(app);

    std::string source_data;
    std::string depth_map_dir;
    std::string sparse_scaffold;
    std::string sparse_points_json;
    std::string output_dir;
    std::string settings_json;
    std::string settings_key = "generate_model";

    app.add_option("--source-data", source_data, "源数据: depth_maps 或 rpc_height_plane_sweep");
    app.add_option("--depth-map-dir", depth_map_dir, "深度图输出目录");
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
    settings.remove(QStringLiteral("quality"));
    settings.remove(QStringLiteral("qualityProfile"));
    settings.remove(QStringLiteral("modelQualityProfile"));
    settings.remove(QStringLiteral("targetFaces"));
    settings.remove(QStringLiteral("splitIntoBlocks"));
    settings.remove(QStringLiteral("blockSizeMeters"));
    settings.remove(QStringLiteral("skipBoundaryBlocks"));
    settings.remove(QStringLiteral("saveAfterEachStep"));
    settings.remove(QStringLiteral("strictVolumetricMasks"));
    settings.remove(QStringLiteral("surface_type"));
    settings.remove(QStringLiteral("interpolation"));
    settings.remove(QStringLiteral("calculateVertexColors"));
    settings[QStringLiteral("modelGenerationContractRevision")] = 1;
    settings[QStringLiteral("depthQualityProfile")] = QStringLiteral("medium");
    const QString face_mode = settings.value(QStringLiteral("faceCountMode")).toString();
    const int custom_faces = qBound(1, settings.value(QStringLiteral("faceCountCustom")).toInt(200000), 2000000);
    const int target_faces = face_mode == QStringLiteral("low")      ? 20000
                             : face_mode == QStringLiteral("medium") ? 100000
                             : face_mode == QStringLiteral("high")   ? 200000
                                                                     : custom_faces;
    settings[QStringLiteral("faceCountMode")] =
        face_mode == QStringLiteral("low") || face_mode == QStringLiteral("medium") ||
                face_mode == QStringLiteral("high") || face_mode == QStringLiteral("custom")
            ? face_mode
            : QStringLiteral("high");
    settings[QStringLiteral("faceCountCustom")] = custom_faces;
    settings[QStringLiteral("simplifyTargetFaces")] = target_faces;
    QString source_data_qt = QString::fromUtf8(source_data).trimmed();
    if (source_data_qt.isEmpty())
    {
        source_data_qt = settings.value(QStringLiteral("source_data"))
                             .toString(QStringLiteral("depth_maps"));
    }
    if (source_data_qt != QStringLiteral("depth_maps") &&
        source_data_qt != QStringLiteral("rpc_height_plane_sweep"))
    {
        cli::fatal("canonical v1 生成模型仅支持 --source-data depth_maps 或 rpc_height_plane_sweep",
                   cli::EXIT_ARG_ERR);
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

    const QString requested_mode = settings.value(QStringLiteral("reconstruction_mode"))
                                       .toString()
                                       .trimmed()
                                       .toLower();
    if (!requested_mode.isEmpty() && requested_mode != QStringLiteral("recovered_ooc") &&
        requested_mode != QStringLiteral("rpc_height_plane_sweep"))
    {
        cli::fatal("canonical v1 不接受旧 reconstruction_mode", cli::EXIT_ARG_ERR);
    }
    const bool rpc_mode = source_data_qt == QStringLiteral("rpc_height_plane_sweep");
    if (rpc_mode != (requested_mode == QStringLiteral("rpc_height_plane_sweep")))
    {
        cli::fatal("source_data 与 reconstruction_mode 必须同为 RPC 高程平面扫描", cli::EXIT_ARG_ERR);
    }
    if (rpc_mode)
    {
        const QJsonArray rpc_images = settings.value(QStringLiteral("rpcImagePaths")).toArray();
        const QJsonValue min_height = settings.value(QStringLiteral("rpcHeightMinMeters"));
        const QJsonValue max_height = settings.value(QStringLiteral("rpcHeightMaxMeters"));
        if (rpc_images.size() != 2 || !rpc_images.at(0).isString() || !rpc_images.at(1).isString() ||
            rpc_images.at(0).toString().trimmed().isEmpty() || rpc_images.at(1).toString().trimmed().isEmpty() ||
            !min_height.isDouble() || !max_height.isDouble() || !std::isfinite(min_height.toDouble()) ||
            !std::isfinite(max_height.toDouble()) || min_height.toDouble() >= max_height.toDouble())
        {
            cli::fatal("RPC 模式要求恰好两条非空 rpcImagePaths 和递增的有限物理高程范围",
                       cli::EXIT_ARG_ERR);
        }
    }
    settings[QStringLiteral("source_data")] = source_data_qt;
    settings[QStringLiteral("reconstruction_mode")] =
        rpc_mode ? QStringLiteral("rpc_height_plane_sweep") : QStringLiteral("recovered_ooc");
    settings[QStringLiteral("surfaceQualityProfile")] = settings.value(QStringLiteral("reconstruction_mode"));
    xjw::mesh::workflow::ModelBuildRequest request;
    request.sourceData = source_data_qt;
    request.requestedSourcePath = rpc_mode
        ? settings.value(QStringLiteral("rpcImagePaths")).toArray().at(0).toString()
        : QString::fromUtf8(depth_map_dir);
    request.sourcePointCloudPath = QString();
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
