// GUI“生成纹理”的无界面入口。
#include "cli_common.h"
#include "CliJsonIO.h"

#include "ModelWorkflowService.h"

#include <QCoreApplication>
#include <QJsonObject>
#include <QString>

#include <cstdio>
#include <string>

namespace
{

QJsonObject readSettingsObject(const QString &path, const QString &settings_key)
{
    QJsonObject root;
    QString error;
    if (!xjw::cli::readJsonFile(path, &root, &error))
    {
        cli::fatal(error.toStdString(), cli::EXIT_IO_ERR);
    }
    if (settings_key.isEmpty())
    {
        return root;
    }
    const QJsonValue value = root.value(settings_key);
    if (!value.isObject())
    {
        cli::fatal(
            QStringLiteral("设置 JSON 中缺少对象: %1")
                .arg(settings_key)
                .toStdString(),
            cli::EXIT_IO_ERR);
    }
    return value.toObject();
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication qt_application(argc, argv);
    CLI::App app{"PlaScan GUI 等价多视图纹理生成工具"};

    std::string mesh_path;
    std::string depth_map_dir;
    std::string output_dir;
    std::string settings_json;
    std::string settings_key = "texture_mapping";
    std::string texture_run_id;
    bool allow_vertex_color_fallback = false;

    app.add_option("--mesh", mesh_path, "输入三角网格 PLY/OBJ 路径")
        ->required();
    app.add_option(
        "--depth-map-dir",
        depth_map_dir,
        "MVS 工作区；提供后执行带深度、掩膜和最终网格遮挡检查的多视图纹理");
    app.add_option("--output-dir", output_dir, "纹理模型输出目录")
        ->required();
    app.add_option("--settings-json", settings_json, "纹理设置 JSON 文件")
        ->required();
    app.add_option(
        "--settings-key",
        settings_key,
        "设置对象键；空字符串表示 JSON 根对象");
    app.add_option(
        "--texture-run-id",
        texture_run_id,
        "可选纹理运行标识；非空时写出运行清单");
    app.add_flag(
        "--allow-vertex-color-fallback",
        allow_vertex_color_fallback,
        "缺少 MVS 相机证据时，明确允许退回顶点色平面纹理");

    CLI11_PARSE(app, argc, argv);

    const QJsonObject settings = readSettingsObject(
        QString::fromUtf8(settings_json),
        QString::fromUtf8(settings_key));
    xjw::mesh::workflow::TextureBuildRequest request;
    request.meshPath = QString::fromUtf8(mesh_path);
    request.depthMapSourcePath = QString::fromUtf8(depth_map_dir);
    request.outputDir = QString::fromUtf8(output_dir);
    request.textureRunId = QString::fromUtf8(texture_run_id);
    request.texture =
        xjw::mesh::workflow::textureConfigFromSettings(settings);
    request.allowVertexColorFallback = allow_vertex_color_fallback;
    request.progress = [](const QString &stage, int percent)
    {
        const QByteArray message =
            QStringLiteral("[%1%] %2\n").arg(percent).arg(stage).toUtf8();
        std::fwrite(
            message.constData(),
            1,
            static_cast<std::size_t>(message.size()),
            stderr);
        std::fflush(stderr);
    };

    const xjw::mesh::workflow::WorkflowResult result =
        xjw::mesh::workflow::buildTextureOnly(request);
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
