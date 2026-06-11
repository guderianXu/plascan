// =============================================================================
// 文件: cli_reconstruct_pipeline.cpp
// 功能: PlaScan 一键重建 CLI
//       .lis(image camera) -> SFM 稀疏点云 -> MVS 稠密点云 -> 三维模型 -> DEM/DOM
// =============================================================================
#include "cli_common.h"

#include "Camera.h"
#include "DepthMapGenerator.h"
#include "ModelWorkflowService.h"
#include "ProjectDenseWorkflowConfig.h"
#include "SFMService.h"
#include "SparseCloudPreprocessor.h"
#include "TerrainPipeline.h"
#include "project/ProjectCommonUtils.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QTextStream>
#include <QTimer>
#include <QtGlobal>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace
{

struct InputItem
{
    QString imagePath;
    QString cameraPath;
    xjw::Camera camera;
};

QString cleanAbsolutePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

QString resolveListToken(const QString &token, const QDir &baseDir)
{
    QString trimmed = token.trimmed();
    if (trimmed.startsWith(QStringLiteral("~/")))
    {
        trimmed = QDir::home().filePath(trimmed.mid(2));
    }

    const QFileInfo info(trimmed);
    if (info.isAbsolute())
    {
        return QDir::cleanPath(info.absoluteFilePath());
    }
    return QDir::cleanPath(QFileInfo(baseDir.filePath(trimmed)).absoluteFilePath());
}

bool hasUnquotedComma(const QString &line)
{
    bool inQuote = false;
    QChar quoteChar;
    bool escaped = false;

    for (const QChar ch : line)
    {
        if (escaped)
        {
            escaped = false;
            continue;
        }
        if (ch == QLatin1Char('\\'))
        {
            escaped = true;
            continue;
        }
        if (inQuote)
        {
            if (ch == quoteChar)
            {
                inQuote = false;
            }
            continue;
        }
        if (ch == QLatin1Char('\'') || ch == QLatin1Char('"'))
        {
            inQuote = true;
            quoteChar = ch;
            continue;
        }
        if (ch == QLatin1Char(','))
        {
            return true;
        }
    }

    return false;
}

bool appendParsedToken(QStringList *parts, QString *token, bool *hasToken)
{
    if (!parts || !token || !hasToken)
    {
        return false;
    }
    if (*hasToken || !token->isEmpty())
    {
        parts->append(token->trimmed());
        token->clear();
        *hasToken = false;
    }
    return true;
}

bool parseShellTokens(const QString &line, QStringList *parts, QString *error)
{
    if (!parts)
    {
        if (error) *error = QStringLiteral("内部错误：列表行输出对象为空");
        return false;
    }

    parts->clear();
    QString token;
    bool hasToken = false;
    bool inQuote = false;
    QChar quoteChar;
    bool escaped = false;

    for (const QChar ch : line)
    {
        if (escaped)
        {
            token.append(ch);
            hasToken = true;
            escaped = false;
            continue;
        }
        if (ch == QLatin1Char('\\'))
        {
            escaped = true;
            hasToken = true;
            continue;
        }
        if (inQuote)
        {
            if (ch == quoteChar)
            {
                inQuote = false;
            }
            else
            {
                token.append(ch);
            }
            hasToken = true;
            continue;
        }
        if (ch == QLatin1Char('\'') || ch == QLatin1Char('"'))
        {
            inQuote = true;
            quoteChar = ch;
            hasToken = true;
            continue;
        }
        if (ch.isSpace())
        {
            appendParsedToken(parts, &token, &hasToken);
            continue;
        }

        token.append(ch);
        hasToken = true;
    }

    if (escaped)
    {
        if (error) *error = QStringLiteral("行尾转义字符缺少目标字符");
        return false;
    }
    if (inQuote)
    {
        if (error) *error = QStringLiteral("引号未闭合");
        return false;
    }

    appendParsedToken(parts, &token, &hasToken);
    return true;
}

bool parseCsvTokens(const QString &line, QStringList *parts, QString *error)
{
    if (!parts)
    {
        if (error) *error = QStringLiteral("内部错误：列表行输出对象为空");
        return false;
    }

    parts->clear();
    QString token;
    bool hasToken = false;
    bool inQuote = false;
    QChar quoteChar;
    bool escaped = false;

    for (int index = 0; index < line.size(); ++index)
    {
        const QChar ch = line.at(index);
        if (escaped)
        {
            token.append(ch);
            hasToken = true;
            escaped = false;
            continue;
        }
        if (ch == QLatin1Char('\\'))
        {
            escaped = true;
            hasToken = true;
            continue;
        }
        if (inQuote)
        {
            if (ch == quoteChar)
            {
                if (quoteChar == QLatin1Char('"')
                    && index + 1 < line.size()
                    && line.at(index + 1) == QLatin1Char('"'))
                {
                    token.append(ch);
                    hasToken = true;
                    ++index;
                }
                else
                {
                    inQuote = false;
                    hasToken = true;
                }
            }
            else
            {
                token.append(ch);
                hasToken = true;
            }
            continue;
        }
        if (ch == QLatin1Char('\'') || ch == QLatin1Char('"'))
        {
            inQuote = true;
            quoteChar = ch;
            hasToken = true;
            continue;
        }
        if (ch == QLatin1Char(','))
        {
            parts->append(token.trimmed());
            token.clear();
            hasToken = false;
            continue;
        }

        token.append(ch);
        hasToken = true;
    }

    if (escaped)
    {
        if (error) *error = QStringLiteral("行尾转义字符缺少目标字符");
        return false;
    }
    if (inQuote)
    {
        if (error) *error = QStringLiteral("引号未闭合");
        return false;
    }

    if (hasToken || !token.isEmpty() || line.endsWith(QLatin1Char(',')))
    {
        parts->append(token.trimmed());
    }
    return true;
}

bool parseListLine(const QString &line, QStringList *parts, QString *error)
{
    if (hasUnquotedComma(line))
    {
        return parseCsvTokens(line, parts, error);
    }
    return parseShellTokens(line, parts, error);
}

QStringList criticalOutputPaths(const QString &outputDir)
{
    const QDir dir(outputDir);
    return {
        dir.filePath(QStringLiteral("report.json")),
        dir.filePath(QStringLiteral("headless.plascan")),
        dir.filePath(QStringLiteral("sparse")),
        dir.filePath(QStringLiteral("mvs/dense_cloud.ply")),
        dir.filePath(QStringLiteral("model")),
        dir.filePath(QStringLiteral("terrain/products/dem.tif")),
        dir.filePath(QStringLiteral("terrain/products/dom.png"))
    };
}

bool validateOutputDirectory(const QString &outputDir, bool force, QString *error)
{
    const QFileInfo outputInfo(outputDir);
    if (outputInfo.exists() && !outputInfo.isDir())
    {
        if (error) *error = QStringLiteral("输出路径已存在但不是目录: %1").arg(outputDir);
        return false;
    }

    if (force)
    {
        return true;
    }

    if (outputInfo.exists())
    {
        const QDir dir(outputDir);
        const QFileInfoList entries = dir.entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries);
        if (!entries.isEmpty())
        {
            if (error) *error = QStringLiteral("输出目录非空，拒绝覆盖已有结果: %1；如需复用/覆盖请添加 --force").arg(outputDir);
            return false;
        }
    }

    for (const QString &path : criticalOutputPaths(outputDir))
    {
        if (QFileInfo::exists(path))
        {
            if (error) *error = QStringLiteral("输出目录已有关键输出文件，拒绝覆盖: %1；如需复用/覆盖请添加 --force").arg(path);
            return false;
        }
    }

    return true;
}

QJsonObject cameraToJson(const xjw::Camera &camera)
{
    const auto intrinsics = camera.intrinsics();
    const auto distortion = camera.distortion();
    const auto center = camera.cameraCenter();
    const auto rotation = camera.cameraToWorldRotation();

    QJsonObject object;
    object[QStringLiteral("model")] = QStringLiteral("tsai");
    object[QStringLiteral("intrinsics_unit")] = QStringLiteral("mm");
    object[QStringLiteral("camera_center_unit")] = QStringLiteral("m");
    object[QStringLiteral("pitch")] = camera.pixelPitch();
    object[QStringLiteral("fu")] = camera.focalXMillimeters();
    object[QStringLiteral("fv")] = camera.focalYMillimeters();
    object[QStringLiteral("cu")] = camera.principalXMillimeters();
    object[QStringLiteral("cv")] = camera.principalYMillimeters();
    object[QStringLiteral("k1")] = distortion.radialK1;
    object[QStringLiteral("k2")] = distortion.radialK2;
    object[QStringLiteral("k3")] = distortion.radialK3;
    object[QStringLiteral("p1")] = distortion.tangentialP1;
    object[QStringLiteral("p2")] = distortion.tangentialP2;
    object[QStringLiteral("u_direction")] = intrinsics.uAxisSign;
    object[QStringLiteral("v_direction")] = intrinsics.vAxisSign;
    object[QStringLiteral("depth_axis_flipped")] = camera.depthAxisFlipped();

    QJsonArray centerArray;
    for (const double value : center)
    {
        centerArray.append(value);
    }
    object[QStringLiteral("C")] = centerArray;

    QJsonArray rotationArray;
    for (const double value : rotation)
    {
        rotationArray.append(value);
    }
    object[QStringLiteral("R")] = rotationArray;
    return object;
}

bool readImageCameraList(const QString &listPath,
                         std::vector<InputItem> *items,
                         QJsonObject *projectMeta,
                         QString *error)
{
    if (!items || !projectMeta)
    {
        if (error) *error = QStringLiteral("内部错误：列表输出对象为空");
        return false;
    }

    QFile file(listPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        if (error) *error = QStringLiteral("无法打开列表文件: %1").arg(listPath);
        return false;
    }

    items->clear();
    QJsonArray imageArray;
    const QDir listDir(QFileInfo(listPath).absolutePath());
    QTextStream stream(&file);
    int lineNumber = 0;
    while (!stream.atEnd())
    {
        ++lineNumber;
        const QString line = stream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#')))
        {
            continue;
        }

        QStringList parts;
        QString parseError;
        if (!parseListLine(line, &parts, &parseError))
        {
            if (error)
            {
                *error = QStringLiteral("%1:%2 %3").arg(listPath).arg(lineNumber).arg(parseError);
            }
            return false;
        }

        if (parts.size() != 2)
        {
            if (error)
            {
                *error = QStringLiteral("%1:%2 需要 '<image> <camera.tsai>'")
                             .arg(listPath)
                             .arg(lineNumber);
            }
            return false;
        }

        InputItem item;
        item.imagePath = resolveListToken(parts.at(0), listDir);
        item.cameraPath = resolveListToken(parts.at(1), listDir);
        if (!QFileInfo::exists(item.imagePath))
        {
            if (error) *error = QStringLiteral("%1:%2 影像不存在: %3").arg(listPath).arg(lineNumber).arg(item.imagePath);
            return false;
        }
        if (!item.camera.loadFromFile(item.cameraPath.toStdString()) || !item.camera.isValid())
        {
            if (error) *error = QStringLiteral("%1:%2 相机读取失败: %3").arg(listPath).arg(lineNumber).arg(item.cameraPath);
            return false;
        }

        QJsonObject imageObject;
        imageObject[QStringLiteral("path")] = item.imagePath;
        imageObject[QStringLiteral("name")] = QFileInfo(item.imagePath).fileName();
        imageObject[QStringLiteral("camera")] = cameraToJson(item.camera);
        imageArray.append(imageObject);
        items->push_back(std::move(item));
    }

    if (items->size() < 2)
    {
        if (error) *error = QStringLiteral("至少需要 2 组 image/camera 输入");
        return false;
    }

    (*projectMeta)[QStringLiteral("images")] = imageArray;
    return true;
}

bool writeDenseCloudPly(const QString &path,
                        const std::vector<xjw::mvs::DensePoint> &cloud,
                        QString *error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    try
    {
        using PlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(cloud.size(), 3);
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(cloud.size(), 3);
        for (std::size_t i = 0; i < cloud.size(); ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            points(row, 0) = cloud[i].x;
            points(row, 1) = cloud[i].y;
            points(row, 2) = cloud[i].z;
            colors(row, 0) = cloud[i].r;
            colors(row, 1) = cloud[i].g;
            colors(row, 2) = cloud[i].b;
        }
        PlaCloud pointCloud(std::move(points));
        pointCloud.setColors(std::move(colors));
        plapoint::io::writePly(path.toStdString(), pointCloud, plapoint::io::PlyFormat::ASCII);
        return true;
    }
    catch (const std::exception &e)
    {
        if (error) *error = QString::fromStdString(e.what());
        return false;
    }
}

bool writeReport(const QString &outputDir, const QJsonObject &report, QJsonObject *writtenReport, QString *error)
{
    if (!QDir().mkpath(outputDir))
    {
        if (error) *error = QStringLiteral("无法创建报告目录: %1").arg(outputDir);
        return false;
    }

    const QString reportPath = QDir(outputDir).filePath(QStringLiteral("report.json"));
    QJsonObject out = report;
    out[QStringLiteral("report_json")] = reportPath;
    const QByteArray payload = QJsonDocument(out).toJson(QJsonDocument::Indented);

    QSaveFile file(reportPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (error) *error = QStringLiteral("无法打开报告文件: %1 (%2)").arg(reportPath, file.errorString());
        return false;
    }
    if (file.write(payload) != payload.size())
    {
        if (error) *error = QStringLiteral("报告写入失败: %1 (%2)").arg(reportPath, file.errorString());
        return false;
    }
    if (!file.commit())
    {
        if (error) *error = QStringLiteral("报告提交失败: %1 (%2)").arg(reportPath, file.errorString());
        return false;
    }

    if (writtenReport)
    {
        *writtenReport = out;
    }
    return true;
}

QJsonArray inputsToJson(const std::vector<InputItem> &items)
{
    QJsonArray array;
    for (const InputItem &item : items)
    {
        array.append(QJsonObject{
            {QStringLiteral("image"), item.imagePath},
            {QStringLiteral("camera"), item.cameraPath}
        });
    }
    return array;
}

QString domOutputPath(const QJsonObject &dom)
{
    QString path = dom.value(QStringLiteral("dom_png")).toString();
    if (path.isEmpty())
    {
        path = dom.value(QStringLiteral("output_path")).toString();
    }
    return path;
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication qtApp(argc, argv);

    CLI::App app{"PlaScan GUI-equivalent reconstruction pipeline"};
    std::string listPathArg;
    std::string outputDirArg = "full_pipeline_output";
    std::string device = "cpu";
    int quality = 3;
    int threads = 8;
    int cudaParallelPairs = 1;
    double demResolution = 0.0;
    int meshResolution = 160;
    bool skipModel = false;
    bool skipTerrain = false;
    bool exportObj = false;
    bool forceOutput = false;

    app.add_option("list_file", listPathArg, "image/camera .lis file")->required();
    app.add_option("-o,--output-dir", outputDirArg, "output directory");
    app.add_option("--device", device, "auto, cpu, cuda")->check(CLI::IsMember({"auto", "cpu", "cuda"}));
    app.add_option("--quality", quality, "SFM quality level 0..3");
    app.add_option("--threads", threads, "CPU thread count");
    app.add_option("--cuda-parallel-pairs", cudaParallelPairs, "SuperGlue CUDA parallel pair count");
    app.add_option("--dem-resolution", demResolution, "DEM/DOM resolution; 0 lets TerrainPipeline choose");
    app.add_option("--mesh-resolution", meshResolution, "mesh reconstruction grid resolution");
    app.add_flag("--skip-model", skipModel, "skip mesh reconstruction");
    app.add_flag("--skip-terrain", skipTerrain, "skip DEM/DOM generation");
    app.add_flag("--export-obj", exportObj, "also export OBJ/MTL/texture where supported");
    app.add_flag("--force", forceOutput, "allow reusing or overwriting a non-empty output directory");

    CLI11_PARSE(app, argc, argv);

    const QString listPath = cleanAbsolutePath(QString::fromStdString(listPathArg));
    const QString outputDir = cleanAbsolutePath(QString::fromStdString(outputDirArg));
    QString error;
    if (!validateOutputDirectory(outputDir, forceOutput, &error))
    {
        std::fprintf(stderr, "输出目录错误: %s\n", qUtf8Printable(error));
        return cli::EXIT_ARG_ERR;
    }
    if (!QDir().mkpath(outputDir))
    {
        std::fprintf(stderr, "输出目录创建失败: %s\n", qUtf8Printable(outputDir));
        return cli::EXIT_IO_ERR;
    }

    std::vector<InputItem> items;
    QJsonObject projectMeta;
    if (!readImageCameraList(listPath, &items, &projectMeta, &error))
    {
        std::fprintf(stderr, "列表读取失败: %s\n", qUtf8Printable(error));
        return cli::EXIT_ARG_ERR;
    }

    QStringList images;
    QStringList cameraPaths;
    for (const InputItem &item : items)
    {
        images.append(item.imagePath);
        cameraPaths.append(item.cameraPath);
    }

    QJsonObject report;
    report[QStringLiteral("status")] = QStringLiteral("running");
    report[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    report[QStringLiteral("list_file")] = listPath;
    report[QStringLiteral("output_dir")] = outputDir;
    report[QStringLiteral("inputs")] = inputsToJson(items);

    auto writeFinalReport = [&](QJsonObject *finalReport) {
        QString reportError;
        if (!writeReport(outputDir, report, finalReport, &reportError))
        {
            std::fprintf(stderr, "报告写入失败: %s\n", qUtf8Printable(reportError));
            return false;
        }
        return true;
    };

    std::fprintf(stdout, "[1/4] SFM 稀疏重建...\n");
    xjw::gui::SFMServiceOptions sfmOptions;
    sfmOptions.images = images;
    sfmOptions.cameraPaths = cameraPaths;
    sfmOptions.projectMeta = projectMeta;
    sfmOptions.plascanPath = QDir(outputDir).filePath(QStringLiteral("headless.plascan"));
    sfmOptions.outputDir = QDir(outputDir).filePath(QStringLiteral("sparse"));
    sfmOptions.device = QString::fromStdString(device);
    sfmOptions.quality = qBound(0, quality, 3);
    sfmOptions.threads = std::max(1, threads);
    sfmOptions.cudaParallelPairs = std::max(1, cudaParallelPairs);
    sfmOptions.progressFn = [](const QString &stage, int percent) {
        std::fprintf(stdout, "  [SFM %3d%%] %s\n", percent, qUtf8Printable(stage));
    };

    const xjw::gui::SFMServiceResult sfmResult = xjw::gui::SFMService::run(sfmOptions);
    QJsonObject sfmJson;
    sfmJson[QStringLiteral("success")] = sfmResult.success;
    sfmJson[QStringLiteral("summary")] = sfmResult.summary;
    sfmJson[QStringLiteral("sparse_cloud")] = sfmResult.sparseCloudPath;
    sfmJson[QStringLiteral("registered_images")] = sfmResult.numRegisteredImages;
    sfmJson[QStringLiteral("points")] = sfmResult.numPoints3D;
    sfmJson[QStringLiteral("mean_reprojection_error")] = sfmResult.meanReprojError;
    report[QStringLiteral("sfm")] = sfmJson;
    if (!sfmResult.success || sfmResult.sparseCloudPath.isEmpty())
    {
        report[QStringLiteral("status")] = QStringLiteral("failed");
        report[QStringLiteral("reason")] = sfmResult.errorMessage;
        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stderr, "SFM 失败: %s\n", qUtf8Printable(sfmResult.errorMessage));
        std::fprintf(stderr, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }

    QMap<QString, xjw::Camera> cameraByImage;
    for (const InputItem &item : items)
    {
        cameraByImage.insert(item.imagePath, item.camera);
    }
    for (auto it = sfmResult.pendingCamUpdates.constBegin(); it != sfmResult.pendingCamUpdates.constEnd(); ++it)
    {
        xjw::Camera camera;
        if (xjw::common::project::cameraFromJson(it.value(), &camera) && camera.isValid())
        {
            cameraByImage.insert(cleanAbsolutePath(it.key()), camera);
        }
    }

    QJsonArray imageMetaArray;
    std::vector<xjw::mvs::CameraView> views;
    views.reserve(items.size());
    for (const InputItem &item : items)
    {
        const QString imagePath = cleanAbsolutePath(item.imagePath);
        const xjw::Camera camera = cameraByImage.value(imagePath, item.camera);
        if (!camera.isValid())
        {
            continue;
        }

        xjw::mvs::CameraView view;
        view.imagePath = imagePath.toStdString();
        view.camera = camera;
        cv::Mat image = cv::imread(view.imagePath, cv::IMREAD_GRAYSCALE);
        if (!image.empty())
        {
            view.imageWidth = image.cols;
            view.imageHeight = image.rows;
        }
        views.push_back(std::move(view));

        imageMetaArray.append(QJsonObject{
            {QStringLiteral("path"), imagePath},
            {QStringLiteral("name"), QFileInfo(imagePath).fileName()},
            {QStringLiteral("camera"), cameraToJson(camera)}
        });
    }
    projectMeta[QStringLiteral("images")] = imageMetaArray;

    if (views.size() < 2)
    {
        report[QStringLiteral("status")] = QStringLiteral("failed");
        report[QStringLiteral("reason")] = QStringLiteral("SFM 后可用于 MVS 的相机不足 2 张");
        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stderr, "MVS 输入不足: report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }

    xjw::mvs::SparseCloud sparse;
    {
        xjw::mvs::SparseCloudPreprocessor preprocessor;
        xjw::mvs::PreprocessResult preprocessResult;
        std::string preprocessError;
        if (preprocessor.run(sfmResult.sparseCloudPath.toStdString(), views, preprocessResult, &preprocessError))
        {
            sparse = preprocessResult.cloud;
        }
        else
        {
            std::fprintf(stderr, "稀疏点云预处理失败，继续尝试 MVS: %s\n", preprocessError.c_str());
        }
    }

    std::fprintf(stdout, "[2/4] MVS 稠密点云...\n");
    const QString mvsDir = QDir(outputDir).filePath(QStringLiteral("mvs"));
    QDir().mkpath(mvsDir);
    xjw::gui::project::DenseGenerationSettings denseSettings;
    denseSettings.threads = std::max(1, threads);
    denseSettings.useCuda = (device == "cuda" || device == "auto");
    denseSettings.pipelineMode = true;
    xjw::mvs::DepthGenConfig depthConfig =
        xjw::gui::project::buildDepthGenConfig(denseSettings, static_cast<int>(views.size()));
    depthConfig.runFusion = true;
    depthConfig.saveIntermediateDepthMaps = true;
    depthConfig.intermediateDir = mvsDir.toStdString();

    xjw::mvs::DepthMapGenerator generator;
    generator.setViews(views);
    generator.setSparseCloud(sparse);
    generator.setConfig(depthConfig);
    generator.setOutputDir(mvsDir.toStdString());

    QEventLoop loop;
    bool mvsOk = false;
    QString mvsError;
    std::vector<xjw::mvs::DensePoint> denseCloud;
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::progressChanged, &loop,
                     [](const QString &stage, float ratio) {
        std::fprintf(stdout, "  [MVS %3d%%] %s\n", static_cast<int>(ratio * 100.0f), qUtf8Printable(stage));
    });
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::errorOccurred, &loop,
                     [&mvsError](const QString &message) {
        mvsError = message;
        std::fprintf(stderr, "  [MVS] %s\n", qUtf8Printable(message));
    });
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::pointCloudReady, &loop,
                     [&denseCloud](const std::vector<xjw::mvs::DensePoint> &cloud) {
        denseCloud = cloud;
    });
    QObject::connect(&generator, &xjw::mvs::DepthMapGenerator::finished, &loop,
                     [&loop, &mvsOk](bool success) {
        mvsOk = success;
        loop.quit();
    });
    QTimer::singleShot(0, &generator, &xjw::mvs::DepthMapGenerator::start);
    loop.exec();

    const QString denseCloudPath = QDir(mvsDir).filePath(QStringLiteral("dense_cloud.ply"));
    if (!mvsOk || denseCloud.empty() || !writeDenseCloudPly(denseCloudPath, denseCloud, &error))
    {
        report[QStringLiteral("status")] = QStringLiteral("failed");
        report[QStringLiteral("reason")] = !error.isEmpty() ? error : (mvsError.isEmpty() ? QStringLiteral("MVS 未生成有效稠密点云") : mvsError);
        QJsonObject finalReport;
        if (!writeFinalReport(&finalReport))
        {
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stderr, "MVS 失败: %s\n", qUtf8Printable(report.value(QStringLiteral("reason")).toString()));
        std::fprintf(stderr, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));
        return cli::EXIT_ALGO_ERR;
    }
    report[QStringLiteral("dense")] = QJsonObject{
        {QStringLiteral("point_cloud"), denseCloudPath},
        {QStringLiteral("points"), static_cast<int>(denseCloud.size())},
        {QStringLiteral("has_rgb"), true}
    };

    if (!skipModel)
    {
        std::fprintf(stdout, "[3/4] 三维网格模型...\n");
        xjw::mesh::workflow::MeshBuildRequest meshRequest;
        meshRequest.pointCloudPath = denseCloudPath;
        meshRequest.outputRoot = QDir(outputDir).filePath(QStringLiteral("model"));
        meshRequest.exportObj = exportObj;
        meshRequest.reconstruction.resolution = qBound(64, meshResolution, 1024);
        meshRequest.reconstruction.poissonDepth = 8;
        meshRequest.reconstruction.simplifyTargetFaces = 18000;
        meshRequest.progress = [](const QString &stage, int percent) {
            std::fprintf(stdout, "  [Mesh %3d%%] %s\n", percent, qUtf8Printable(stage));
        };
        xjw::mesh::workflow::WorkflowResult meshResult;
        try
        {
            meshResult = xjw::mesh::workflow::buildMeshAndOptionalTexture(meshRequest);
        }
        catch (const std::exception &ex)
        {
            meshResult.ok = false;
            meshResult.errorMessage = QStringLiteral("模型生成异常: %1").arg(QString::fromUtf8(ex.what()));
        }
        report[QStringLiteral("model")] = meshResult.payload;
        if (!meshResult.ok)
        {
            report[QStringLiteral("model_error")] = meshResult.errorMessage;
            std::fprintf(stderr, "模型生成失败: %s\n", qUtf8Printable(meshResult.errorMessage));
        }
    }

    if (!skipTerrain)
    {
        std::fprintf(stdout, "[4/4] DEM/DOM 产品...\n");
        const QString terrainDir = QDir(outputDir).filePath(QStringLiteral("terrain"));
        QJsonObject demResult;
        if (!xjw::TerrainPipeline::generateDemProducts(denseCloudPath,
                                                       terrainDir,
                                                       demResolution,
                                                       QStringLiteral("float32"),
                                                       true,
                                                       &demResult,
                                                       &error))
        {
            report[QStringLiteral("terrain_error")] = error;
            std::fprintf(stderr, "DEM 生成失败: %s\n", qUtf8Printable(error));
        }
        else
        {
            QJsonObject domResult;
            const QString domPath = QDir(terrainDir).filePath(QStringLiteral("products/dom.png"));
            if (!xjw::TerrainPipeline::generateOrthoProduct(images,
                                                            demResult.value(QStringLiteral("dem_tif")).toString(),
                                                            domPath,
                                                            demResolution,
                                                            projectMeta,
                                                            &domResult,
                                                            &error))
            {
                report[QStringLiteral("terrain_error")] = error;
                std::fprintf(stderr, "DOM 生成失败: %s\n", qUtf8Printable(error));
            }
            report[QStringLiteral("terrain")] = QJsonObject{
                {QStringLiteral("dem"), demResult},
                {QStringLiteral("dom"), domResult}
            };
        }
    }

    const QJsonObject terrain = report.value(QStringLiteral("terrain")).toObject();
    const QString demPath = terrain.value(QStringLiteral("dem")).toObject().value(QStringLiteral("dem_tif")).toString();
    const QString domPath = domOutputPath(terrain.value(QStringLiteral("dom")).toObject());
    const QJsonObject model = report.value(QStringLiteral("model")).toObject();
    const QString modelPath = model.value(QStringLiteral("final_model_path")).toString(
        model.value(QStringLiteral("model_ply")).toString());

    const bool modelOk = skipModel || (!modelPath.isEmpty() && QFileInfo::exists(modelPath));
    const bool terrainOk = skipTerrain || ((!demPath.isEmpty() && QFileInfo::exists(demPath))
                                           && (!domPath.isEmpty() && QFileInfo::exists(domPath)));
    report[QStringLiteral("status")] = (modelOk && terrainOk) ? QStringLiteral("ok") : QStringLiteral("partial");
    QJsonObject finalReport;
    if (!writeFinalReport(&finalReport))
    {
        return cli::EXIT_IO_ERR;
    }

    std::fprintf(stdout, "status=%s\n", qUtf8Printable(report.value(QStringLiteral("status")).toString()));
    std::fprintf(stdout, "output_dir=%s\n", qUtf8Printable(outputDir));
    std::fprintf(stdout, "sparse_cloud=%s\n", qUtf8Printable(sfmResult.sparseCloudPath));
    std::fprintf(stdout, "dense_cloud=%s points=%zu\n", qUtf8Printable(denseCloudPath), denseCloud.size());
    if (!modelPath.isEmpty()) std::fprintf(stdout, "model=%s\n", qUtf8Printable(modelPath));
    if (!demPath.isEmpty()) std::fprintf(stdout, "dem=%s\n", qUtf8Printable(demPath));
    if (!domPath.isEmpty()) std::fprintf(stdout, "dom=%s\n", qUtf8Printable(domPath));
    std::fprintf(stdout, "report=%s\n", qUtf8Printable(finalReport.value(QStringLiteral("report_json")).toString()));

    return (modelOk && terrainOk) ? cli::EXIT_OK : cli::EXIT_ALGO_ERR;
}
