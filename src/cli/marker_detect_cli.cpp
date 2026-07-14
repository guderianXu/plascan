#include "cli_common.h"

#include "detection/MarkerDetectorFactory.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace
{

QString fromCliPath(const std::string &path)
{
    return QDir::cleanPath(QString::fromUtf8(path.data(), static_cast<qsizetype>(path.size())));
}

QImage readImage(const QString &path)
{
    QImageReader reader(path);
    reader.setAutoTransform(false);
    return reader.read();
}

QString fileSha256(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return QString();
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
    {
        hash.addData(file.read(1024 * 1024));
    }
    return QStringLiteral("sha256:%1").arg(QString::fromLatin1(hash.result().toHex()));
}

QString fallbackImageId(const QString &path)
{
    const QByteArray digest = QCryptographicHash::hash(
        QFileInfo(path).absoluteFilePath().toUtf8(), QCryptographicHash::Sha256).toHex();
    return QStringLiteral("cli-%1").arg(QString::fromLatin1(digest.left(20)));
}

QJsonArray point(const QPointF &value)
{
    return QJsonArray{value.x(), value.y()};
}

QJsonObject detectionJson(const QString &imageId,
                          const QString &imagePath,
                          const QString &signature,
                          const xjw::control_points::MarkerDetection &detection)
{
    QJsonArray corners;
    for (const QPointF &corner : detection.corners)
    {
        corners.push_back(point(corner));
    }
    return {
        {QStringLiteral("image_id"), imageId},
        {QStringLiteral("image_path"), imagePath},
        {QStringLiteral("image_content_signature"), signature},
        {QStringLiteral("family"), xjw::control_points::markerTargetFamilyName(detection.family)},
        {QStringLiteral("target_id"), detection.targetId},
        {QStringLiteral("center"), point(detection.center)},
        {QStringLiteral("corners"), corners},
        {QStringLiteral("confidence"), detection.confidence},
        {QStringLiteral("center_sigma_px"), detection.centerSigmaPx},
        {QStringLiteral("decision_margin"), detection.decisionMargin},
        {QStringLiteral("hamming"), detection.hamming},
        {QStringLiteral("size_px"), detection.sizePx},
        {QStringLiteral("rotation_degrees"), detection.rotationDegrees},
        {QStringLiteral("source"), detection.source},
    };
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication application(argc, argv);
    CLI::App app{"PlaScan 编码/非编码标靶检测工具"};

    std::vector<std::string> imagePaths;
    std::vector<std::string> maskPaths;
    std::vector<std::string> imageIds;
    std::string familyName;
    std::string outputPath;
    double minimumDecisionMargin = 20.0;
    int maximumHamming = 1;
    int threads = 0;
    double quadDecimate = 1.0;

    app.add_option("--image", imagePaths, "输入影像，可重复指定")->required();
    app.add_option("--mask", maskPaths, "与 --image 按顺序对应的蒙版，非零像素表示排除");
    app.add_option("--image-id", imageIds, "与 --image 对应的稳定影像 ID；省略时按绝对路径生成");
    app.add_option("--family", familyName, "标靶族，例如 tag36h11 或 noncoded-circle")->required();
    app.add_option("--output", outputPath, "输出检测观测 JSON")->required();
    app.add_option("--min-decision-margin", minimumDecisionMargin, "AprilTag 最小判决裕量");
    app.add_option("--max-hamming", maximumHamming, "AprilTag 最大纠错位数 (0-2)");
    app.add_option("--threads", threads, "单影像检测线程数，0 表示自动");
    app.add_option("--quad-decimate", quadDecimate, "AprilTag 四边形检测降采样系数");
    CLI11_PARSE(app, argc, argv);

    if (!maskPaths.empty() && maskPaths.size() != imagePaths.size())
    {
        cli::fatal("--mask 数量必须为 0 或与 --image 数量一致", cli::EXIT_ARG_ERR);
    }
    if (!imageIds.empty() && imageIds.size() != imagePaths.size())
    {
        cli::fatal("--image-id 数量必须为 0 或与 --image 数量一致", cli::EXIT_ARG_ERR);
    }

    const auto family = xjw::control_points::MarkerDetectorFactory::parseFamily(
        QString::fromUtf8(familyName.data(), static_cast<qsizetype>(familyName.size())));
    if (!family)
    {
        cli::fatal("不支持的标靶族: " + familyName, cli::EXIT_ARG_ERR);
    }

    std::unique_ptr<xjw::control_points::MarkerDetector> detector;
    try
    {
        detector = xjw::control_points::MarkerDetectorFactory::create(*family);
    }
    catch (const std::exception &exception)
    {
        cli::fatal(exception.what(), cli::EXIT_ALGO_ERR);
    }

    xjw::control_points::MarkerDetectionOptions options;
    options.minDecisionMargin = minimumDecisionMargin;
    options.maxHamming = maximumHamming;
    options.threadCount = threads;
    options.quadDecimate = quadDecimate;
    QJsonArray observations;

    for (std::size_t index = 0; index < imagePaths.size(); ++index)
    {
        const QString image_path = QFileInfo(fromCliPath(imagePaths[index])).absoluteFilePath();
        const QImage image = readImage(image_path);
        if (image.isNull())
        {
            cli::fatal("无法读取影像: " + imagePaths[index], cli::EXIT_IO_ERR);
        }

        QImage mask;
        if (!maskPaths.empty())
        {
            const QString mask_path = QFileInfo(fromCliPath(maskPaths[index])).absoluteFilePath();
            mask = readImage(mask_path);
            if (mask.isNull())
            {
                cli::fatal("无法读取蒙版: " + maskPaths[index], cli::EXIT_IO_ERR);
            }
        }

        QVector<xjw::control_points::MarkerDetection> detections;
        try
        {
            detections = detector->detect(image, mask, options);
        }
        catch (const std::exception &exception)
        {
            cli::fatal(exception.what(), cli::EXIT_ALGO_ERR);
        }
        std::sort(detections.begin(), detections.end(), [](const auto &left, const auto &right)
        {
            if (left.targetId != right.targetId) return left.targetId < right.targetId;
            if (left.center.y() != right.center.y()) return left.center.y() < right.center.y();
            return left.center.x() < right.center.x();
        });

        const QString image_id = imageIds.empty()
            ? fallbackImageId(image_path)
            : QString::fromUtf8(imageIds[index].data(), static_cast<qsizetype>(imageIds[index].size()));
        const QString signature = fileSha256(image_path);
        for (const auto &detection : detections)
        {
            observations.push_back(detectionJson(image_id, image_path, signature, detection));
        }
    }

    const QJsonObject document{
        {QStringLiteral("schema"), QStringLiteral("plascan.marker-detections.v1")},
        {QStringLiteral("family"), xjw::control_points::markerTargetFamilyName(*family)},
        {QStringLiteral("image_count"), static_cast<int>(imagePaths.size())},
        {QStringLiteral("observation_count"), observations.size()},
        {QStringLiteral("observations"), observations},
    };
    const QString output_path = QFileInfo(fromCliPath(outputPath)).absoluteFilePath();
    if (!QDir().mkpath(QFileInfo(output_path).absolutePath()))
    {
        cli::fatal("无法创建输出目录", cli::EXIT_IO_ERR);
    }
    QSaveFile output(output_path);
    if (!output.open(QIODevice::WriteOnly)
        || output.write(QJsonDocument(document).toJson(QJsonDocument::Indented)) < 0
        || !output.commit())
    {
        cli::fatal("无法写入检测结果", cli::EXIT_IO_ERR);
    }

    std::fprintf(stdout,
                 "检测完成: %zu 张影像, %lld 条观测, 输出 %s\n",
                 imagePaths.size(),
                 static_cast<long long>(observations.size()),
                 outputPath.c_str());
    return cli::EXIT_OK;
}
