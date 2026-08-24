#include "cli_common.h"
#include "RpcDomGenerator.h"
#include "RpcStereoDemGenerator.h"

#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QTextStream>

namespace
{

    void printProgress(const QString& message, int percent)
    {
        QTextStream(stderr) << '[' << percent << "%] " << message << Qt::endl;
    }

} // namespace

int main(int argc, char** argv)
{
    QCoreApplication application(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("rpc_stereo_products_cli"));
#ifdef PLASCAN_VERSION
    QCoreApplication::setApplicationVersion(QStringLiteral(PLASCAN_VERSION));
#endif

    QCommandLineParser parser;
    parser.setApplicationDescription(QStringLiteral("从带 RPC 的 TIFF 立体像对生成地理参考 DEM 和 DOM"));
    parser.addHelpOption();
    parser.addVersionOption();
    const QCommandLineOption leftOption(
        QStringList{QStringLiteral("left")}, QStringLiteral("左影像 TIFF"), QStringLiteral("path"));
    const QCommandLineOption rightOption(
        QStringList{QStringLiteral("right")}, QStringLiteral("右影像 TIFF"), QStringLiteral("path"));
    const QCommandLineOption outputOption(
        QStringList{QStringLiteral("output")}, QStringLiteral("输出目录"), QStringLiteral("directory"));
    const QCommandLineOption resolutionOption(QStringList{QStringLiteral("resolution")},
                                              QStringLiteral("DEM 网格分辨率（米）"),
                                              QStringLiteral("meters"),
                                              QStringLiteral("2.0"));
    const QCommandLineOption featureOption(QStringList{QStringLiteral("max-features")},
                                           QStringLiteral("每张影像最多提取的 SIFT 特征数"),
                                           QStringLiteral("count"),
                                           QStringLiteral("20000"));
    const QCommandLineOption demOnlyOption(QStringList{QStringLiteral("dem-only")},
                                           QStringLiteral("只生成 DEM，不生成 DOM"));
    parser.addOptions({leftOption, rightOption, outputOption, resolutionOption, featureOption, demOnlyOption});
    parser.process(application);

    if (!parser.isSet(leftOption) || !parser.isSet(rightOption) || !parser.isSet(outputOption))
    {
        QTextStream(stderr) << "必须提供 --left、--right 和 --output" << Qt::endl;
        parser.showHelp(cli::EXIT_ARG_ERR);
    }

    bool resolutionOk = false;
    bool featureCountOk = false;
    const double resolution = parser.value(resolutionOption).toDouble(&resolutionOk);
    const int featureCount = parser.value(featureOption).toInt(&featureCountOk);
    if (!resolutionOk || resolution <= 0.0 || !featureCountOk || featureCount < 100)
    {
        QTextStream(stderr) << "--resolution 或 --max-features 参数无效" << Qt::endl;
        return cli::EXIT_ARG_ERR;
    }

    const QString leftPath = QFileInfo(parser.value(leftOption)).absoluteFilePath();
    const QString rightPath = QFileInfo(parser.value(rightOption)).absoluteFilePath();
    const QString outputDirectory = QDir(parser.value(outputOption)).absolutePath();
    xjw::RpcStereoDemOptions demOptions;
    demOptions.gridResolutionMeters = resolution;
    demOptions.maximumFeatures = featureCount;

    QJsonObject demResult;
    QString error;
    if (!xjw::RpcStereoDemGenerator::generate(
            leftPath, rightPath, outputDirectory, demOptions, &demResult, &error, printProgress))
    {
        QTextStream(stderr) << "DEM 生成失败: " << error << Qt::endl;
        return cli::EXIT_ALGO_ERR;
    }

    QJsonObject finalResult{{QStringLiteral("dem"), demResult}};
    if (!parser.isSet(demOnlyOption))
    {
        const QString demPath = demResult.value(QStringLiteral("dem_path")).toString();
        const QString domPath = QDir(outputDirectory).filePath(QStringLiteral("dom.tif"));
        QJsonObject domResult;
        if (!xjw::RpcDomGenerator::generate(
                {leftPath, rightPath}, demPath, domPath, xjw::RpcDomOptions{}, &domResult, &error, printProgress))
        {
            QTextStream(stderr) << "DOM 生成失败: " << error << Qt::endl;
            return cli::EXIT_ALGO_ERR;
        }
        finalResult.insert(QStringLiteral("dom"), domResult);
    }

    QTextStream(stdout) << QJsonDocument(finalResult).toJson(QJsonDocument::Indented);
    return cli::EXIT_OK;
}
