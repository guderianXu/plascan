// =============================================================================
// 文件: MatcherFactory.cpp
// 功能: 匹配器工厂 (C++ 适配器 + Python 子进程)
// =============================================================================
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267)
#endif

#include "MatcherFactory.h"
#include "SuperGlueMatcher.h"
#include "LightGlueMatcher.h"
#include "LoFTRMatcher.h"
#include "TraditionalFeatureMatcher.h"
#include "FeatureOutput.h"
#include "FeatureData.h"
#include "FeatureFileIO.h"
#include "MatchFileIO.h"

#include <QProcess>
#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFileInfo>
#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

namespace
{

// 加载两个特征文件并返回 FeatureData
QJsonArray makePointArray(float x, float y)
{
    QJsonArray point;
    point.append(static_cast<double>(x));
    point.append(static_cast<double>(y));
    return point;
}

static bool loadPair(const std::string &sp1,
                     const std::string &sp2,
                     xjw::feature_extractors::FeatureData &fd0,
                     xjw::feature_extractors::FeatureData &fd1,
                     QString *imageName0 = nullptr,
                     QString *imageName1 = nullptr)
{
    QString n1, n2;
    if (!FeatureFileIO::readData(QString::fromStdString(sp1), n1, fd0))
    {
        return false;
    }
    if (!FeatureFileIO::readData(QString::fromStdString(sp2), n2, fd1))
    {
        return false;
    }
    if (imageName0)
    {
        *imageName0 = n1;
    }
    if (imageName1)
    {
        *imageName1 = n2;
    }
    return true;
}

void writeFeatureSidecar(const QString &outPath,
                         const QString &feature0Path,
                         const QString &feature1Path,
                         const QString &image0Name,
                         const QString &image1Name,
                         const xjw::feature_extractors::FeatureData &fd0,
                         const xjw::feature_extractors::FeatureData &fd1,
                         const xjw::feature_match::MatchResult &result,
                         const QString &matchAlgorithm,
                         float matchThreshold)
{
    QJsonArray points0;
    QJsonArray points1;
    QJsonArray indices0;
    QJsonArray indices1;
    QJsonArray scores;

    for (size_t i = 0; i < result.matches0.size(); ++i)
    {
        const int idx1 = result.matches0[i];
        if (idx1 < 0 ||
            i >= fd0.keypoints.size() ||
            idx1 >= static_cast<int>(fd1.keypoints.size()))
        {
            continue;
        }

        indices0.append(static_cast<int>(i));
        indices1.append(idx1);
        points0.append(makePointArray(fd0.keypoints[i].pt.x, fd0.keypoints[i].pt.y));
        points1.append(makePointArray(fd1.keypoints[static_cast<size_t>(idx1)].pt.x,
                                      fd1.keypoints[static_cast<size_t>(idx1)].pt.y));
        const float score = i < result.matchingScores0.size()
            ? result.matchingScores0[i]
            : 1.0f;
        scores.append(static_cast<double>(score));
    }

    QJsonObject sidecar;
    sidecar[QStringLiteral("match_file")] = outPath;
    sidecar[QStringLiteral("image0_name")] = image0Name;
    sidecar[QStringLiteral("image1_name")] = image1Name;
    sidecar[QStringLiteral("feature0_path")] = feature0Path;
    sidecar[QStringLiteral("feature1_path")] = feature1Path;
    sidecar[QStringLiteral("sp0_path")] = feature0Path;
    sidecar[QStringLiteral("sp1_path")] = feature1Path;
    sidecar[QStringLiteral("feature_algorithm")] = QString::fromStdString(fd0.sourceAlgorithm);
    sidecar[QStringLiteral("match_algorithm")] = matchAlgorithm;
    sidecar[QStringLiteral("feature_format_version")] = 2;
    sidecar[QStringLiteral("num_matches")] = indices0.size();
    sidecar[QStringLiteral("match_threshold")] = static_cast<double>(matchThreshold);
    sidecar[QStringLiteral("matched_points0")] = points0;
    sidecar[QStringLiteral("matched_points1")] = points1;
    sidecar[QStringLiteral("matched_indices0")] = indices0;
    sidecar[QStringLiteral("matched_indices1")] = indices1;
    sidecar[QStringLiteral("matched_scores")] = scores;

    QFile file(outPath + QStringLiteral(".json"));
    if (file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        file.write(QJsonDocument(sidecar).toJson(QJsonDocument::Compact));
    }
}

xjw::feature_match::MatchResult makeIndexedResultFromPoints(const std::vector<cv::Point2f> &points0,
                                                            const std::vector<cv::Point2f> &points1,
                                                            const std::vector<float> &confidences,
                                                            const std::string &algorithm)
{
    const int count = static_cast<int>(std::min(points0.size(), points1.size()));
    xjw::feature_match::MatchResult result;
    result.matches0.assign(static_cast<size_t>(count), -1);
    result.matches1.assign(static_cast<size_t>(count), -1);
    result.matchingScores0.assign(static_cast<size_t>(count), 0.0f);
    result.matchingScores1.assign(static_cast<size_t>(count), 0.0f);
    result.sourceAlgorithm = algorithm;

    for (int i = 0; i < count; ++i)
    {
        const float score = i < static_cast<int>(confidences.size()) ? confidences[static_cast<size_t>(i)] : 1.0f;
        result.matches0[static_cast<size_t>(i)] = i;
        result.matches1[static_cast<size_t>(i)] = i;
        result.matchingScores0[static_cast<size_t>(i)] = score;
        result.matchingScores1[static_cast<size_t>(i)] = score;

        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = i;
        match.distance = 1.0f - score;
        result.cvMatches.push_back(match);
    }
    result.numMatches = static_cast<int>(result.cvMatches.size());
    return result;
}

int writeEndToEndMatchOutputs(const QString &algorithm,
                              const QString &image0Path,
                              const QString &image1Path,
                              const QString &outPath,
                              const std::vector<cv::Point2f> &points0,
                              const std::vector<cv::Point2f> &points1,
                              const std::vector<float> &confidences)
{
    const xjw::feature_match::MatchResult result =
        makeIndexedResultFromPoints(points0, points1, confidences, algorithm.toStdString());
    const QString image0Name = QFileInfo(image0Path).completeBaseName();
    const QString image1Name = QFileInfo(image1Path).completeBaseName();
    if (!xjw::feature_match::writeIndexedMatchFile(outPath, image0Name, image1Name, result))
    {
        return -1;
    }

    QJsonArray jsonPoints0;
    QJsonArray jsonPoints1;
    QJsonArray indices0;
    QJsonArray indices1;
    QJsonArray scores;
    for (int i = 0; i < result.numMatches; ++i)
    {
        jsonPoints0.append(makePointArray(points0[static_cast<size_t>(i)].x,
                                          points0[static_cast<size_t>(i)].y));
        jsonPoints1.append(makePointArray(points1[static_cast<size_t>(i)].x,
                                          points1[static_cast<size_t>(i)].y));
        indices0.append(i);
        indices1.append(i);
        scores.append(static_cast<double>(
            i < static_cast<int>(confidences.size()) ? confidences[static_cast<size_t>(i)] : 1.0f));
    }

    QJsonObject sidecar;
    sidecar[QStringLiteral("match_file")] = outPath;
    sidecar[QStringLiteral("image0_path")] = image0Path;
    sidecar[QStringLiteral("image1_path")] = image1Path;
    sidecar[QStringLiteral("image0_name")] = image0Name;
    sidecar[QStringLiteral("image1_name")] = image1Name;
    sidecar[QStringLiteral("feature_algorithm")] = algorithm;
    sidecar[QStringLiteral("match_algorithm")] = algorithm;
    sidecar[QStringLiteral("backend")] = algorithm == QStringLiteral("loftr")
        ? QStringLiteral("cpp_torchscript")
        : QStringLiteral("python");
    sidecar[QStringLiteral("feature_format_version")] = 2;
    sidecar[QStringLiteral("num_matches")] = result.numMatches;
    sidecar[QStringLiteral("matched_points0")] = jsonPoints0;
    sidecar[QStringLiteral("matched_points1")] = jsonPoints1;
    sidecar[QStringLiteral("matched_indices0")] = indices0;
    sidecar[QStringLiteral("matched_indices1")] = indices1;
    sidecar[QStringLiteral("matched_scores")] = scores;

    QFile file(outPath + QStringLiteral(".json"));
    if (file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        file.write(QJsonDocument(sidecar).toJson(QJsonDocument::Compact));
    }
    return result.numMatches;
}

QString resolvedExecutablePath(const QString &candidate)
{
    const QString trimmed = candidate.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }

    if (trimmed.contains(QLatin1Char('/')) || trimmed.contains(QLatin1Char('\\')))
    {
        const QFileInfo info(trimmed);
        if (info.exists() && info.isFile() && info.isExecutable())
        {
            return QDir::cleanPath(info.absoluteFilePath());
        }
        return QString();
    }

    const QString resolved = QStandardPaths::findExecutable(trimmed);
    return resolved.isEmpty() ? QString() : QDir::cleanPath(QFileInfo(resolved).absoluteFilePath());
}

QString pythonExecutable()
{
    static const QString cached = []()
    {
        QStringList candidates;
        candidates << qEnvironmentVariable("PLASCAN_PYTHON_EXECUTABLE").trimmed()
                   << qEnvironmentVariable("PLASCAN_PYTHON").trimmed()
                   << qEnvironmentVariable("PYTHON").trimmed()
                   << QStringLiteral("python3")
                   << QStringLiteral("python");
        candidates.removeAll(QString());
        candidates.removeDuplicates();

        for (const QString &candidate : candidates)
        {
            const QString resolved = resolvedExecutablePath(candidate);
            if (!resolved.isEmpty())
            {
                return resolved;
            }
        }
        return QStringLiteral("python");
    }();
    return cached;
}

int readMatchCountFromOutputs(const QString &outPath)
{
    QFile sidecarFile(outPath + QStringLiteral(".json"));
    if (sidecarFile.open(QIODevice::ReadOnly))
    {
        const QJsonObject sidecar = QJsonDocument::fromJson(sidecarFile.readAll()).object();
        const int count = sidecar.value(QStringLiteral("num_matches")).toInt(-1);
        if (count >= 0)
        {
            return count;
        }
    }

    QString image0Name;
    QString image1Name;
    xjw::feature_match::MatchResult result;
    if (xjw::feature_match::readIndexedMatchFile(outPath, image0Name, image1Name, result))
    {
        return result.numMatches;
    }
    return -1;
}

QString findScriptFile(const QString &scriptName)
{
    QStringList candidates;

    const QString envScriptDir = qEnvironmentVariable("PLASCAN_SCRIPT_DIR").trimmed();
    if (!envScriptDir.isEmpty())
    {
        candidates.append(QDir(envScriptDir).filePath(scriptName));
    }

#ifdef PLASCAN_SOURCE_DIR
    candidates.append(
        QDir(QStringLiteral(PLASCAN_SOURCE_DIR)).filePath(QStringLiteral("scripts/%1").arg(scriptName)));
#endif

    const QString exeDir = QCoreApplication::applicationDirPath();
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../scripts/%1").arg(scriptName)));
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../../scripts/%1").arg(scriptName)));
    candidates.append(QDir(exeDir).filePath(QStringLiteral("../../../scripts/%1").arg(scriptName)));
    candidates.append(QDir(QDir::currentPath()).filePath(QStringLiteral("scripts/%1").arg(scriptName)));

    for (const QString &candidate : candidates)
    {
        if (QFileInfo::exists(candidate))
        {
            return QDir::cleanPath(QFileInfo(candidate).absoluteFilePath());
        }
    }

    return QString();
}

// ── SuperGlue ──
class SuperGlueAdapter : public IMatcher
{
public:
    explicit SuperGlueAdapter(const MatcherConfig &cfg)
        : _config(cfg)
    {
    }

    int match(const std::string &sp1, const std::string &sp2,
              const std::string &, const std::string &,
              const std::string &outPath) override
    {
        superglue::SuperGlueConfig sgCfg;
        sgCfg.model_path      = _config.modelPath;
        sgCfg.match_threshold = _config.matchThreshold;
        sgCfg.max_keypoints   = _config.maxKeypoints;
        sgCfg.use_cuda        = _config.useCuda;
        sgCfg.cuda_device_id  = _config.cudaDevice;

        superglue::SuperGlueMatcher matcher(sgCfg);

        xjw::feature_extractors::FeatureData fd0, fd1;
        QString imageName0, imageName1;
        if (!loadPair(sp1, sp2, fd0, fd1, &imageName0, &imageName1))
        {
            return -1;
        }

        auto mr = matcher.match(fd0, fd1);
        if (mr.numMatches == 0)
        {
            return 0;
        }
        const QString qOutPath = QString::fromStdString(outPath);
        if (!xjw::feature_match::writeIndexedMatchFile(qOutPath, imageName0, imageName1, mr))
        {
            return -1;
        }
        writeFeatureSidecar(qOutPath,
                            QString::fromStdString(sp1),
                            QString::fromStdString(sp2),
                            imageName0,
                            imageName1,
                            fd0,
                            fd1,
                            mr,
                            QStringLiteral("superglue"),
                            _config.matchThreshold);
        return mr.numMatches;
    }

    std::string algorithmName() const override
    {
        return "superglue";
    }

private:
    MatcherConfig _config;
};

// ── LightGlue ──
class LightGlueAdapter : public IMatcher
{
public:
    explicit LightGlueAdapter(const MatcherConfig &cfg)
        : _config(cfg)
    {
    }

    int match(const std::string &sp1, const std::string &sp2,
              const std::string &, const std::string &,
              const std::string &outPath) override
    {
        xjw::feature_match::LightGlueConfig lgCfg;
        lgCfg.matcherModelPath = _config.modelPath;
        lgCfg.spModelPath      = _config.spModelPath;
        lgCfg.useCuda          = _config.useCuda;
        lgCfg.cudaDevice       = _config.cudaDevice;
        lgCfg.scoreThreshold   = _config.matchThreshold;

        xjw::feature_match::LightGlueMatcher matcher(lgCfg);

        xjw::feature_extractors::FeatureData fd0, fd1;
        QString imageName0, imageName1;
        if (!loadPair(sp1, sp2, fd0, fd1, &imageName0, &imageName1))
        {
            return -1;
        }

        auto mr = matcher.match(fd0, fd1);
        if (mr.numMatches == 0)
        {
            return 0;
        }
        const QString qOutPath = QString::fromStdString(outPath);
        if (!xjw::feature_match::writeIndexedMatchFile(qOutPath, imageName0, imageName1, mr))
        {
            return -1;
        }
        writeFeatureSidecar(qOutPath,
                            QString::fromStdString(sp1),
                            QString::fromStdString(sp2),
                            imageName0,
                            imageName1,
                            fd0,
                            fd1,
                            mr,
                            QStringLiteral("lightglue"),
                            _config.matchThreshold);
        return mr.numMatches;
    }

    std::string algorithmName() const override
    {
        return "lightglue";
    }

private:
    MatcherConfig _config;
};

// ── LoFTR C++ TorchScript 端到端匹配 ──
class LoFTRAdapter : public IMatcher
{
public:
    explicit LoFTRAdapter(const MatcherConfig &cfg)
        : _config(cfg)
    {
    }

    int match(const std::string &, const std::string &,
              const std::string &imgL, const std::string &imgR,
              const std::string &outPath) override
    {
        if (_config.modelPath.empty())
        {
            fprintf(stderr, "LoFTR C++ matcher requires --model pointing to a TorchScript model\n");
            return -1;
        }
        if (imgL.empty() || imgR.empty())
        {
            fprintf(stderr, "LoFTR matcher requires -L/--left and -R/--right images\n");
            return -1;
        }

        const cv::Mat left = cv::imread(imgL, cv::IMREAD_GRAYSCALE);
        const cv::Mat right = cv::imread(imgR, cv::IMREAD_GRAYSCALE);
        if (left.empty() || right.empty())
        {
            fprintf(stderr, "LoFTR matcher cannot read input images\n");
            return -1;
        }

        xjw::feature_match::LoFTRConfig cfg;
        cfg.modelPath = _config.modelPath;
        cfg.useCuda = _config.useCuda;
        cfg.matchThreshold = _config.matchThreshold;
        cfg.maxImageDim = _config.maxImageDim;

        xjw::feature_match::LoFTRMatcher matcher(cfg);
        const xjw::feature_match::LoFTRResult result = matcher.match(left, right);
        return writeEndToEndMatchOutputs(QStringLiteral("loftr"),
                                         QString::fromStdString(imgL),
                                         QString::fromStdString(imgR),
                                         QString::fromStdString(outPath),
                                         result.pts0,
                                         result.pts1,
                                         result.confidences);
    }

    std::string algorithmName() const override
    {
        return "loftr";
    }

    bool needsFeatureFiles() const override
    {
        return false;
    }

private:
    MatcherConfig _config;
};

// ── Python 子进程 (RoMa/DeDoDe) ──
class PythonAdapter : public IMatcher
{
public:
    PythonAdapter(const std::string &algo, const MatcherConfig &cfg)
        : _algorithm(algo), _config(cfg)
    {
    }

    int match(const std::string &sp1, const std::string &sp2,
              const std::string &imgL, const std::string &imgR,
              const std::string &outPath) override
    {
        QString scriptName;
        if (_algorithm == "roma")
        {
            scriptName = QStringLiteral("match_roma.py");
        }
        else if (_algorithm == "dedode")
        {
            scriptName = QStringLiteral("run_dedode.py");
        }
        else
        {
            fprintf(stderr, "unsupported Python matcher: %s\n", _algorithm.c_str());
            return -1;
        }

        const QString script = findScriptFile(scriptName);
        if (script.isEmpty())
        {
            fprintf(stderr, "Python matcher script not found for %s\n", _algorithm.c_str());
            return -1;
        }

        QStringList args;
        args << script;
        if (_algorithm == "dedode" && !sp1.empty() && !sp2.empty())
        {
            args << "--feature-left" << QString::fromStdString(sp1)
                 << "--feature-right" << QString::fromStdString(sp2);
        }
        else
        {
            if (imgL.empty() || imgR.empty())
            {
                fprintf(stderr, "%s matcher requires image paths\n", _algorithm.c_str());
                return -1;
            }
            args << "-L" << QString::fromStdString(imgL)
                 << "-R" << QString::fromStdString(imgR);
        }
        args << "-o" << QString::fromStdString(outPath);

        if (_algorithm == "roma")
        {
            args << "--threshold" << QString::number(_config.matchThreshold)
                 << "--max-keypoints" << QString::number(_config.maxKeypoints);
        }
        else if (_algorithm == "dedode")
        {
            args << "--min-score" << QString::number(_config.matchThreshold)
                 << "--max-kp" << QString::number(_config.maxKeypoints);
        }
        if (_config.useCuda)
        {
            args << "--cuda";
        }

        QProcess proc;
        proc.start(pythonExecutable(), args);
        if (!proc.waitForFinished(600000))
        {
            proc.kill();
            return -1;
        }
        fprintf(stdout, "%s", proc.readAllStandardOutput().constData());
        if (proc.exitCode() != 0)
        {
            fprintf(stderr, "%s", proc.readAllStandardError().constData());
            return -1;
        }
        return readMatchCountFromOutputs(QString::fromStdString(outPath));
    }

    std::string algorithmName() const override
    {
        return _algorithm;
    }

    bool needsFeatureFiles() const override
    {
        return _algorithm == "dedode";
    }

private:
    std::string _algorithm;
    MatcherConfig _config;
};

} // anonymous namespace

std::unique_ptr<IMatcher> createMatcher(const std::string &algo,
                                         const MatcherConfig &cfg)
{
    if (algo == "superglue")
    {
        return std::make_unique<SuperGlueAdapter>(cfg);
    }
    if (algo == "lightglue")
    {
        return std::make_unique<LightGlueAdapter>(cfg);
    }
    if (algo == "loftr")
    {
        return std::make_unique<LoFTRAdapter>(cfg);
    }
    if (algo == "roma" || algo == "dedode")
    {
        return std::make_unique<PythonAdapter>(algo, cfg);
    }
    throw std::runtime_error("unsupported matcher: " + algo);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
