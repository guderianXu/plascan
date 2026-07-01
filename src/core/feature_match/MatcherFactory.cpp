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
#include "TraditionalFeatureMatcher.h"
#include "FeatureOutput.h"
#include "FeatureFileIO.h"
#include "FeatureData.h"

#include <QProcess>
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QDataStream>
#include <stdexcept>
#include <cstdio>

namespace
{

// 保存 .match 文件, 返回匹配点数
static int saveMatch(const std::string &path,
                     const xjw::feature_match::MatchResult &mr,
                     const std::vector<cv::KeyPoint> &kp0,
                     const std::vector<cv::KeyPoint> &kp1)
{
    QFile f(QString::fromStdString(path));
    if (!f.open(QIODevice::WriteOnly))
    {
        return -1;
    }
    QDataStream ds(&f);
    ds.setByteOrder(QDataStream::BigEndian);
    qint32 count = 0;
    for (size_t i = 0; i < mr.matches0.size(); ++i)
    {
        if (mr.matches0[i] >= 0)
        {
            ++count;
        }
    }
    ds << count;
    for (size_t i = 0; i < mr.matches0.size(); ++i)
    {
        int m1 = mr.matches0[i];
        if (m1 < 0)
        {
            continue;
        }
        ds << kp0[i].pt.x << kp0[i].pt.y << kp1[m1].pt.x << kp1[m1].pt.y;
    }
    f.close();
    return count;
}

// 加载两个特征文件并返回 FeatureData
static bool loadPair(const std::string &sp1, const std::string &sp2,
                     xjw::feature_extractors::FeatureData &fd0,
                     xjw::feature_extractors::FeatureData &fd1)
{
    FeatureOutput spo1, spo2;
    QString n1, n2;
    if (!FeatureFileIO::read(QString::fromStdString(sp1), n1, spo1))
    {
        return false;
    }
    if (!FeatureFileIO::read(QString::fromStdString(sp2), n2, spo2))
    {
        return false;
    }
    fd0 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo1);
    fd1 = xjw::feature_extractors::FeatureData::fromFeatureOutput(spo2);
    return true;
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
        if (!loadPair(sp1, sp2, fd0, fd1))
        {
            return -1;
        }

        auto mr = matcher.match(fd0, fd1);
        if (mr.numMatches == 0)
        {
            return 0;
        }
        return saveMatch(outPath, mr, fd0.keypoints, fd1.keypoints);
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
        lgCfg.useCuda          = _config.useCuda;
        lgCfg.scoreThreshold   = _config.matchThreshold;

        xjw::feature_match::LightGlueMatcher matcher(lgCfg);

        xjw::feature_extractors::FeatureData fd0, fd1;
        if (!loadPair(sp1, sp2, fd0, fd1))
        {
            return -1;
        }

        auto mr = matcher.match(fd0, fd1);
        if (mr.numMatches == 0)
        {
            return 0;
        }
        return saveMatch(outPath, mr, fd0.keypoints, fd1.keypoints);
    }

    std::string algorithmName() const override
    {
        return "lightglue";
    }

private:
    MatcherConfig _config;
};

// ── Python 子进程 (LoFTR/DISK/ALIKED) ──
class PythonAdapter : public IMatcher
{
public:
    PythonAdapter(const std::string &algo, const MatcherConfig &cfg)
        : _algorithm(algo), _config(cfg)
    {
    }

    int match(const std::string &, const std::string &,
              const std::string &imgL, const std::string &imgR,
              const std::string &outPath) override
    {
        QString script = findScriptFile(QStringLiteral("run_%1.py").arg(QString::fromStdString(_algorithm)));
        if (_algorithm == "disk" || _algorithm == "aliked")
        {
            script = findScriptFile(QStringLiteral("run_disk_aliked.py"));
        }
        if (script.isEmpty())
        {
            fprintf(stderr, "Python matcher script not found for %s\n", _algorithm.c_str());
            return -1;
        }

        QStringList args;
        args << script
             << "-L" << QString::fromStdString(imgL)
             << "-R" << QString::fromStdString(imgR)
             << "-o" << QString::fromStdString(outPath);
        if (_algorithm == "disk" || _algorithm == "aliked")
        {
            args << "-a" << QString::fromStdString(_algorithm);
        }
        if (_config.useCuda)
        {
            args << "--cuda";
        }

        QProcess proc;
        proc.start("python3", args);
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
        return 1;
    }

    std::string algorithmName() const override
    {
        return _algorithm;
    }

    bool needsFeatureFiles() const override
    {
        return false;
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
    if (algo == "loftr" || algo == "disk" || algo == "aliked")
    {
        return std::make_unique<PythonAdapter>(algo, cfg);
    }
    throw std::runtime_error("unsupported matcher: " + algo);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
