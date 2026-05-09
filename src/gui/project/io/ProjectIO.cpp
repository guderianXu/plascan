#include "ProjectIO.h"

#include <QDir>
#include <QFileInfo>

QString ProjectIO::projectRootFromPlascan(const QString &plascanPath)
{
    if (plascanPath.trimmed().isEmpty()) return QString();
    return QFileInfo(plascanPath).absolutePath();
}

QString ProjectIO::projectAssetsDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral("assets"));
}

QString ProjectIO::projectImagesDir(const QString &plascanPath)
{
    const QString assets = projectAssetsDir(plascanPath);
    if (assets.isEmpty()) return QString();
    return QDir(assets).filePath(QStringLiteral("images"));
}

QString ProjectIO::tmpDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral(".plascan_tmp"));
}

QString ProjectIO::tempFilesPath(const QString &plascanPath)
{
    const QString tmp = tmpDir(plascanPath);
    if (tmp.isEmpty()) return QString();
    return QDir(tmp).filePath(QStringLiteral("project_files.json"));
}

QString ProjectIO::tempConfigPath(const QString &plascanPath)
{
    const QString tmp = tmpDir(plascanPath);
    if (tmp.isEmpty()) return QString();
    return QDir(tmp).filePath(QStringLiteral("project_config.json"));
}

QString ProjectIO::tempResultsPath(const QString &plascanPath)
{
    const QString tmp = tmpDir(plascanPath);
    if (tmp.isEmpty()) return QString();
    return QDir(tmp).filePath(QStringLiteral("project_results.json"));
}

QString ProjectIO::ipfindOutputDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral("assets/ip"));
}

QString ProjectIO::ipmatchOutputDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral("assets/matches"));
}

QString ProjectIO::tmpIpfindDir(const QString &plascanPath)
{
    const QString root = projectRootFromPlascan(plascanPath);
    if (root.isEmpty()) return QString();
    return QDir(root).filePath(QStringLiteral(".plascan_tmp/ip"));
}

QString ProjectIO::vwipOutputPathForImage(const QString &plascanPath, const QString &imagePath)
{
    QFileInfo fi(imagePath);
    if (fi.fileName().isEmpty()) return QString();

    const QString outDir = ipfindOutputDir(plascanPath);
    if (outDir.isEmpty()) return QString();

    return QDir(outDir).filePath(fi.completeBaseName() + QStringLiteral(".sp"));
}

QString ProjectIO::featureOutputPathForImage(const QString &plascanPath,
                                              const QString &imagePath,
                                              const QString &suffix)
{
    QFileInfo fi(imagePath);
    if (fi.fileName().isEmpty()) return QString();

    const QString outDir = ipfindOutputDir(plascanPath);
    if (outDir.isEmpty()) return QString();

    return QDir(outDir).filePath(fi.completeBaseName() + suffix);
}

QStringList ProjectIO::spCandidates(const QString &plascanPath, const QString &imagePath)
{
    QStringList candidates;

    QFileInfo fi(imagePath);
    if (fi.fileName().isEmpty()) return candidates;
    const QString baseName = fi.completeBaseName() + QStringLiteral(".sp");

    const QString tmpDir = tmpIpfindDir(plascanPath);
    if (!tmpDir.isEmpty()) {
        candidates << QDir(tmpDir).filePath(baseName);
    }

    // 标准目录：assets/ip
    const QString ipDir = QDir(projectRootFromPlascan(plascanPath)).filePath(QStringLiteral("assets/ip"));
    if (!ipDir.isEmpty()) {
        candidates << QDir(ipDir).filePath(baseName);
    }

    candidates << fi.absoluteDir().filePath(baseName);
    return candidates;
}

QString ProjectIO::findSpForImage(const QString &plascanPath, const QString &imagePath)
{
    const QStringList candidates = spCandidates(plascanPath, imagePath);
    for (const QString &path : candidates) {
        if (QFileInfo::exists(path)) {
            return path;
        }
    }
    return QString();
}

QString ProjectIO::findFeatureForImage(const QString &plascanPath, const QString &imagePath)
{
    // 生成 .sp 候选路径，然后用所有后缀替换
    const QStringList spPaths = spCandidates(plascanPath, imagePath);
    static const char *suffixes[] = {".sp", ".dsk", ".alk", ".sift", ".orb", ".akz", ".dedode"};
    for (const QString &spPath : spPaths)
    {
        QString base = spPath;
        if (base.endsWith(".sp"))
            base.chop(3);
        for (const char *suf : suffixes)
        {
            QString candidate = base + suf;
            if (QFileInfo::exists(candidate))
                return candidate;
        }
    }
    return QString();
}

QString ProjectIO::featureFileForSuffix(const QString &plascanPath, const QString &imagePath,
                                         const QString &suffix)
{
    const QStringList candidates = spCandidates(plascanPath, imagePath);
    for (const QString &spPath : candidates)
    {
        QString base = spPath;
        if (base.endsWith(QStringLiteral(".sp")))
            base.chop(3);
        const QString path = base + suffix;
        if (QFileInfo::exists(path))
            return path;
    }
    return QString();
}

QStringList ProjectIO::availableFeatureSuffixes(const QString &plascanPath,
                                                  const QString &imagePath)
{
    QStringList result;
    const QStringList spPaths = spCandidates(plascanPath, imagePath);
    static const char *suffixes[] = {".sp", ".dsk", ".alk", ".sift", ".orb", ".akz", ".dedode"};

    for (const QString &spPath : spPaths)
    {
        QString base = spPath;
        if (base.endsWith(".sp"))
            base.chop(3);
        for (const char *suf : suffixes)
        {
            if (QFileInfo::exists(base + suf) && !result.contains(suf))
                result.append(suf);
        }
    }
    return result;
}
