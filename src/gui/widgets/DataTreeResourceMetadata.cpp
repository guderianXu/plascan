#include "DataTreeResourceUtils.h"

#include <QFileInfo>
#include <QLocale>
#include <QVector>

namespace xjw::gui::widgets::data_tree
{

const int SectionRole = Qt::UserRole + 1;
const int ResourcePathRole = Qt::UserRole + 2;
const int AggregateResourcePathsRole = Qt::UserRole + 3;
const int WorkspaceSectionRole = Qt::UserRole + 4;
const int ChunkIdRole = Qt::UserRole + 5;
const int ChunkDirectoryRole = Qt::UserRole + 6;
const int WorkspaceRootRole = Qt::UserRole + 7;

QString workspaceSummaryLabel(int chunkCount, int imageCount)
{
    return QStringLiteral("工作区 (%1个块, %2个图像)")
        .arg(QLocale().toString(chunkCount),
             QLocale().toString(imageCount));
}

QString chunkSummaryLabel(const QString &name,
                          int imageCount,
                          int tiePointCount)
{
    QStringList summary;
    if (imageCount >= 0)
    {
        summary.append(QStringLiteral("%1个图像")
                           .arg(QLocale().toString(imageCount)));
    }
    if (tiePointCount >= 0)
    {
        summary.append(QStringLiteral("%1个连接点")
                           .arg(QLocale().toString(tiePointCount)));
    }
    return summary.isEmpty()
        ? name
        : QStringLiteral("%1 (%2)")
              .arg(name, summary.join(QStringLiteral(", ")));
}

QString workspaceSectionName(xjw::gui::widgets::WorkspaceSection section)
{
    using xjw::gui::widgets::WorkspaceSection;
    switch (section)
    {
    case WorkspaceSection::Photos: return QStringLiteral("照片");
    case WorkspaceSection::Masks: return QStringLiteral("掩膜");
    case WorkspaceSection::ObservationNetwork: return QStringLiteral("观测网络");
    case WorkspaceSection::TiePoints: return QStringLiteral("连接点");
    case WorkspaceSection::DepthMaps: return QStringLiteral("深度图");
    case WorkspaceSection::DenseCloud: return QStringLiteral("稠密点云");
    case WorkspaceSection::Model3D: return QStringLiteral("3D模型");
    case WorkspaceSection::Dem: return QStringLiteral("DEM");
    case WorkspaceSection::Orthomosaic: return QStringLiteral("正射影像");
    case WorkspaceSection::ReferenceData: return QStringLiteral("参考数据");
    case WorkspaceSection::Reports: return QStringLiteral("报告");
    case WorkspaceSection::Unknown: return QString();
    }
    return QString();
}

QString depthRecordPrimaryPath(const QJsonObject &record)
{
    for (const QString &key : {
             QStringLiteral("depth_png"),
             QStringLiteral("raw_depth_path"),
             QStringLiteral("valid_mask_path"),
             QStringLiteral("preview_path")})
    {
        const QString path = record.value(key).toString().trimmed();
        if (!path.isEmpty())
        {
            return path;
        }
    }
    return QString();
}

QString depthQualityLabel(QString profile)
{
    profile = profile.trimmed().toLower();
    if (profile == QStringLiteral("highest")) return QStringLiteral("超高质量");
    if (profile == QStringLiteral("high")) return QStringLiteral("高质量");
    if (profile == QStringLiteral("medium")) return QStringLiteral("中等质量");
    if (profile == QStringLiteral("low")) return QStringLiteral("低质量");
    if (profile == QStringLiteral("lowest")) return QStringLiteral("超低质量");
    if (profile == QStringLiteral("custom")) return QStringLiteral("自定义质量");
    return QString();
}

QString depthFilterLabel(QString mode)
{
    mode = mode.trimmed().toLower();
    if (mode == QStringLiteral("mild")) return QStringLiteral("轻度过滤");
    if (mode == QStringLiteral("moderate")) return QStringLiteral("中度过滤");
    if (mode == QStringLiteral("aggressive")) return QStringLiteral("强过滤");
    if (mode == QStringLiteral("auto")) return QStringLiteral("自适应过滤");
    return QString();
}

bool isDisplayableMeshResult(const QJsonObject &record)
{
    if (record.contains(QStringLiteral("face_count")) &&
        record.value(QStringLiteral("face_count")).toInt(0) <= 0)
    {
        return false;
    }

    return !record.value(QStringLiteral("final_model_path")).toString().isEmpty()
        || !record.value(QStringLiteral("model_obj")).toString().isEmpty()
        || !record.value(QStringLiteral("model_ply")).toString().isEmpty()
        || !record.value(QStringLiteral("mesh_ply")).toString().isEmpty();
}

int displayableMeshResultCount(const QJsonArray &modelResults)
{
    int count = 0;
    for (const QJsonValue &value : modelResults)
    {
        if (!value.isObject())
        {
            continue;
        }
        if (isDisplayableMeshResult(value.toObject()))
        {
            ++count;
        }
    }
    return count;
}

bool isTreeResultKey(const QString &key)
{
    return key == QStringLiteral("image_match_results")
        || key == QStringLiteral("observation_network_results")
        || key == QStringLiteral("aerial_triangulation_results")
        || key == QStringLiteral("depth_map_results")
        || key == QStringLiteral("dense_cloud_results")
        || key == QStringLiteral("model_results")
        || key == QStringLiteral("dem_results")
        || key == QStringLiteral("ortho_results")
        || key == QStringLiteral("report_results")
        || key == QStringLiteral("reference_datasets");
}

bool hasTreeResultKeys(const QJsonObject &meta)
{
    for (auto it = meta.constBegin(); it != meta.constEnd(); ++it)
    {
        if (isTreeResultKey(it.key()))
        {
            return true;
        }
    }
    return false;
}

int compareNaturalText(QString lhs, QString rhs)
{
    int li = 0;
    int ri = 0;
    const int ln = lhs.size();
    const int rn = rhs.size();

    while (li < ln && ri < rn)
    {
        const QChar lc = lhs.at(li);
        const QChar rc = rhs.at(ri);
        if (lc.isDigit() && rc.isDigit())
        {
            const int lhsStart = li;
            const int rhsStart = ri;
            while (li < ln && lhs.at(li).isDigit()) ++li;
            while (ri < rn && rhs.at(ri).isDigit()) ++ri;

            int lhsSig = lhsStart;
            int rhsSig = rhsStart;
            while (lhsSig + 1 < li && lhs.at(lhsSig) == QLatin1Char('0')) ++lhsSig;
            while (rhsSig + 1 < ri && rhs.at(rhsSig) == QLatin1Char('0')) ++rhsSig;

            const int lhsDigits = li - lhsSig;
            const int rhsDigits = ri - rhsSig;
            if (lhsDigits != rhsDigits)
            {
                return lhsDigits < rhsDigits ? -1 : 1;
            }
            for (int offset = 0; offset < lhsDigits; ++offset)
            {
                const int ld = lhs.at(lhsSig + offset).digitValue();
                const int rd = rhs.at(rhsSig + offset).digitValue();
                if (ld != rd)
                {
                    return ld < rd ? -1 : 1;
                }
            }

            const int lhsRun = li - lhsStart;
            const int rhsRun = ri - rhsStart;
            if (lhsRun != rhsRun)
            {
                return lhsRun < rhsRun ? -1 : 1;
            }
            continue;
        }

        const QChar lf = lc.toCaseFolded();
        const QChar rf = rc.toCaseFolded();
        if (lf != rf)
        {
            return lf.unicode() < rf.unicode() ? -1 : 1;
        }
        ++li;
        ++ri;
    }

    if (li == ln && ri == rn)
    {
        return 0;
    }
    return li == ln ? -1 : 1;
}

QString referenceDatasetPath(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("path")).toString();
    if (path.isEmpty()) path = record.value(QStringLiteral("file_path")).toString();
    if (path.isEmpty()) path = record.value(QStringLiteral("dem_path")).toString();
    if (path.isEmpty()) path = record.value(QStringLiteral("lidar_path")).toString();
    if (path.isEmpty()) path = record.value(QStringLiteral("cloud_path")).toString();
    return path;
}

QString referenceDatasetTypeLabel(QString type)
{
    type = type.trimmed().toLower();
    if (type == QStringLiteral("dem") || type == QStringLiteral("reference_dem"))
    {
        return QStringLiteral("DEM");
    }
    if (type == QStringLiteral("lidar") || type == QStringLiteral("las") || type == QStringLiteral("laz") ||
        type == QStringLiteral("copc") || type == QStringLiteral("reference_lidar"))
    {
        return QStringLiteral("LiDAR");
    }
    if (type == QStringLiteral("point_cloud") || type == QStringLiteral("cloud"))
    {
        return QStringLiteral("点云");
    }
    if (type.isEmpty())
    {
        return QStringLiteral("参考");
    }
    return type;
}

QString referenceDatasetRoleLabel(QString role)
{
    role = role.trimmed().toLower();
    if (role == QStringLiteral("validation") || role == QStringLiteral("quality_check"))
    {
        return QStringLiteral("精度检查");
    }
    if (role == QStringLiteral("ba_prior") || role == QStringLiteral("bundle_adjustment") ||
        role == QStringLiteral("reference_prior"))
    {
        return QStringLiteral("BA约束");
    }
    if (role == QStringLiteral("alignment") || role == QStringLiteral("registration"))
    {
        return QStringLiteral("配准");
    }
    if (role.isEmpty())
    {
        return QString();
    }
    return role;
}

QString resultPath(const QJsonObject &record, std::initializer_list<const char *> keys)
{
    for (const char *key : keys)
    {
        const QString path = record.value(QString::fromLatin1(key)).toString().trimmed();
        if (!path.isEmpty())
        {
            return path;
        }
    }
    return QString();
}

int countObjectsWithPath(const QJsonArray &records, std::initializer_list<const char *> keys)
{
    int count = 0;
    for (const QJsonValue &value : records)
    {
        if (!value.isObject())
        {
            continue;
        }
        if (!resultPath(value.toObject(), keys).isEmpty())
        {
            ++count;
        }
    }
    return count;
}

int displayableSparseResultCount(const QJsonArray &atResults)
{
    int count = 0;
    for (const QJsonValue &value : atResults)
    {
        if (!value.isObject())
        {
            continue;
        }

        const QJsonObject files = value.toObject().value(QStringLiteral("files")).toObject();
        if (!files.value(QStringLiteral("sparse_cloud_xyz")).toString().trimmed().isEmpty())
        {
            ++count;
        }
    }
    return count;
}

} // namespace xjw::gui::widgets::data_tree
