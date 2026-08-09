#include "CameraReferenceCsvExporter.h"

#include "model/CameraReferenceSet.h"

#include <QSaveFile>
#include <QStringConverter>
#include <QTextStream>

namespace xjw::gui::reference
{
namespace
{

QString csvCell(QString text)
{
    text.replace(QLatin1Char('"'), QStringLiteral("\"\""));
    return QStringLiteral("\"%1\"").arg(text);
}

QString optionalValue(const std::optional<double> &value)
{
    return value ? QString::number(*value, 'g', 16) : QString();
}

void writeRawRow(QTextStream &stream,
                 const QString &imageUuid,
                 const QString &sourceLabel,
                 const QString &matchStatus,
                 const QString &enabled,
                 const QString &reason,
                 const camera_reference::RawCameraReference &raw)
{
    const auto position = raw.position;
    const auto orientation = raw.orientationYprDegrees;
    const auto sigma = raw.positionSigma;
    stream << csvCell(imageUuid) << ','
           << csvCell(sourceLabel) << ','
           << csvCell(matchStatus) << ','
           << enabled << ','
           << csvCell(reason) << ','
           << (position ? QString::number((*position)[0], 'g', 16) : QString()) << ','
           << (position ? QString::number((*position)[1], 'g', 16) : QString()) << ','
           << (position ? QString::number((*position)[2], 'g', 16) : QString()) << ','
           << (orientation ? QString::number((*orientation)[0], 'g', 16) : QString()) << ','
           << (orientation ? QString::number((*orientation)[1], 'g', 16) : QString()) << ','
           << (orientation ? QString::number((*orientation)[2], 'g', 16) : QString()) << ','
           << (sigma ? QString::number((*sigma)[0], 'g', 16) : QString()) << ','
           << (sigma ? QString::number((*sigma)[1], 'g', 16) : QString()) << ','
           << (sigma ? QString::number((*sigma)[2], 'g', 16) : QString()) << ','
           << optionalValue(raw.horizontalSigmaMeters) << ','
           << csvCell(raw.timestamp) << '\n';
}

} // namespace

bool exportCameraReferenceCsv(const camera_reference::CameraReferenceSet &referenceSet,
                              const QString &path,
                              QString *error)
{
    if (error)
    {
        error->clear();
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        if (error)
        {
            *error = file.errorString();
        }
        return false;
    }

    QTextStream stream(&file);
    stream.setEncoding(QStringConverter::Utf8);
    stream << "image_uuid,file,match_status,enabled,reason,longitude_deg,latitude_deg,height_m,"
              "yaw_deg,pitch_deg,roll_deg,sigma_e_m,sigma_n_m,sigma_u_m,"
              "sigma_horizontal_m,timestamp\n";
    for (const camera_reference::CameraReferenceRecord &record : referenceSet.records())
    {
        writeRawRow(stream,
                    record.imageUuid,
                    record.sourceLabel,
                    QStringLiteral("matched"),
                    record.enabled ? QStringLiteral("true") : QStringLiteral("false"),
                    QString(),
                    record.raw);
    }
    for (const camera_reference::UnmatchedCameraReferenceRecord &record
         : referenceSet.unmatchedRecords())
    {
        writeRawRow(stream,
                    QString(),
                    record.sourceLabel,
                    QStringLiteral("unmatched"),
                    QString(),
                    record.reason,
                    record.raw);
    }
    stream.flush();
    if (stream.status() != QTextStream::Ok)
    {
        if (error)
        {
            *error = QStringLiteral("写入相机参考 CSV 失败");
        }
        file.cancelWriting();
        return false;
    }
    if (!file.commit())
    {
        if (error)
        {
            *error = file.errorString();
        }
        return false;
    }
    return true;
}

} // namespace xjw::gui::reference
