#include "MetashapeCameraReferenceImporter.h"

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QRegularExpression>
#include <QSet>
#include <QStringConverter>

#include <array>
#include <cmath>
#include <initializer_list>
#include <utility>

namespace xjw::gui::reference_import
{
namespace
{

struct CameraColumns
{
    int file = -1;
    int latitude = -1;
    int longitude = -1;
    int height = -1;
    int roll = -1;
    int pitch = -1;
    int yaw = -1;
    int time = -1;
    int stdDevNorth = -1;
    int stdDevEast = -1;
    int stdDevUp = -1;
    int stdDevHorizontal = -1;
};

QString displayPath(const QString &path)
{
    return QFileInfo(path).absoluteFilePath();
}

QString portableFileName(QString path)
{
    path.replace(QLatin1Char('\\'), QLatin1Char('/'));
    return path.section(QLatin1Char('/'), -1).trimmed();
}

QString lineError(const QString &path, int lineNumber, const QString &message)
{
    return QStringLiteral("%1：第 %2 行：%3")
        .arg(displayPath(path))
        .arg(lineNumber)
        .arg(message);
}

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

bool readUtf8Lines(const QString &path,
                   const QString &description,
                   QStringList *lines,
                   QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法读取%1 %2：%3")
                     .arg(description, displayPath(path), file.errorString()));
        return false;
    }

    const QByteArray bytes = file.readAll();
    if (file.error() != QFileDevice::NoError)
    {
        setError(errorMessage,
                 QStringLiteral("读取%1 %2 时失败：%3")
                     .arg(description, displayPath(path), file.errorString()));
        return false;
    }
    QStringDecoder decoder(QStringDecoder::Utf8);
    QString content = decoder.decode(bytes);
    if (decoder.hasError())
    {
        setError(errorMessage,
                 QStringLiteral("%1 %2 不是有效的 UTF-8 文本")
                     .arg(description, displayPath(path)));
        return false;
    }
    if (!content.isEmpty() && content.front() == QChar::ByteOrderMark)
    {
        content.remove(0, 1);
    }
    if (lines)
    {
        *lines = content.split(QLatin1Char('\n'), Qt::KeepEmptyParts);
    }
    return true;
}

QStringList tabFields(QString line)
{
    if (line.endsWith(QLatin1Char('\r')))
    {
        line.chop(1);
    }
    QStringList fields = line.split(QLatin1Char('\t'), Qt::KeepEmptyParts);
    for (QString &field : fields)
    {
        field = field.trimmed();
    }
    while (!fields.isEmpty() && fields.back().isEmpty())
    {
        fields.removeLast();
    }
    return fields;
}

QString normalizeHeader(QString value)
{
    value = value.trimmed().toCaseFolded();
    if (value.startsWith(QLatin1Char('#')))
    {
        value.remove(0, 1);
    }
    value = value.trimmed();
    value.replace(QRegularExpression(QStringLiteral("\\s+")), QStringLiteral(" "));
    return value;
}

int findColumn(const QHash<QString, int> &columns,
               std::initializer_list<const char *> aliases,
               bool *ambiguous)
{
    int result = -1;
    for (const char *alias : aliases)
    {
        const auto it = columns.constFind(QString::fromLatin1(alias));
        if (it == columns.constEnd())
        {
            continue;
        }
        if (result >= 0 && result != it.value())
        {
            *ambiguous = true;
            return -1;
        }
        result = it.value();
    }
    return result;
}

bool resolveColumns(const QStringList &header,
                    const QString &path,
                    int lineNumber,
                    CameraColumns *out,
                    QStringList *warnings,
                    QString *errorMessage)
{
    QHash<QString, int> columns;
    for (int index = 0; index < header.size(); ++index)
    {
        const QString name = normalizeHeader(header.at(index));
        if (name.isEmpty())
        {
            continue;
        }
        if (columns.contains(name))
        {
            setError(errorMessage,
                     lineError(path,
                               lineNumber,
                               QStringLiteral("表头列重复：%1").arg(header.at(index))));
            return false;
        }
        columns.insert(name, index);
    }

    bool ambiguous = false;
    out->file = findColumn(columns, {"file"}, &ambiguous);
    out->latitude = findColumn(columns, {"wgs84_lat", "wgs84 lat"}, &ambiguous);
    out->longitude = findColumn(columns, {"wgs84_lon", "wgs84 lon"}, &ambiguous);
    out->height = findColumn(columns,
                             {"wgs84_h", "wgs84 h", "wgs84_height", "wgs84 height"},
                             &ambiguous);
    out->roll = findColumn(columns, {"roll"}, &ambiguous);
    out->pitch = findColumn(columns, {"pitch"}, &ambiguous);
    out->yaw = findColumn(columns, {"yaw"}, &ambiguous);
    out->time = findColumn(columns, {"time"}, &ambiguous);
    out->stdDevNorth = findColumn(columns,
                                  {"std dev n (m)", "std dev north (m)"},
                                  &ambiguous);
    out->stdDevEast = findColumn(columns,
                                 {"std dev e (m)", "std dev east (m)"},
                                 &ambiguous);
    out->stdDevUp = findColumn(columns,
                               {"std dev u (m)", "std dev up (m)"},
                               &ambiguous);
    out->stdDevHorizontal = findColumn(columns,
                                       {"std dev hz (m)",
                                        "std dev h (m)",
                                        "std dev horizontal (m)"},
                                       &ambiguous);
    if (ambiguous)
    {
        setError(errorMessage,
                 lineError(path, lineNumber, QStringLiteral("表头中存在含义重复的列")));
        return false;
    }

    const std::array<std::pair<int, QString>, 8> required_columns{{
        {out->file, QStringLiteral("file")},
        {out->latitude, QStringLiteral("WGS84_lat")},
        {out->longitude, QStringLiteral("WGS84_lon")},
        {out->height, QStringLiteral("WGS84_H")},
        {out->roll, QStringLiteral("roll")},
        {out->pitch, QStringLiteral("pitch")},
        {out->yaw, QStringLiteral("yaw")},
        {out->time, QStringLiteral("time")},
    }};
    QStringList missing_columns;
    for (const auto &[index, name] : required_columns)
    {
        if (index < 0)
        {
            missing_columns.push_back(name);
        }
    }
    if (!missing_columns.isEmpty())
    {
        setError(errorMessage,
                 lineError(path,
                           lineNumber,
                           QStringLiteral("缺少必需列：%1").arg(missing_columns.join(", "))));
        return false;
    }

    const std::array<std::pair<int, QString>, 4> optional_columns{{
        {out->stdDevNorth, QStringLiteral("Std Dev n (m)")},
        {out->stdDevEast, QStringLiteral("Std Dev e (m)")},
        {out->stdDevUp, QStringLiteral("Std Dev u (m)")},
        {out->stdDevHorizontal, QStringLiteral("Std Dev Hz (m)")},
    }};
    for (const auto &[index, name] : optional_columns)
    {
        if (index < 0)
        {
            warnings->push_back(QStringLiteral("未找到可选列 %1；对应精度将留空").arg(name));
        }
    }
    return true;
}

QString fieldAt(const QStringList &fields, int index)
{
    return index >= 0 && index < fields.size() ? fields.at(index) : QString();
}

bool parseFinite(const QString &text,
                 const QString &path,
                 int lineNumber,
                 const QString &fieldName,
                 double *value,
                 QStringList *errors)
{
    bool ok = false;
    const double parsed_value = text.toDouble(&ok);
    if (!ok || !std::isfinite(parsed_value))
    {
        errors->push_back(
            lineError(path,
                      lineNumber,
                      QStringLiteral("字段 %1 必须是有限数值，当前值为“%2”")
                          .arg(fieldName, text)));
        return false;
    }
    *value = parsed_value;
    return true;
}

bool parseRecord(const QStringList &fields,
                 const CameraColumns &columns,
                 const QString &path,
                 int lineNumber,
                 RawCameraReferenceRecord *record,
                 QStringList *errors)
{
    struct RequiredNumber
    {
        int column;
        const char *name;
        double RawCameraReferenceRecord::*value;
    };
    const std::array<RequiredNumber, 6> required_numbers{{
        {columns.latitude, "WGS84_lat", &RawCameraReferenceRecord::wgs84LatitudeDegrees},
        {columns.longitude, "WGS84_lon", &RawCameraReferenceRecord::wgs84LongitudeDegrees},
        {columns.height, "WGS84_H", &RawCameraReferenceRecord::wgs84EllipsoidalHeightMeters},
        {columns.roll, "roll", &RawCameraReferenceRecord::rollDegrees},
        {columns.pitch, "pitch", &RawCameraReferenceRecord::pitchDegrees},
        {columns.yaw, "yaw", &RawCameraReferenceRecord::yawDegrees},
    }};

    bool ok = true;
    for (const RequiredNumber &number : required_numbers)
    {
        ok = parseFinite(fieldAt(fields, number.column),
                         path,
                         lineNumber,
                         QString::fromLatin1(number.name),
                         &(record->*number.value),
                         errors) && ok;
    }

    struct OptionalNumber
    {
        int column;
        const char *name;
        std::optional<double> RawCameraReferenceRecord::*value;
    };
    const std::array<OptionalNumber, 4> optional_numbers{{
        {columns.stdDevNorth, "Std Dev n (m)", &RawCameraReferenceRecord::stdDevNorthMeters},
        {columns.stdDevEast, "Std Dev e (m)", &RawCameraReferenceRecord::stdDevEastMeters},
        {columns.stdDevUp, "Std Dev u (m)", &RawCameraReferenceRecord::stdDevUpMeters},
        {columns.stdDevHorizontal,
         "Std Dev Hz (m)",
         &RawCameraReferenceRecord::stdDevHorizontalMeters},
    }};
    for (const OptionalNumber &number : optional_numbers)
    {
        const QString text = fieldAt(fields, number.column);
        if (text.isEmpty())
        {
            (record->*number.value).reset();
            continue;
        }
        double value = 0.0;
        const bool value_ok = parseFinite(text,
                                          path,
                                          lineNumber,
                                          QString::fromLatin1(number.name),
                                          &value,
                                          errors);
        if (value_ok)
        {
            record->*number.value = value;
        }
        ok = value_ok && ok;
    }

    if (record->wgs84LatitudeDegrees < -90.0 || record->wgs84LatitudeDegrees > 90.0)
    {
        errors->push_back(lineError(path, lineNumber, QStringLiteral("WGS84_lat 必须位于 [-90, 90]")));
        ok = false;
    }
    if (record->wgs84LongitudeDegrees < -180.0 || record->wgs84LongitudeDegrees > 180.0)
    {
        errors->push_back(lineError(path, lineNumber, QStringLiteral("WGS84_lon 必须位于 [-180, 180]")));
        ok = false;
    }
    const std::array<std::pair<const std::optional<double> *, const char *>, 4> sigmas{{
        {&record->stdDevNorthMeters, "Std Dev n (m)"},
        {&record->stdDevEastMeters, "Std Dev e (m)"},
        {&record->stdDevUpMeters, "Std Dev u (m)"},
        {&record->stdDevHorizontalMeters, "Std Dev Hz (m)"}
    }};
    for (const auto &[sigma, name] : sigmas)
    {
        if (*sigma && **sigma <= 0.0)
        {
            errors->push_back(lineError(
                path,
                lineNumber,
                QStringLiteral("字段 %1 必须大于 0").arg(QString::fromLatin1(name))));
            ok = false;
        }
    }
    const int position_sigma_count = static_cast<int>(record->stdDevNorthMeters.has_value())
        + static_cast<int>(record->stdDevEastMeters.has_value())
        + static_cast<int>(record->stdDevUpMeters.has_value());
    if (position_sigma_count != 0 && position_sigma_count != 3)
    {
        errors->push_back(lineError(
            path,
            lineNumber,
            QStringLiteral("Std Dev n/e/u 必须同时提供或同时留空")));
        ok = false;
    }
    return ok;
}

bool parseCameraFile(const QString &path,
                     MetashapeCameraReferenceImportResult *result,
                     QString *errorMessage)
{
    QStringList lines;
    if (!readUtf8Lines(path, QStringLiteral("相机参考文件"), &lines, errorMessage))
    {
        return false;
    }

    int header_index = -1;
    for (int index = 0; index < lines.size(); ++index)
    {
        if (!lines.at(index).trimmed().isEmpty())
        {
            header_index = index;
            break;
        }
    }
    if (header_index < 0)
    {
        setError(errorMessage, QStringLiteral("相机参考文件为空：%1").arg(displayPath(path)));
        return false;
    }

    const QStringList header = tabFields(lines.at(header_index));
    if (header.size() < 2)
    {
        setError(errorMessage,
                 lineError(path, header_index + 1, QStringLiteral("表头必须使用制表符分隔")));
        return false;
    }

    CameraColumns columns;
    if (!resolveColumns(header,
                        path,
                        header_index + 1,
                        &columns,
                        &result->warnings,
                        errorMessage))
    {
        return false;
    }

    QSet<QString> seen_file_names;
    QStringList row_errors;
    for (int index = header_index + 1; index < lines.size(); ++index)
    {
        const QString trimmed_line = lines.at(index).trimmed();
        if (trimmed_line.isEmpty() || trimmed_line.startsWith(QLatin1Char('#')))
        {
            continue;
        }

        const QStringList fields = tabFields(lines.at(index));
        if (fields.size() > header.size())
        {
            row_errors.push_back(
                lineError(path,
                          index + 1,
                          QStringLiteral("数据列多于表头，且尾部包含非空值")));
            continue;
        }

        RawCameraReferenceRecord record;
        record.fileName = fieldAt(fields, columns.file);
        const QString duplicate_key = portableFileName(record.fileName).toCaseFolded();
        if (record.fileName.isEmpty() || duplicate_key.isEmpty())
        {
            row_errors.push_back(
                lineError(path, index + 1, QStringLiteral("字段 file 不能为空且必须是有效文件名")));
            continue;
        }
        if (seen_file_names.contains(duplicate_key))
        {
            row_errors.push_back(
                lineError(path,
                          index + 1,
                          QStringLiteral("文件名重复：%1").arg(record.fileName)));
            continue;
        }
        seen_file_names.insert(duplicate_key);

        record.timeText = fieldAt(fields, columns.time);
        if (parseRecord(fields, columns, path, index + 1, &record, &row_errors))
        {
            result->records.push_back(record);
        }
    }

    if (!row_errors.isEmpty())
    {
        setError(errorMessage, row_errors.join(QLatin1Char('\n')));
        return false;
    }
    if (result->records.isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("相机参考文件不包含数据记录：%1").arg(displayPath(path)));
        return false;
    }
    return true;
}

bool parseLeverArmFile(const QString &path, LeverArm *leverArm, QString *errorMessage)
{
    QStringList lines;
    if (!readUtf8Lines(path, QStringLiteral("GNSS 偏移文件"), &lines, errorMessage))
    {
        return false;
    }

    const QRegularExpression pattern(QStringLiteral("^\\s*([XxYyZz])\\s*=\\s*(.*?)\\s*$"));
    QHash<QChar, double> values;
    QStringList errors;
    for (int index = 0; index < lines.size(); ++index)
    {
        const QString line = lines.at(index);
        if (line.trimmed().isEmpty() || line.trimmed().startsWith(QLatin1Char('#')))
        {
            continue;
        }

        const QRegularExpressionMatch match = pattern.match(line);
        if (!match.hasMatch())
        {
            errors.push_back(
                lineError(path,
                          index + 1,
                          QStringLiteral("GNSS 偏移必须使用 X=、Y= 或 Z= 格式")));
            continue;
        }
        const QChar axis = match.captured(1).front().toUpper();
        if (values.contains(axis))
        {
            errors.push_back(
                lineError(path,
                          index + 1,
                          QStringLiteral("GNSS 偏移分量 %1 重复").arg(axis)));
            continue;
        }

        double value = 0.0;
        if (parseFinite(match.captured(2).trimmed(),
                        path,
                        index + 1,
                        QStringLiteral("GNSS %1").arg(axis),
                        &value,
                        &errors))
        {
            values.insert(axis, value);
        }
    }

    if (!errors.isEmpty())
    {
        setError(errorMessage, errors.join(QLatin1Char('\n')));
        return false;
    }
    QStringList missing_axes;
    for (const QChar axis : {QLatin1Char('X'), QLatin1Char('Y'), QLatin1Char('Z')})
    {
        if (!values.contains(axis))
        {
            missing_axes.push_back(QString(axis));
        }
    }
    if (!missing_axes.isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("GNSS 偏移文件 %1 缺少分量：%2")
                     .arg(displayPath(path), missing_axes.join(QStringLiteral(", "))));
        return false;
    }

    leverArm->xMeters = values.value(QLatin1Char('X'));
    leverArm->yMeters = values.value(QLatin1Char('Y'));
    leverArm->zMeters = values.value(QLatin1Char('Z'));
    return true;
}

} // namespace

bool importMetashapeCameraReferenceTxt(
    const QString &cameraTxtPath,
    const QString &gnssOffsetTxtPath,
    MetashapeCameraReferenceImportResult *result,
    QString *errorMessage)
{
    if (!result)
    {
        setError(errorMessage, QStringLiteral("相机参考导入结果输出不能为空"));
        return false;
    }
    if (cameraTxtPath.trimmed().isEmpty())
    {
        setError(errorMessage, QStringLiteral("相机参考文件路径不能为空"));
        return false;
    }

    MetashapeCameraReferenceImportResult parsed_result;
    QString parse_error;
    if (!parseCameraFile(cameraTxtPath, &parsed_result, &parse_error))
    {
        setError(errorMessage, parse_error);
        return false;
    }
    if (!gnssOffsetTxtPath.trimmed().isEmpty())
    {
        LeverArm lever_arm;
        if (!parseLeverArmFile(gnssOffsetTxtPath, &lever_arm, &parse_error))
        {
            setError(errorMessage, parse_error);
            return false;
        }
        parsed_result.leverArm = lever_arm;
    }

    *result = std::move(parsed_result);
    if (errorMessage)
    {
        errorMessage->clear();
    }
    return true;
}

} // namespace xjw::gui::reference_import
