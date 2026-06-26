#include "SurveyControlImport.h"

#include <QFile>
#include <QHash>
#include <QJsonArray>
#include <QTextStream>

#include <cmath>

namespace xjw::qc
{

namespace
{

QString normalizeKey(QString value)
{
    value = value.trimmed().toLower();
    value.replace(QChar(0xFEFF), QString());
    value.replace(QLatin1Char('-'), QLatin1Char('_'));
    value.replace(QLatin1Char(' '), QLatin1Char('_'));
    return value;
}

int countUnquotedDelimiter(const QString &line, QChar delimiter)
{
    int count = 0;
    bool inQuotes = false;
    for (int i = 0; i < line.size(); ++i)
    {
        const QChar ch = line.at(i);
        if (ch == QLatin1Char('"'))
        {
            if (inQuotes && i + 1 < line.size() && line.at(i + 1) == QLatin1Char('"'))
            {
                ++i;
            }
            else
            {
                inQuotes = !inQuotes;
            }
        }
        else if (ch == delimiter && !inQuotes)
        {
            ++count;
        }
    }
    return count;
}

QChar detectDelimiter(const QString &headerLine)
{
    const int commaCount = countUnquotedDelimiter(headerLine, QLatin1Char(','));
    const int semicolonCount = countUnquotedDelimiter(headerLine, QLatin1Char(';'));
    return semicolonCount > commaCount ? QLatin1Char(';') : QLatin1Char(',');
}

QStringList splitCsvLine(const QString &line, QChar delimiter)
{
    QStringList cells;
    QString cell;
    bool inQuotes = false;

    for (int i = 0; i < line.size(); ++i)
    {
        const QChar ch = line.at(i);
        if (ch == QLatin1Char('"'))
        {
            if (inQuotes && i + 1 < line.size() && line.at(i + 1) == QLatin1Char('"'))
            {
                cell.append(ch);
                ++i;
            }
            else
            {
                inQuotes = !inQuotes;
            }
        }
        else if (ch == delimiter && !inQuotes)
        {
            cells.append(cell.trimmed());
            cell.clear();
        }
        else
        {
            cell.append(ch);
        }
    }

    cells.append(cell.trimmed());
    return cells;
}

QString valueFor(const QHash<QString, QString> &row, std::initializer_list<const char *> keys)
{
    for (const char *key : keys)
    {
        const QString value = row.value(QString::fromLatin1(key)).trimmed();
        if (!value.isEmpty())
        {
            return value;
        }
    }
    return {};
}

bool parseDouble(const QString &text, double *value)
{
    if (!value)
    {
        return false;
    }

    bool ok = false;
    const double parsed = text.trimmed().toDouble(&ok);
    if (!ok || !std::isfinite(parsed))
    {
        return false;
    }

    *value = parsed;
    return true;
}

bool parseOptionalDouble(const QHash<QString, QString> &row,
                         std::initializer_list<const char *> keys,
                         double *value)
{
    const QString text = valueFor(row, keys);
    return !text.isEmpty() && parseDouble(text, value);
}

bool parseEnabled(const QString &text)
{
    const QString normalized = text.trimmed().toLower();
    if (normalized.isEmpty())
    {
        return true;
    }

    return !(normalized == QLatin1String("0") ||
             normalized == QLatin1String("false") ||
             normalized == QLatin1String("no") ||
             normalized == QLatin1String("disabled") ||
             normalized == QLatin1String("off"));
}

QString normalizeRole(QString role)
{
    role = normalizeKey(role);
    role.remove(QLatin1Char('_'));

    if (role == QLatin1String("control") ||
        role == QLatin1String("controlpoint") ||
        role == QLatin1String("groundcontrolpoint") ||
        role == QLatin1String("gcp"))
    {
        return QStringLiteral("control");
    }

    if (role == QLatin1String("check") ||
        role == QLatin1String("checkpoint") ||
        role == QLatin1String("checkpointpoint"))
    {
        return QStringLiteral("check");
    }

    if (role == QLatin1String("scalebar") ||
        role == QLatin1String("scale") ||
        role == QLatin1String("distance"))
    {
        return QStringLiteral("scale_bar");
    }

    return {};
}

QString inferRole(const QHash<QString, QString> &row, const SurveyControlImportOptions &options)
{
    QString role = valueFor(row, {"role", "type", "kind"});
    if (role.isEmpty())
    {
        role = options.defaultRole;
    }

    if (!role.isEmpty())
    {
        return normalizeRole(role);
    }

    const bool hasScaleBarFields =
        !valueFor(row, {"from_id", "point_a", "start_id"}).isEmpty() ||
        !valueFor(row, {"to_id", "point_b", "end_id"}).isEmpty() ||
        !valueFor(row, {"measured_m", "distance_m", "length_m"}).isEmpty();
    return hasScaleBarFields ? QStringLiteral("scale_bar") : QStringLiteral("control");
}

bool setPointCoordinates(const QHash<QString, QString> &row,
                         QJsonObject *record,
                         QString *error)
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!parseDouble(valueFor(row, {"x", "e", "east", "longitude"}), &x) ||
        !parseDouble(valueFor(row, {"y", "n", "north", "latitude"}), &y) ||
        !parseDouble(valueFor(row, {"z", "h", "height", "elevation"}), &z))
    {
        if (error)
        {
            *error = QStringLiteral("点记录缺少有效 x/y/z 坐标");
        }
        return false;
    }

    (*record)[QStringLiteral("x")] = x;
    (*record)[QStringLiteral("y")] = y;
    (*record)[QStringLiteral("z")] = z;
    return true;
}

void setOptionalSigma(const QHash<QString, QString> &row, QJsonObject *record)
{
    double value = 0.0;
    if (parseOptionalDouble(row, {"sigma_m", "sigma", "std_m"}, &value))
    {
        (*record)[QStringLiteral("sigma_m")] = value;
    }
    if (parseOptionalDouble(row, {"sigma_x_m", "sigma_x"}, &value))
    {
        (*record)[QStringLiteral("sigma_x_m")] = value;
    }
    if (parseOptionalDouble(row, {"sigma_y_m", "sigma_y"}, &value))
    {
        (*record)[QStringLiteral("sigma_y_m")] = value;
    }
    if (parseOptionalDouble(row, {"sigma_z_m", "sigma_z"}, &value))
    {
        (*record)[QStringLiteral("sigma_z_m")] = value;
    }
}

bool observationFromRow(const QHash<QString, QString> &row,
                        QJsonObject *observation,
                        QString *error)
{
    if (!observation)
    {
        return false;
    }

    const QString imagePath = valueFor(row, {"image_path", "image", "photo", "file", "filename"});
    const QString uText = valueFor(row, {"u", "pixel_u", "pixel_x", "x_px", "col", "column"});
    const QString vText = valueFor(row, {"v", "pixel_v", "pixel_y", "y_px", "row"});
    const bool hasAnyObservationField = !imagePath.isEmpty() || !uText.isEmpty() || !vText.isEmpty();
    if (!hasAnyObservationField)
    {
        return false;
    }

    double u = 0.0;
    double v = 0.0;
    if (imagePath.isEmpty() || !parseDouble(uText, &u) || !parseDouble(vText, &v))
    {
        if (error)
        {
            *error = QStringLiteral("影像观测缺少 image_path/pixel_x/pixel_y");
        }
        return false;
    }

    (*observation)[QStringLiteral("image_path")] = imagePath;
    (*observation)[QStringLiteral("u")] = u;
    (*observation)[QStringLiteral("v")] = v;
    return true;
}

void appendObservation(QJsonObject *record, const QJsonObject &observation)
{
    if (!record || observation.isEmpty())
    {
        return;
    }

    QJsonArray observations = record->value(QStringLiteral("observations")).toArray();
    observations.append(observation);
    (*record)[QStringLiteral("observations")] = observations;
}

void appendOrMergePointRecord(QJsonArray *records,
                              QHash<QString, int> *indexById,
                              const QJsonObject &record)
{
    if (!records || !indexById)
    {
        return;
    }

    const QString id = record.value(QStringLiteral("id")).toString();
    const int existingIndex = indexById->value(id, -1);
    if (existingIndex < 0)
    {
        indexById->insert(id, records->size());
        records->append(record);
        return;
    }

    QJsonObject merged = records->at(existingIndex).toObject();
    const QStringList scalarKeys = {
        QStringLiteral("enabled"),
        QStringLiteral("x"),
        QStringLiteral("y"),
        QStringLiteral("z"),
        QStringLiteral("sigma_m"),
        QStringLiteral("sigma_x_m"),
        QStringLiteral("sigma_y_m"),
        QStringLiteral("sigma_z_m"),
    };
    for (const QString &key : scalarKeys)
    {
        if (record.contains(key))
        {
            merged[key] = record.value(key);
        }
    }

    QJsonArray mergedObservations = merged.value(QStringLiteral("observations")).toArray();
    const QJsonArray newObservations = record.value(QStringLiteral("observations")).toArray();
    for (const QJsonValue &value : newObservations)
    {
        mergedObservations.append(value);
    }
    if (!mergedObservations.isEmpty())
    {
        merged[QStringLiteral("observations")] = mergedObservations;
    }
    records->replace(existingIndex, merged);
}

} // namespace

SurveyControlImportResult parseSurveyControlCsv(const QString &csvText,
                                                const SurveyControlImportOptions &options)
{
    SurveyControlImportResult result;
    QJsonArray controlPoints;
    QJsonArray checkPoints;
    QJsonArray scaleBars;
    QHash<QString, int> controlIndexById;
    QHash<QString, int> checkIndexById;

    QString normalizedText = csvText;
    normalizedText.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    normalizedText.replace(QLatin1Char('\r'), QLatin1Char('\n'));

    const QStringList lines = normalizedText.split(QLatin1Char('\n'));
    int headerLine = -1;
    QStringList headers;
    QChar delimiter = QLatin1Char(',');
    for (int i = 0; i < lines.size(); ++i)
    {
        if (lines.at(i).trimmed().isEmpty())
        {
            continue;
        }

        delimiter = detectDelimiter(lines.at(i));
        headers = splitCsvLine(lines.at(i), delimiter);
        for (QString &header : headers)
        {
            header = normalizeKey(header);
        }
        headerLine = i;
        break;
    }

    if (headerLine < 0 || headers.isEmpty())
    {
        result.error = QStringLiteral("CSV 为空或缺少表头");
        return result;
    }

    if (!headers.contains(QStringLiteral("id")))
    {
        result.error = QStringLiteral("Survey Control CSV 缺少 id 列");
        return result;
    }

    for (int lineIndex = headerLine + 1; lineIndex < lines.size(); ++lineIndex)
    {
        const QString rawLine = lines.at(lineIndex);
        if (rawLine.trimmed().isEmpty())
        {
            continue;
        }

        const QStringList cells = splitCsvLine(rawLine, delimiter);
        QHash<QString, QString> row;
        for (int i = 0; i < headers.size(); ++i)
        {
            row.insert(headers.at(i), i < cells.size() ? cells.at(i) : QString());
        }

        const QString id = valueFor(row, {"id", "name"});
        if (id.isEmpty())
        {
            result.error = QStringLiteral("第 %1 行缺少 id").arg(lineIndex + 1);
            return result;
        }

        const QString role = inferRole(row, options);
        if (role.isEmpty())
        {
            result.error = QStringLiteral("第 %1 行 role/type 无法识别").arg(lineIndex + 1);
            return result;
        }

        QJsonObject record;
        record[QStringLiteral("id")] = id;
        record[QStringLiteral("enabled")] = parseEnabled(valueFor(row, {"enabled", "use", "active"}));
        setOptionalSigma(row, &record);

        if (role == QLatin1String("scale_bar"))
        {
            const QString fromId = valueFor(row, {"from_id", "point_a", "start_id"});
            const QString toId = valueFor(row, {"to_id", "point_b", "end_id"});
            double measured = 0.0;
            if (fromId.isEmpty() || toId.isEmpty() ||
                !parseDouble(valueFor(row, {"measured_m", "distance_m", "length_m"}), &measured))
            {
                result.error = QStringLiteral("第 %1 行比例尺缺少 from_id/to_id/measured_m").arg(lineIndex + 1);
                return result;
            }

            record[QStringLiteral("from_id")] = fromId;
            record[QStringLiteral("to_id")] = toId;
            record[QStringLiteral("measured_m")] = measured;
            scaleBars.append(record);
        }
        else
        {
            QString error;
            if (!setPointCoordinates(row, &record, &error))
            {
                result.error = QStringLiteral("第 %1 行: %2").arg(lineIndex + 1).arg(error);
                return result;
            }

            QJsonObject observation;
            if (observationFromRow(row, &observation, &error))
            {
                appendObservation(&record, observation);
            }
            else if (!error.isEmpty())
            {
                result.error = QStringLiteral("第 %1 行: %2").arg(lineIndex + 1).arg(error);
                return result;
            }

            if (role == QLatin1String("control"))
            {
                appendOrMergePointRecord(&controlPoints, &controlIndexById, record);
            }
            else
            {
                appendOrMergePointRecord(&checkPoints, &checkIndexById, record);
            }
        }
    }

    result.controlPointCount = controlPoints.size();
    result.checkPointCount = checkPoints.size();
    result.scaleBarCount = scaleBars.size();
    result.surveyControl[QStringLiteral("control_points")] = controlPoints;
    result.surveyControl[QStringLiteral("check_points")] = checkPoints;
    result.surveyControl[QStringLiteral("scale_bars")] = scaleBars;
    result.ok = true;
    return result;
}

SurveyControlImportResult readSurveyControlCsvFile(const QString &path,
                                                   const SurveyControlImportOptions &options)
{
    SurveyControlImportResult result;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        result.error = QStringLiteral("无法打开 Survey Control CSV: %1").arg(path);
        return result;
    }

    QTextStream stream(&file);
    stream.setEncoding(QStringConverter::Utf8);
    return parseSurveyControlCsv(stream.readAll(), options);
}

} // namespace xjw::qc
