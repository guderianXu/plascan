#include "MarkerCsv.h"

#include <QDir>
#include <QFile>
#include <QHash>
#include <QTextStream>

#include <cmath>

namespace xjw::control_points
{

namespace
{

QString normalizeKey(QString value)
{
    value = value.trimmed().toLower();
    value.remove(QChar(0xFEFF));
    value.replace(QLatin1Char('-'), QLatin1Char('_'));
    value.replace(QLatin1Char(' '), QLatin1Char('_'));
    return value;
}

int unquotedDelimiterCount(const QString &line, QChar delimiter)
{
    bool in_quotes = false;
    int count = 0;
    for (int index = 0; index < line.size(); ++index)
    {
        const QChar character = line[index];
        if (character == QLatin1Char('"'))
        {
            if (in_quotes && index + 1 < line.size() && line[index + 1] == QLatin1Char('"'))
            {
                ++index;
            }
            else
            {
                in_quotes = !in_quotes;
            }
        }
        else if (!in_quotes && character == delimiter)
        {
            ++count;
        }
    }
    return count;
}

QStringList splitCsvLine(const QString &line, QChar delimiter)
{
    QStringList cells;
    QString cell;
    bool in_quotes = false;
    for (int index = 0; index < line.size(); ++index)
    {
        const QChar character = line[index];
        if (character == QLatin1Char('"'))
        {
            if (in_quotes && index + 1 < line.size() && line[index + 1] == QLatin1Char('"'))
            {
                cell.append(character);
                ++index;
            }
            else
            {
                in_quotes = !in_quotes;
            }
        }
        else if (!in_quotes && character == delimiter)
        {
            cells.push_back(cell.trimmed());
            cell.clear();
        }
        else
        {
            cell.append(character);
        }
    }
    cells.push_back(cell.trimmed());
    return cells;
}

QString firstValue(const QHash<QString, QString> &row, std::initializer_list<const char *> keys)
{
    for (const char *key : keys)
    {
        const QString value = row.value(QString::fromLatin1(key)).trimmed();
        if (!value.isEmpty()) return value;
    }
    return {};
}

bool parseFiniteDouble(const QString &text, double *value)
{
    bool ok = false;
    const double parsed = text.trimmed().toDouble(&ok);
    if (!ok || !std::isfinite(parsed)) return false;
    if (value) *value = parsed;
    return true;
}

double optionalPositiveDouble(const QHash<QString, QString> &row,
                              std::initializer_list<const char *> keys,
                              double fallback)
{
    double value = 0.0;
    return parseFiniteDouble(firstValue(row, keys), &value) && value > 0.0 ? value : fallback;
}

bool parseEnabled(const QString &text)
{
    const QString value = text.trimmed().toLower();
    return value.isEmpty()
        || (value != QLatin1String("0") && value != QLatin1String("false")
            && value != QLatin1String("no") && value != QLatin1String("off")
            && value != QLatin1String("disabled"));
}

QString normalizedRole(QString value)
{
    value = normalizeKey(value);
    value.remove(QLatin1Char('_'));
    if (value == QLatin1String("control") || value == QLatin1String("controlpoint")
        || value == QLatin1String("groundcontrolpoint") || value == QLatin1String("gcp"))
    {
        return QStringLiteral("control");
    }
    if (value == QLatin1String("check") || value == QLatin1String("checkpoint"))
    {
        return QStringLiteral("check");
    }
    if (value == QLatin1String("tie") || value == QLatin1String("tiemarker")
        || value == QLatin1String("marker"))
    {
        return QStringLiteral("tie");
    }
    if (value == QLatin1String("scalebar") || value == QLatin1String("scale")
        || value == QLatin1String("distance"))
    {
        return QStringLiteral("scale_bar");
    }
    return {};
}

QString roleForRow(const QHash<QString, QString> &row, const MarkerCsvImportOptions &options)
{
    QString role = firstValue(row, {"role", "type", "kind"});
    if (role.isEmpty()) role = options.defaultRole;
    if (!role.isEmpty()) return normalizedRole(role);

    const bool has_scale_fields = !firstValue(row, {"from_id", "point_a", "start_id"}).isEmpty()
        || !firstValue(row, {"to_id", "point_b", "end_id"}).isEmpty()
        || !firstValue(row, {"measured_m", "distance_m", "length_m"}).isEmpty();
    return has_scale_fields ? QStringLiteral("scale_bar") : QString();
}

QString normalizedPath(const QString &path)
{
    return QDir::fromNativeSeparators(QDir::cleanPath(path.trimmed()));
}

QString imageIdForPath(const QString &path, const QHash<QString, QString> &identityByPath)
{
    const QString normalized = normalizedPath(path);
    for (auto it = identityByPath.cbegin(); it != identityByPath.cend(); ++it)
    {
        if (normalizedPath(it.key()).compare(normalized, Qt::CaseInsensitive) == 0)
        {
            return it.value();
        }
    }
    return {};
}

struct PendingScaleBar
{
    QString label;
    QString fromLabel;
    QString toLabel;
    double measuredDistance = 0.0;
    double sigma = 1.0;
    bool enabled = true;
};

} // namespace

MarkerCsvImportResult parseMarkerCsv(const QString &csvText, const MarkerCsvImportOptions &options)
{
    MarkerCsvImportResult result;
    QString normalized_text = csvText;
    normalized_text.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    normalized_text.replace(QLatin1Char('\r'), QLatin1Char('\n'));
    const QStringList lines = normalized_text.split(QLatin1Char('\n'));

    int header_line = -1;
    QChar delimiter = QLatin1Char(',');
    QStringList headers;
    for (int index = 0; index < lines.size(); ++index)
    {
        if (lines[index].trimmed().isEmpty()) continue;
        const int comma_count = unquotedDelimiterCount(lines[index], QLatin1Char(','));
        const int semicolon_count = unquotedDelimiterCount(lines[index], QLatin1Char(';'));
        const int tab_count = unquotedDelimiterCount(lines[index], QLatin1Char('\t'));
        if (semicolon_count > comma_count && semicolon_count >= tab_count)
        {
            delimiter = QLatin1Char(';');
        }
        else if (tab_count > comma_count && tab_count > semicolon_count)
        {
            delimiter = QLatin1Char('\t');
        }
        headers = splitCsvLine(lines[index], delimiter);
        for (QString &header : headers) header = normalizeKey(header);
        if (!headers.isEmpty() && headers.front().startsWith(QLatin1Char('#')))
        {
            headers.front().remove(0, 1);
        }
        header_line = index;
        break;
    }
    if (header_line < 0
        || (!headers.contains(QStringLiteral("id")) && !headers.contains(QStringLiteral("name"))))
    {
        result.error = QStringLiteral("Marker CSV 为空或缺少 id/Name 表头");
        return result;
    }

    QHash<QString, MarkerId> marker_id_by_label;
    QVector<PendingScaleBar> pending_scale_bars;
    try
    {
        for (int line_index = header_line + 1; line_index < lines.size(); ++line_index)
        {
            if (lines[line_index].trimmed().isEmpty()) continue;
            const QStringList cells = splitCsvLine(lines[line_index], delimiter);
            QHash<QString, QString> row;
            for (int column = 0; column < headers.size(); ++column)
            {
                row.insert(headers[column], column < cells.size() ? cells[column] : QString());
            }

            const QString label = firstValue(row, {"id", "name"});
            const QString role = roleForRow(row, options);
            if (label.isEmpty() || role.isEmpty())
            {
                result.error = QStringLiteral("第 %1 行缺少有效 id 或 role").arg(line_index + 1);
                return result;
            }

            if (role == QLatin1String("scale_bar"))
            {
                PendingScaleBar scale_bar;
                scale_bar.label = label;
                scale_bar.fromLabel = firstValue(row, {"from_id", "point_a", "start_id"});
                scale_bar.toLabel = firstValue(row, {"to_id", "point_b", "end_id"});
                scale_bar.sigma = optionalPositiveDouble(row, {"sigma_m", "sigma", "std_m"}, 1.0);
                scale_bar.enabled = parseEnabled(firstValue(row, {"enabled", "use", "active"}));
                if (scale_bar.fromLabel.isEmpty() || scale_bar.toLabel.isEmpty()
                    || !parseFiniteDouble(firstValue(row, {"measured_m", "distance_m", "length_m"}),
                                          &scale_bar.measuredDistance)
                    || scale_bar.measuredDistance <= 0.0)
                {
                    result.error = QStringLiteral("第 %1 行比例尺缺少 from_id/to_id/measured_m")
                                       .arg(line_index + 1);
                    return result;
                }
                pending_scale_bars.push_back(scale_bar);
                continue;
            }

            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            if (!parseFiniteDouble(firstValue(row, {"x", "e", "east", "longitude", "lon"}), &x)
                || !parseFiniteDouble(firstValue(row, {"y", "n", "north", "latitude", "lat"}), &y)
                || !parseFiniteDouble(
                    firstValue(row, {"z", "h", "height", "elevation", "ell.h(m)"}), &z))
            {
                result.error = QStringLiteral("第 %1 行缺少有效 x/y/z 坐标").arg(line_index + 1);
                return result;
            }

            MarkerId marker_id = marker_id_by_label.value(label);
            const MarkerRole marker_role = role == QLatin1String("check")
                ? MarkerRole::CheckPoint
                : role == QLatin1String("tie") ? MarkerRole::TieMarker : MarkerRole::ControlPoint;
            if (marker_id.isEmpty())
            {
                marker_id = result.markerSet.addMarker(label, marker_role);
                marker_id_by_label.insert(label, marker_id);
                if (marker_role == MarkerRole::ControlPoint) ++result.controlPointCount;
                else if (marker_role == MarkerRole::CheckPoint) ++result.checkPointCount;
            }
            else if (result.markerSet.marker(marker_id).role != marker_role)
            {
                result.error = QStringLiteral("第 %1 行的标记角色与同名记录不一致").arg(line_index + 1);
                return result;
            }

            result.markerSet.setMarkerEnabled(
                marker_id, parseEnabled(firstValue(row, {"enabled", "use", "active"})));
            const double sigma = optionalPositiveDouble(row, {"sigma_m", "sigma", "std_m"}, 1.0);
            ReferenceCoordinate coordinate;
            coordinate.x = x;
            coordinate.y = y;
            coordinate.z = z;
            coordinate.sigmaX = optionalPositiveDouble(row, {"sigma_x_m", "sigma_x"}, sigma);
            coordinate.sigmaY = optionalPositiveDouble(row, {"sigma_y_m", "sigma_y"}, sigma);
            coordinate.sigmaZ = optionalPositiveDouble(row, {"sigma_z_m", "sigma_z"}, sigma);
            coordinate.sourceCrs = firstValue(row, {"source_crs", "crs", "coordinate_reference"});
            if (coordinate.sourceCrs.isEmpty()) coordinate.sourceCrs = options.sourceCrs;
            coordinate.axisOrder = firstValue(row, {"axis_order", "coordinate_order"});
            if (coordinate.axisOrder.isEmpty()) coordinate.axisOrder = options.axisOrder;
            coordinate.verticalDatum = firstValue(row, {"vertical_datum", "height_datum"});
            if (coordinate.verticalDatum.isEmpty()) coordinate.verticalDatum = options.verticalDatum;
            coordinate.verticalUnit = firstValue(row, {"vertical_unit", "height_unit", "z_unit"});
            if (coordinate.verticalUnit.isEmpty()) coordinate.verticalUnit = options.verticalUnit;
            result.markerSet.setReferenceCoordinate(marker_id, coordinate);

            const QString image_path = firstValue(row, {"image_path", "image", "photo", "file", "filename"});
            const QString u_text = firstValue(row, {"u", "pixel_u", "pixel_x", "x_px", "col", "column"});
            const QString v_text = firstValue(row, {"v", "pixel_v", "pixel_y", "y_px", "row"});
            if (!image_path.isEmpty() || !u_text.isEmpty() || !v_text.isEmpty())
            {
                double u = 0.0;
                double v = 0.0;
                const QString image_id = imageIdForPath(image_path, options.imageIdentityByPath);
                if (image_id.isEmpty() || !parseFiniteDouble(u_text, &u) || !parseFiniteDouble(v_text, &v))
                {
                    result.error = QStringLiteral("第 %1 行的影像观测无法绑定 image_uuid 或像素坐标无效")
                                       .arg(line_index + 1);
                    return result;
                }
                MarkerProjection projection;
                projection.imageId = image_id;
                projection.imagePathSnapshot = image_path;
                projection.xy = QPointF(u, v);
                projection.state = ProjectionState::ManualPinned;
                projection.source = QStringLiteral("marker_csv");
                result.markerSet.upsertProjection(marker_id, projection);
            }
        }

        for (const PendingScaleBar &pending : pending_scale_bars)
        {
            if (!marker_id_by_label.contains(pending.fromLabel)
                || !marker_id_by_label.contains(pending.toLabel))
            {
                result.error = QStringLiteral("比例尺 %1 引用了不存在的端点").arg(pending.label);
                return result;
            }
            result.markerSet.addScaleBar(pending.label,
                                         marker_id_by_label.value(pending.fromLabel),
                                         marker_id_by_label.value(pending.toLabel),
                                         pending.measuredDistance,
                                         pending.sigma);
            ++result.scaleBarCount;
        }
    }
    catch (const MarkerModelError &exception)
    {
        result.error = QString::fromUtf8(exception.what());
        return result;
    }

    result.ok = true;
    return result;
}

MarkerCsvImportResult readMarkerCsvFile(const QString &path, const MarkerCsvImportOptions &options)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        MarkerCsvImportResult result;
        result.error = QStringLiteral("无法打开 Marker CSV: %1").arg(path);
        return result;
    }
    QTextStream stream(&file);
    stream.setEncoding(QStringConverter::Utf8);
    return parseMarkerCsv(stream.readAll(), options);
}

} // namespace xjw::control_points
