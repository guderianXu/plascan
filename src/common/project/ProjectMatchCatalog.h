#pragma once

#include <QJsonObject>
#include <QPair>
#include <QString>
#include <QVector>

namespace xjw::common::project
{

QString encodeImagePairKey(const QString &first,
                           const QString &second,
                           const QString &separator);
QString canonicalImagePairKey(const QString &left,
                              const QString &right,
                              const QString &separator);
bool decodeImagePairKey(const QString &key,
                        const QString &separator,
                        QString *first,
                        QString *second);

QVector<QPair<QString, QString>> collectMatchedImageNamePairs(
    const QString &project_path,
    const QJsonObject &metadata);
} // namespace xjw::common::project
