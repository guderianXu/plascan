#pragma once

#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

#include <cstdio>

namespace xjw::cli
{

bool readJsonFile(const QString &path, QJsonObject *object, QString *errorMessage);
bool writeJsonFile(const QString &path, const QJsonObject &object, QString *errorMessage);
void writeJson(FILE *stream,
               const QJsonObject &object,
               QJsonDocument::JsonFormat format = QJsonDocument::Indented);

} // namespace xjw::cli
