#pragma once

#include <QString>

#include <cstdio>

namespace xjw::cli
{

void printUtf8(FILE *stream, const QString &message, bool appendNewline = true);
void printError(const QString &message);
int registerConsoleLogger();

} // namespace xjw::cli
