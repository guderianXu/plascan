#pragma once

#include <opencv2/core/mat.hpp>

#include <QByteArray>
#include <QString>

#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

namespace xjw::common::io
{

std::filesystem::path toFilesystemPath(const QString &path);
std::string toUtf8Path(const QString &path);
std::string toUtf8Path(const std::filesystem::path &path);
std::string toNativeNarrowPath(const QString &path);
std::string toNativeNarrowPathUtf8(const std::string &path);
QString fromUtf8Path(const std::string &path);
QString fromFilesystemPath(const std::filesystem::path &path);

std::ifstream openInputFile(const QString &path, std::ios::openmode mode = std::ios::binary);
std::ifstream openInputFile(const std::filesystem::path &path, std::ios::openmode mode = std::ios::binary);
std::ofstream openOutputFile(const QString &path, std::ios::openmode mode = std::ios::binary | std::ios::trunc);
std::ofstream openOutputFile(const std::filesystem::path &path,
                             std::ios::openmode mode = std::ios::binary | std::ios::trunc);
std::ifstream openInputFileUtf8(const std::string &path, std::ios::openmode mode = std::ios::binary);
std::ofstream openOutputFileUtf8(const std::string &path,
                                 std::ios::openmode mode = std::ios::binary | std::ios::trunc);

QByteArray readFileBytes(const QString &path, QString *errorMessage = nullptr);
QByteArray readFileBytesUtf8(const std::string &path, QString *errorMessage = nullptr);
bool writeFileBytesAtomic(const QString &path, const QByteArray &bytes, QString *errorMessage = nullptr);
bool writeFileBytesAtomicUtf8(const std::string &path, const QByteArray &bytes, QString *errorMessage = nullptr);

cv::Mat readImage(const QString &path, int flags);
cv::Mat readImageUtf8(const std::string &path, int flags);

bool writeImage(const QString &path, const cv::Mat &image, const std::vector<int> &params = {});
bool writeImageUtf8(const std::string &path, const cv::Mat &image, const std::vector<int> &params = {});

} // namespace xjw::common::io
