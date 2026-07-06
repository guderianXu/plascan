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

// QString 是 PlaScan 内部首选路径类型；std::string 路径按 UTF-8 元数据/CLI 字符串处理。
// 调用方应优先使用下面的通用读写函数，不在业务代码里选择 UTF-8 或本地窄字符编码。
std::filesystem::path toFilesystemPath(const QString &path);
std::filesystem::path toFilesystemPath(const std::string &path);
std::string toUtf8Path(const QString &path);
std::string toUtf8Path(const std::filesystem::path &path);
std::string toNativeNarrowPath(const QString &path);
std::string toNativeNarrowPath(const std::string &path);
std::string toNativeNarrowPath(const std::filesystem::path &path);
QString fromUtf8Path(const std::string &path);
QString fromFilesystemPath(const std::filesystem::path &path);

std::ifstream openInputFile(const QString &path, std::ios::openmode mode = std::ios::binary);
std::ifstream openInputFile(const std::string &path, std::ios::openmode mode = std::ios::binary);
std::ifstream openInputFile(const std::filesystem::path &path, std::ios::openmode mode = std::ios::binary);
std::ofstream openOutputFile(const QString &path, std::ios::openmode mode = std::ios::binary | std::ios::trunc);
std::ofstream openOutputFile(const std::string &path,
                             std::ios::openmode mode = std::ios::binary | std::ios::trunc);
std::ofstream openOutputFile(const std::filesystem::path &path,
                             std::ios::openmode mode = std::ios::binary | std::ios::trunc);

QByteArray readFileBytes(const QString &path, QString *errorMessage = nullptr);
QByteArray readFileBytes(const std::string &path, QString *errorMessage = nullptr);
QByteArray readFileBytes(const std::filesystem::path &path, QString *errorMessage = nullptr);
bool writeFileBytesAtomic(const QString &path, const QByteArray &bytes, QString *errorMessage = nullptr);
bool writeFileBytesAtomic(const std::string &path, const QByteArray &bytes, QString *errorMessage = nullptr);
bool writeFileBytesAtomic(const std::filesystem::path &path, const QByteArray &bytes, QString *errorMessage = nullptr);

cv::Mat readImage(const QString &path, int flags);
cv::Mat readImage(const std::string &path, int flags);
cv::Mat readImage(const std::filesystem::path &path, int flags);

bool writeImage(const QString &path, const cv::Mat &image, const std::vector<int> &params = {});
bool writeImage(const std::string &path, const cv::Mat &image, const std::vector<int> &params = {});
bool writeImage(const std::filesystem::path &path, const cv::Mat &image, const std::vector<int> &params = {});

} // namespace xjw::common::io
