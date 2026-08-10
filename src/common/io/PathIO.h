#pragma once

#include "io/ImageIO.h"

#include <QByteArray>
#include <QString>

#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>

namespace xjw::common::io
{

/**
 * @brief 两个路径经过安全规范化后的关系。
 *
 * `valid=false` 表示至少一个路径无法可靠规范化或查询；涉及删除、覆盖等
 * 破坏性操作的调用方必须拒绝继续。`equivalent` 同时覆盖规范路径相等和
 * 已存在对象的文件系统等价关系（例如硬链接）。祖先关系均为严格关系，
 * 路径相等时两个祖先字段都为 false。
 */
struct SafePathComparison
{
    bool valid = false;
    std::filesystem::path normalizedFirst;
    std::filesystem::path normalizedSecond;
    bool equivalent = false;
    bool firstIsAncestorOfSecond = false;
    bool secondIsAncestorOfFirst = false;
    bool firstIsRoot = false;
    bool secondIsRoot = false;
    std::error_code error;
};

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

/**
 * @brief 以适合覆盖/删除前校验的方式比较两个路径。
 *
 * 路径先转为绝对路径并使用 `weakly_canonical` 解析已存在的父目录，因此
 * 末级尚不存在时仍能识别父目录符号链接和 `..`。Windows 上按文件系统
 * 的大小写不敏感语义比较路径分量。任何错误都会返回 `valid=false`。
 */
SafePathComparison comparePathsSafely(const std::filesystem::path &first,
                                      const std::filesystem::path &second);

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

} // namespace xjw::common::io
