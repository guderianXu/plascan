#pragma once

#include "Camera.h"

#include <QMap>
#include <QJsonArray>
#include <QJsonObject>
#include <QPair>
#include <QStringList>
#include <QString>
#include <QVector>

namespace xjw
{
namespace gui
{
namespace project
{

// 规范化路径：转换为绝对路径并清理 ./../ 等冗余段，便于后续一致比较。
QString normalizePath(const QString &path);

// 将项目元数据统一转换为 project_files 根对象；若已是根对象则直接返回。
QJsonObject projectFilesRootObject(const QJsonObject &meta);

// 读取项目中的影像条目数组（兼容顶层和 project_files 嵌套结构）。
QJsonArray projectImageEntries(const QJsonObject &meta);

// 提取项目中的影像路径列表，自动跳过空路径条目。
QStringList projectImagePaths(const QJsonObject &meta);

// 建立影像路径到影像元数据的映射；可选对路径执行 normalize 以便稳健匹配。
QMap<QString, QJsonObject> projectImageMetaByPath(const QJsonObject &meta,
                                                  bool normalizePaths = false);

// 判断 token 是否能匹配到某张影像：支持绝对路径、文件名、去扩展名三种匹配策略。
bool pathTokenMatchesImage(const QString &token, const QString &imagePath);

// 根据 token 在项目影像列表中解析出实际影像路径；支持绝对路径、文件名、去扩展名匹配。
QString resolveProjectImagePathFromToken(const QString &token, const QJsonObject &meta);

// 根据 token 解析对应的 .sp 特征文件路径；优先通过项目影像匹配，再回退到 token 自身。
QString resolveProjectFeaturePathFromToken(const QString &plascanPath,
                                          const QJsonObject &meta,
                                          const QString &token);

// 根据 token + 后缀解析对应提取器的特征文件路径
QString resolveFeaturePathBySuffix(const QString &plascanPath, const QJsonObject &meta,
                                   const QString &token, const QString &suffix);

// 收集项目中实际存在的特征文件后缀，按默认偏好顺序返回。
QStringList projectFeatureSuffixes(const QString &plascanPath, const QJsonObject &meta);

// 根据项目中已存在的特征文件推断默认显示后缀。
// 优先级面向当前默认流程：.dsk, .alk, .sp, .sift, .orb, .akz, .dedode。
QString inferPreferredFeatureSuffix(const QString &plascanPath, const QJsonObject &meta);

// 判断项目中是否存在指定后缀的特征文件；suffix 可传 ".dsk" 或 "dsk"。
bool projectHasFeatureSuffix(const QString &plascanPath, const QJsonObject &meta, const QString &suffix);

// 将内部相机模型序列化为 JSON 元数据（用于写入项目 metadata）。
QJsonObject cameraToJson(const xjw::Camera &camera);

// 解析 tsai 相机文件并转换为统一 JSON 元数据；失败时返回 false 并写入错误信息。
bool parseTsaiCamera(const QString &tsaiPath, QJsonObject *cameraMeta, QString *errorMsg = nullptr);

// 从项目中的相机 JSON 反序列化为 xjw::Camera；字段不完整时返回 false。
bool cameraFromJson(const QJsonObject &camObj, xjw::Camera *cam);

// 从单个影像元数据条目中读取相机对象；内部复用 image.camera JSON。
bool imageCameraFromEntry(const QJsonObject &imageObj, xjw::Camera *cam);

// 收集项目中已存在的匹配影像对，返回值中的每项为“影像名1 / 影像名2”。
// 优先扫描 assets/matches，再回退到 ipmatch_results 元数据；结果已去重并稳定排序。
QVector<QPair<QString, QString>> collectMatchedImageNamePairs(const QString &plascanPath,
                                                              const QJsonObject &meta);

// 收集项目中已确认无有效匹配的影像对，返回值中的每项为“影像名1 / 影像名2”。
// 这些 pair 已被匹配流程处理过，不应在空三预检中被当作“缺失匹配”触发整轮重匹配。
QVector<QPair<QString, QString>> collectSettledNoMatchImageNamePairs(const QString &plascanPath,
                                                                     const QJsonObject &meta);

} // namespace project
} // namespace gui
} // namespace xjw
