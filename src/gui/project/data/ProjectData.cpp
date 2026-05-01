// =============================================================================
// 文件名: ProjectData.cpp
// 描述:   ProjectData 数据层实现。
//
//         主要逻辑：
//           1. createProject  - 初始化 .plascan ZIP 归档 + 目录结构
//           2. openProject    - 优先从临时缓存恢复，再读归档
//           3. saveProject    - 将内存元数据写回 .plascan 归档
//           4. addImages      - 追加影像引用到 images[] 数组
//           5. setImageCameras- 批量写入相机参数到 images[*].camera
//           6. appendXxx      - 各类结果追加并立即持久化（双保险策略：
//                               先写归档，失败则写临时缓存）
//           7. saveIpfindSettings / saveUiSettings - 直接写归档不经临时缓存
//
//         持久化策略（双保险）：
//           - 优先写 .plascan 归档（原子性好）
//           - 归档写失败时回退写 .plascan_tmp/（防止数据丢失）
//           - 下次 openProject 时优先从 .plascan_tmp/ 恢复
// =============================================================================
#include "ProjectData.h"
#include "PlascanArchive.h"
#include "ProjectIO.h"
#include "Logger.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDateTime>
#include <QTimer>

ProjectData::ProjectData(QObject *parent)
    : QObject(parent)
{
    // 防抖归档写入定时器：将多次 appendIpfind/appendIpmatch/setImageCameras 合并为一次 ZIP 写入
    m_archiveSyncTimer = new QTimer(this);
    m_archiveSyncTimer->setSingleShot(true);
    connect(m_archiveSyncTimer, &QTimer::timeout, this, &ProjectData::syncToArchive);
}

bool ProjectData::createProject(const QString &plascanPath, const QString &projectName)
{
    // 步骤1：构建 project_files.json 的默认 JSON 结构（空影像列表等）
    QJsonObject filesMeta = ProjectFilesManager::defaultFiles();

    // 步骤2：构建 project_config.json 的默认结构，记录项目名称与创建时间
    QJsonObject configMeta = ProjectConfigManager::defaultConfig();
    configMeta["project_name"] = projectName;
    configMeta["created_at"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    configMeta["version"] = "1.0";

    // 步骤3：构建 manifest.json —— ZIP 容器的元数据头，标明格式版本与类型
    QJsonObject manifest;
    manifest["format_version"] = "1.0";
    manifest["type"] = "plascan_project";

    // 步骤4：调用 PlascanArchive::createArchive 创建 .plascan 文件（ZIP）
    //         createArchive 会同时写入 manifest.json 和 project.json（兼容旧版）
    QString err;
    QJsonDocument filesDoc(filesMeta);
    QJsonDocument manifestDoc(manifest);

    if (!PlascanArchive::createArchive(plascanPath,
                                      manifestDoc.toJson(QJsonDocument::Compact),
                                      filesDoc.toJson(QJsonDocument::Compact),
                                      &err)) {
        LOG_ERROR(QStringLiteral("创建项目失败: %1").arg(err));
        return false;
    }

    // 步骤5：向归档追加 project_files.json 和 project_config.json（新版格式）
    PlascanArchive archive(plascanPath);
    if (archive.isValid()) {
        archive.writeEntry("project_files.json", filesDoc.toJson(QJsonDocument::Compact), &err);
        QJsonDocument configDoc(configMeta);
        archive.writeEntry("project_config.json", configDoc.toJson(QJsonDocument::Compact), &err);
    }

    // 步骤6：更新内存状态
    m_projectPath = plascanPath;
    updateMetadata(filesMeta, false);   // false = 不标记为脏（刚创建，不需要保存）
    updateConfig(configMeta, false);
    saveTemporaryMetadata();             // 同步写一份到临时目录以便崩溃恢复

    // 步骤7：在磁盘上创建项目所需的子目录结构（assets/images、assets/ip、assets/matches）
    QDir projectRoot(ProjectIO::projectRootFromPlascan(m_projectPath));
    projectRoot.mkpath(QDir(projectRoot.path()).relativeFilePath(ProjectIO::projectImagesDir(m_projectPath)));
    projectRoot.mkpath(QDir(projectRoot.path()).relativeFilePath(ProjectIO::ipfindOutputDir(m_projectPath)));
    projectRoot.mkpath(QDir(projectRoot.path()).relativeFilePath(ProjectIO::ipmatchOutputDir(m_projectPath)));

    LOG_INFO(QStringLiteral("项目创建成功: %1").arg(plascanPath));
    // 发出 projectOpened 信号（新建项目视同已打开）
    emit projectOpened(plascanPath);

    return true;
}

bool ProjectData::openProject(const QString &plascanPath, QString *errorMsg)
{
    // 先验证归档是否可访问
    PlascanArchive archive(plascanPath);
    if (!archive.isValid()) {
        if (errorMsg) *errorMsg = QStringLiteral("无法打开项目文件");
        return false;
    }

    // 记录路径并清空旧数据
    m_projectPath = plascanPath;
    m_filesManager.setData(QJsonObject());
    m_configManager.setData(QJsonObject());
    m_resultsLoaded = false;    // results 延迟加载：不在此时读取

    // 优先从 .plascan_tmp/ 临时目录恢复（处理上次异常退出后的未保存数据）
    if (loadTemporaryMetadata()) {
        LOG_INFO(QStringLiteral("从临时目录加载项目数据"));
    }

    // 若临时目录中没有 core 数据，则从归档读取（仅读核心数据）
    QString err;
    if (m_filesManager.coreData().isEmpty()) {
        // 优先尝试新版文件名（project_files.json），回退到旧版（project.json）
        QByteArray filesData = archive.readEntry("project_files.json", &err);
        if (filesData.isEmpty()) {
            filesData = archive.readEntry("project.json", &err);
        }

        if (filesData.isEmpty()) {
            m_filesManager.setData(ProjectFilesManager::defaultFiles());
        } else {
            QJsonDocument doc = QJsonDocument::fromJson(filesData);
            QJsonObject obj = doc.object();
            // 斷断格式：旧格式 project_files.json 可能包含 results 键
            // 若包含，则通过 setData() 拆分，并标记 results 已加载（旧格式一次性读入）
            bool hasLegacyResults = obj.contains("ipfind_results")
                                 || obj.contains("ipmatch_results")
                                 || obj.contains("intersection_results")
                                 || obj.contains("bundle_adjust_results");
            m_filesManager.setData(obj);    // setData() 自动拆分到 coreFiles + resultFiles
            if (hasLegacyResults) {
                m_resultsLoaded = true;     // 旧格式数据已全部和 core 一起读入
                LOG_INFO(QStringLiteral("检测到旧格式项目，已将内嵌 results 拆分到独立存储"));
            }
        }
    }

    // 若临时目录中没有 config 数据，则从归档读取
    if (m_configManager.data().isEmpty()) {
        QByteArray configData = archive.readEntry("project_config.json", &err);
        if (configData.isEmpty()) {
            updateConfig(ProjectConfigManager::defaultConfig(), false);
        } else {
            QJsonDocument doc = QJsonDocument::fromJson(configData);
            updateConfig(ProjectConfigManager::mergeWithDefaults(doc.object()), false);
        }
    }

    LOG_INFO(QStringLiteral("项目核心数据加载完成: %1").arg(plascanPath));
    emit projectOpened(plascanPath);

    return true;
}

bool ProjectData::saveProject(QString *errorMsg)
{
    if (m_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    // 完整保存前首先取消防抖定时器，避免重复写入
    if (m_archiveSyncTimer) m_archiveSyncTimer->stop();
    m_resultsDirtyForArchive = false;
    m_coreFileDirtyForArchive = false;

    PlascanArchive archive(m_projectPath);
    if (!archive.isValid()) {
        if (errorMsg) *errorMsg = QStringLiteral("无法打开项目文件");
        return false;
    }

    QString err;

    // 写入核心数据（小，需常写）
    QJsonDocument coreDoc(m_filesManager.coreData());
    if (!archive.writeEntry(ProjectFilesManager::kArchiveCoreFile,
                            coreDoc.toJson(QJsonDocument::Compact), &err)) {
        if (errorMsg) *errorMsg = QStringLiteral("写入 %1 失败: %2")
                                      .arg(ProjectFilesManager::kArchiveCoreFile, err);
        return false;
    }

    // 写入结果数据（Compact JSON，zip 内层压缩）
    if (m_resultsLoaded) {
        QJsonDocument resultsDoc(m_filesManager.resultsData());
        if (!archive.writeEntry(ProjectFilesManager::kArchiveResultsFile,
                                resultsDoc.toJson(QJsonDocument::Compact), &err)) {
            LOG_WARN(QStringLiteral("写入 %1 失败: %2")
                                         .arg(ProjectFilesManager::kArchiveResultsFile, err));
        }
    }

    // 写入配置数据
    QJsonDocument configDoc(m_configManager.data());
    if (!archive.writeEntry("project_config.json",
                            configDoc.toJson(QJsonDocument::Compact), &err)) {
        if (errorMsg) *errorMsg = QStringLiteral("写入 project_config.json 失败: %1").arg(err);
        return false;
    }

    m_isDirty = false;
    emit dirtyStateChanged(false);
    emit projectSaved(m_projectPath);

    clearTemporaryMetadata();
    LOG_INFO(QStringLiteral("项目已保存: %1").arg(m_projectPath));
    return true;
}

void ProjectData::closeProject()
{
    // 关闭前将待写入归档的脚脂数据刷新
    if (m_archiveSyncTimer && m_archiveSyncTimer->isActive()) {
        m_archiveSyncTimer->stop();
        syncToArchive();
    }
    m_projectPath.clear();
    m_filesManager.setData(QJsonObject());
    m_configManager.setData(QJsonObject());
    m_isDirty = false;
    m_resultsLoaded = false;
    m_resultsDirtyForArchive = false;
    m_coreFileDirtyForArchive = false;
    emit projectClosed();
}

// 防抖定时器回调：将内存脚脂数据批量写入 .plascan 归档（最多 2 次 zip_open/close）
void ProjectData::syncToArchive()
{
    if (m_projectPath.isEmpty()) return;
    if (!m_resultsDirtyForArchive && !m_coreFileDirtyForArchive) return;

    PlascanArchive archive(m_projectPath);
    if (!archive.isValid()) {
        // 归档无法访问，回退到临时文件
        saveTemporaryMetadata();
        return;
    }

    if (m_resultsDirtyForArchive && m_resultsLoaded) {
        QString err;
        if (!archive.writeEntry(ProjectFilesManager::kArchiveResultsFile,
                                QJsonDocument(m_filesManager.resultsData()).toJson(QJsonDocument::Compact), &err)) {
            LOG_WARN(QStringLiteral("同步写入 %1 失败: %2")
                                         .arg(ProjectFilesManager::kArchiveResultsFile, err));
        }
        m_resultsDirtyForArchive = false;
    }

    if (m_coreFileDirtyForArchive) {
        QString err;
        if (!archive.writeEntry(ProjectFilesManager::kArchiveCoreFile,
                                QJsonDocument(m_filesManager.coreData()).toJson(QJsonDocument::Compact), &err)) {
            LOG_WARN(QStringLiteral("同步写入 %1 失败: %2")
                                         .arg(QLatin1String(ProjectFilesManager::kArchiveCoreFile), err));
        }
        m_coreFileDirtyForArchive = false;
    }

    LOG_INFO(QStringLiteral("归档已同步 (防抖写入): %1").arg(m_projectPath));
}

void ProjectData::updateMetadata(const QJsonObject &meta, bool markDirty)
{
    m_filesManager.setData(meta);
    
    if (markDirty && !m_isDirty) {
        m_isDirty = true;
        emit dirtyStateChanged(true);
    }
    
    emit metadataChanged(meta);
}

// 惰性加载 project_results.json：仅在首次访问 results 时读归档
void ProjectData::ensureResultsLoaded() const
{
    if (m_resultsLoaded) return;
    m_resultsLoaded = true;   // 先置位，防止递归

    if (m_projectPath.isEmpty()) return;

    // 尝试从 tmp 先加载（崩溃恢复）
    const QString tmpPath = tempResultsPath();
    if (!tmpPath.isEmpty() && QFile::exists(tmpPath)) {
        QFile f(tmpPath);
        if (f.open(QIODevice::ReadOnly)) {
            const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
            if (!doc.isNull() && doc.isObject()) {
                m_filesManager.setResultsData(doc.object());
                LOG_INFO(QStringLiteral("从临时目录加载 results"));
                return;
            }
        }
    }

    // 从归档读取
    PlascanArchive archive(m_projectPath);
    if (!archive.isValid()) return;

    QString err;
    // 从归档加载 results（Compact JSON，zip 内层已压缩）
    QByteArray data = archive.readEntry(ProjectFilesManager::kArchiveResultsFile, &err);
    if (data.isEmpty()) {
        // 归档无 results 条目——不是错误
        return;
    }
    const QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isObject()) {
        m_filesManager.setResultsData(doc.object());
        LOG_INFO(
            QStringLiteral("从归档惰性加载 results（%1 字节）").arg(data.size()));
    }
}

void ProjectData::updateConfig(const QJsonObject &config, bool markDirty)
{
    m_configManager.setData(config);

    if (markDirty && !m_isDirty) {
        m_isDirty = true;
        emit dirtyStateChanged(true);
    }
}

bool ProjectData::loadTemporaryMetadata()
{
    bool loaded = false;

    // 尝试从 .plascan_tmp/project_files.json 恢复核心数据
    QString filesPath = tempFilesPath();
    if (!filesPath.isEmpty() && QFile::exists(filesPath)) {
        QFile file(filesPath);
        if (file.open(QIODevice::ReadOnly)) {
            QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
            if (!doc.isNull() && doc.isObject()) {
                m_filesManager.setCoreData(doc.object());
                loaded = true;
            }
        }
    }

    // 尝试从 .plascan_tmp/project_results.json 恢复结果数据
    // （新版以 qCompress 压缩写入；通过首字节区分压缩/明文 JSON）
    QString resultsPath = tempResultsPath();
    if (!resultsPath.isEmpty() && QFile::exists(resultsPath)) {
        QFile file(resultsPath);
        if (file.open(QIODevice::ReadOnly)) {
            const QByteArray bytes = file.readAll();
            QJsonObject resultsObj;
            bool parseOk = false;
            // qCompress 输出首 4 字节为大端 uint32 原始大小，首字节 != '{'
            if (!bytes.isEmpty() && (unsigned char)bytes[0] != '{') {
                const QByteArray uncompressed = qUncompress(bytes);
                if (!uncompressed.isEmpty()) {
                    const QJsonDocument doc = QJsonDocument::fromJson(uncompressed);
                    if (!doc.isNull() && doc.isObject()) {
                        resultsObj = doc.object();
                        parseOk = true;
                    }
                }
            }
            if (!parseOk) {
                const QJsonDocument doc = QJsonDocument::fromJson(bytes);
                if (!doc.isNull() && doc.isObject()) {
                    resultsObj = doc.object();
                    parseOk = true;
                }
            }
            if (parseOk) {
                m_filesManager.setResultsData(resultsObj);
                m_resultsLoaded = true;
                loaded = true;
            }
        }
    }

    // 尝试从 .plascan_tmp/project_config.json 恢复配置数据
    QString configPath = tempConfigPath();
    if (!configPath.isEmpty() && QFile::exists(configPath)) {
        QFile file(configPath);
        if (file.open(QIODevice::ReadOnly)) {
            QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
            if (!doc.isNull() && doc.isObject()) {
                updateConfig(ProjectConfigManager::mergeWithDefaults(doc.object()), false);
                loaded = true;
            }
        }
    }

    return loaded;
}

bool ProjectData::saveTemporaryMetadata()
{
    QString filesPath = tempFilesPath();
    QString configPath = tempConfigPath();
    if (filesPath.isEmpty() || configPath.isEmpty())
        return false;

    QDir().mkpath(QFileInfo(filesPath).absolutePath());

    // 写核心数据（Compact JSON）
    QFile filesFile(filesPath);
    if (!filesFile.open(QIODevice::WriteOnly))
        return false;
    filesFile.write(QJsonDocument(m_filesManager.coreData()).toJson(QJsonDocument::Compact));
    filesFile.close();

    // 写结果数据（qCompress 压缩的 Compact JSON，比原始 JSON 小 60-70%）
    if (m_resultsLoaded) {
        QString resultsPath = tempResultsPath();
        if (!resultsPath.isEmpty()) {
            QFile resultsFile(resultsPath);
            if (resultsFile.open(QIODevice::WriteOnly)) {
                const QByteArray json =
                    QJsonDocument(m_filesManager.resultsData()).toJson(QJsonDocument::Compact);
                resultsFile.write(qCompress(json, 1));  // level 1 = 最快压缩
                resultsFile.close();
            }
        }
    }

    // 写配置数据（Compact JSON）
    QFile configFile(configPath);
    if (!configFile.open(QIODevice::WriteOnly))
        return false;
    configFile.write(QJsonDocument(m_configManager.data()).toJson(QJsonDocument::Compact));
    configFile.close();

    return true;
}

void ProjectData::clearTemporaryMetadata()
{
    auto removeIfExists = [](const QString &p) {
        if (!p.isEmpty() && QFile::exists(p)) QFile::remove(p);
    };
    removeIfExists(tempFilesPath());
    removeIfExists(tempResultsPath());
    removeIfExists(tempConfigPath());
}

bool ProjectData::hasTemporaryMetadata() const
{
    auto exists = [](const QString &p) { return !p.isEmpty() && QFile::exists(p); };
    return exists(tempFilesPath()) || exists(tempResultsPath()) || exists(tempConfigPath());
}

bool ProjectData::addImages(const QStringList &imagePaths, QString *errorMsg)
{
    if (m_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    // 获取现有影像列表，准备追加
    QJsonArray images = m_filesManager.data().value("images").toArray();

    // 构建已有路径集合用于去重
    QSet<QString> existingPaths;
    existingPaths.reserve(images.size());
    for (const QJsonValue &val : images) {
        const QString p = val.toObject().value("path").toString();
        if (!p.isEmpty())
            existingPaths.insert(p);
    }

    int skipped = 0;
    for (const QString &srcPath : imagePaths) {
        const QString absPath = QFileInfo(srcPath).absoluteFilePath();

        // 跳过已存在的重复图片
        if (existingPaths.contains(absPath)) {
            ++skipped;
            continue;
        }
        existingPaths.insert(absPath); // 防止同批次内重复

        // 构建影像条目 JSON 对象：
        //   path:     影像绝对路径（引用型，不复制原始文件）
        //   type:     "reference" 表示外部引用（不打包到归档）
        //   added_at: UTC 时间戳，方便排序和溯源
        QJsonObject imgObj;
        imgObj["path"] = absPath;
        imgObj["type"] = "reference";
        imgObj["added_at"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        images.append(imgObj);
    }

    if (skipped > 0 && errorMsg) {
        *errorMsg = QStringLiteral("已跳过 %1 张重复图片").arg(skipped);
    }

    // 更新元数据并写入临时缓存
    QJsonObject core = m_filesManager.coreData();
    core["images"] = images;
    m_filesManager.setCoreData(core);
    if (!m_isDirty) { m_isDirty = true; emit dirtyStateChanged(true); }
    emit metadataChanged(m_filesManager.data());
    saveTemporaryMetadata();

    return true;
}

bool ProjectData::addImagesFromFolder(const QString &folderPath, QString *errorMsg)
{
    QDir dir(folderPath);
    if (!dir.exists()) {
        if (errorMsg) *errorMsg = QStringLiteral("文件夹不存在");
        return false;
    }

    // 支持的影像格式过滤器（包含大小写两种，兼容 Linux/Windows 文件系统）
    QStringList filters;
    filters << "*.tif" << "*.tiff" << "*.TIF" << "*.TIFF"
            << "*.png" << "*.PNG"
            << "*.jpg" << "*.jpeg" << "*.JPG" << "*.JPEG";
    
    // 递归只查找当前目录（QDir::Files，不含子目录）
    QStringList imagePaths;
    for (const QFileInfo &fi : dir.entryInfoList(filters, QDir::Files)) {
        imagePaths << fi.absoluteFilePath();
    }

    // 委托给 addImages 完成实际添加逻辑
    return addImages(imagePaths, errorMsg);
}

bool ProjectData::removeResource(const QString &resourcePath)
{
    return removeResources(QStringList() << resourcePath);
}

bool ProjectData::removeResources(const QStringList &resourcePaths)
{
    QJsonArray images = m_filesManager.coreData().value("images").toArray();
    QJsonArray newImages;

    for (const QJsonValue &val : images) {
        QJsonObject obj = val.toObject();
        if (!resourcePaths.contains(obj.value("path").toString()))
            newImages.append(val);
    }

    QJsonObject core = m_filesManager.coreData();
    core["images"] = newImages;
    m_filesManager.setCoreData(core);
    if (!m_isDirty) { m_isDirty = true; emit dirtyStateChanged(true); }
    emit metadataChanged(m_filesManager.data());
    saveTemporaryMetadata();
    return true;
}

bool ProjectData::setImageCamera(const QString &imagePath, const QJsonObject &cameraMeta, QString *errorMsg)
{
    QMap<QString, QJsonObject> one;
    one.insert(imagePath, cameraMeta);
    return setImageCameras(one, nullptr, errorMsg);
}

bool ProjectData::setImageCameras(const QMap<QString, QJsonObject> &cameraMetaByImage,
                                  int *updatedCount,
                                  QString *errorMsg)
{
    if (updatedCount) *updatedCount = 0;

    if (m_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    if (cameraMetaByImage.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有可写入的相机元数据");
        return false;
    }

    // 预处理：将输入的所有键路径规范化为 cleanPath + absoluteFilePath，
    // 避免不同表示方式（相对路径 vs 绝对路径、双斜线等）导致匹配失败
    QMap<QString, QJsonObject> normalizedMap;
    for (auto it = cameraMetaByImage.constBegin(); it != cameraMetaByImage.constEnd(); ++it) {
        normalizedMap.insert(QDir::cleanPath(QFileInfo(it.key()).absoluteFilePath()), it.value());
    }

    // 遍历影像数组，对匹配的条目写入 camera 字段
    QJsonObject meta = m_filesManager.data();
    QJsonArray images = meta.value("images").toArray();
    int changed = 0;

    for (int i = 0; i < images.size(); ++i) {
        if (!images[i].isObject()) continue;
        QJsonObject imgObj = images[i].toObject();
        // 同样规范化影像路径后再查找
        const QString imgPath = QDir::cleanPath(QFileInfo(imgObj.value("path").toString()).absoluteFilePath());
        auto it = normalizedMap.constFind(imgPath);
        if (it == normalizedMap.constEnd()) continue;

        // 写入相机参数到影像对象的 camera 字段
        imgObj["camera"] = it.value();
        images[i] = imgObj;
        ++changed;
    }

    if (changed <= 0) {
        if (errorMsg) *errorMsg = QStringLiteral("未找到可匹配的影像记录");
        return false;
    }

    meta["images"] = images;
    updateMetadata(meta);   // 更新内存并发出 metadataChanged 信号

    // 延迟写入 .plascan 归档（防抖 2s），避免逐次打开 ZIP
    m_coreFileDirtyForArchive = true;
    m_archiveSyncTimer->start(2000);
    saveTemporaryMetadata();   // 保留一次临时写入用于崩溃恢复

    if (updatedCount) *updatedCount = changed;
    return true;
}

// ---------- 清除相机参数 ----------

bool ProjectData::clearImageCameras(const QStringList &imagePaths,
                                    int *clearedCount,
                                    QString *errorMsg)
{
    if (clearedCount) *clearedCount = 0;

    if (m_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    if (imagePaths.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有指定要清除的影像");
        return false;
    }

    // 规范化路径
    QSet<QString> normalizedSet;
    for (const QString &p : imagePaths)
        normalizedSet.insert(QDir::cleanPath(QFileInfo(p).absoluteFilePath()));

    QJsonObject core = m_filesManager.coreData();
    QJsonArray images = core.value("images").toArray();
    int cleared = 0;

    for (int i = 0; i < images.size(); ++i) {
        if (!images[i].isObject()) continue;
        QJsonObject imgObj = images[i].toObject();
        const QString imgPath = QDir::cleanPath(QFileInfo(imgObj.value("path").toString()).absoluteFilePath());
        if (!normalizedSet.contains(imgPath)) continue;

        imgObj.remove(QStringLiteral("camera"));
        images[i] = imgObj;
        ++cleared;
    }

    if (cleared <= 0) {
        if (errorMsg) *errorMsg = QStringLiteral("未找到可匹配的影像记录");
        return false;
    }

    core["images"] = images;
    m_filesManager.setCoreData(core);
    if (!m_isDirty) { m_isDirty = true; emit dirtyStateChanged(true); }
    emit metadataChanged(m_filesManager.data());

    if (!m_projectPath.isEmpty()) {
        PlascanArchive archive(m_projectPath);
        if (archive.isValid()) {
            QString err;
            if (!archive.writeEntry(ProjectFilesManager::kArchiveCoreFile,
                                    QJsonDocument(core).toJson(QJsonDocument::Compact), &err)) {
                LOG_WARN(QStringLiteral("写入 %1 失败: %2")
                                             .arg(ProjectFilesManager::kArchiveCoreFile, err));
                saveTemporaryMetadata();
            } else {
                saveTemporaryMetadata();
            }
        } else {
            LOG_WARN(QStringLiteral("无法打开项目归档"));
            saveTemporaryMetadata();
        }
    } else {
        saveTemporaryMetadata();
    }

    if (clearedCount) *clearedCount = cleared;
    return true;
}

bool ProjectData::appendIntersectionResult(const QJsonObject &result, QString *errorMsg)
{
    if (m_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    ensureResultsLoaded();   // 确保内存中已有历史数据
    QJsonObject results = m_filesManager.resultsData();
    QJsonArray arr = results.value(QLatin1String("intersection_results")).toArray();
    arr.append(result);
    results[QLatin1String("intersection_results")] = arr;
    m_filesManager.setResultsData(results);
    m_filesManager.clearResultsDirty();
    if (!m_isDirty) { m_isDirty = true; emit dirtyStateChanged(true); }

    PlascanArchive archive(m_projectPath);
    if (archive.isValid()) {
        QString err;
        if (!archive.writeEntry(ProjectFilesManager::kArchiveResultsFile,
                                QJsonDocument(results).toJson(QJsonDocument::Compact), &err)) {
            if (errorMsg) *errorMsg = QStringLiteral("写入 %1 失败: %2")
                                          .arg(ProjectFilesManager::kArchiveResultsFile, err);
            saveTemporaryMetadata();
            return false;
        }
    } else {
        saveTemporaryMetadata();
        if (errorMsg) *errorMsg = QStringLiteral("无法打开项目归档，已写入临时元数据");
        return false;
    }
    return true;
}

QJsonArray ProjectData::getIntersectionResults() const
{
    ensureResultsLoaded();
    return m_filesManager.resultsData().value(QLatin1String("intersection_results")).toArray();
}

bool ProjectData::appendBundleAdjustResult(const QJsonObject &result, QString *errorMsg)
{
    if (m_projectPath.isEmpty()) {
        if (errorMsg) *errorMsg = QStringLiteral("没有打开的项目");
        return false;
    }

    ensureResultsLoaded();
    QJsonObject results = m_filesManager.resultsData();
    QJsonArray arr = results.value(QLatin1String("bundle_adjust_results")).toArray();
    arr.append(result);
    results[QLatin1String("bundle_adjust_results")] = arr;
    m_filesManager.setResultsData(results);
    m_filesManager.clearResultsDirty();
    if (!m_isDirty) { m_isDirty = true; emit dirtyStateChanged(true); }

    PlascanArchive archive(m_projectPath);
    if (archive.isValid()) {
        QString err;
        if (!archive.writeEntry(ProjectFilesManager::kArchiveResultsFile,
                                QJsonDocument(results).toJson(QJsonDocument::Compact), &err)) {
            if (errorMsg) *errorMsg = QStringLiteral("写入 %1 失败: %2")
                                          .arg(ProjectFilesManager::kArchiveResultsFile, err);
            saveTemporaryMetadata();
            return false;
        }
    } else {
        saveTemporaryMetadata();
        if (errorMsg) *errorMsg = QStringLiteral("无法打开项目归档，已写入临时元数据");
        return false;
    }
    return true;
}

QJsonArray ProjectData::getBundleAdjustResults() const
{
    ensureResultsLoaded();
    return m_filesManager.resultsData().value(QLatin1String("bundle_adjust_results")).toArray();
}

bool ProjectData::packResource(const QString &resourcePath, QString *errorMsg)
{
    // TODO: 实现打包外部资源
    if (errorMsg) *errorMsg = QStringLiteral("打包功能未实现");
    return false;
}

QStringList ProjectData::getAllImages() const
{
    return m_filesManager.getAllImages();
}

QStringList ProjectData::getImagesByCategory(const QString &category) const
{
    return m_filesManager.getImagesByCategory(category);
}

QMap<QString, QString> ProjectData::getIpfindOutputMap() const
{
    return m_filesManager.getIpfindOutputMap();
}

QString ProjectData::findMatchFile(const QString &imgA, const QString &imgB) const
{
    return m_filesManager.findMatchFile(imgA, imgB);
}

void ProjectData::saveIpfindSettings(const QJsonObject &settings)
{
    m_configManager.setWorkflowSettings("ipfind", settings);
    updateConfig(m_configManager.data());

    // 立即写入项目配置：如果存在已打开项目，则直接将 project_config.json 写入归档，
    // 以避免只写入临时元数据的行为。若未打开项目，则回落为保存到临时目录。
    if (!m_projectPath.isEmpty()) 
    {
        PlascanArchive archive(m_projectPath);
        if (archive.isValid()) 
        {
            QString err;
            QJsonDocument configDoc(m_configManager.data());
            if (!archive.writeEntry("project_config.json", configDoc.toJson(QJsonDocument::Compact), &err)) 
            {
                LOG_WARN(QStringLiteral("保存 ipfind 设置到项目归档失败: %1").arg(err));
                // 写入失败：不回落到临时元数据以避免在无项目状态下产生文件
            }
        } 
        else 
        {
            // 无效归档：记录警告，但不写入临时元数据
            LOG_WARN(QStringLiteral("无法打开项目归档以保存 ipfind 设置"));
        }
    } 
    else 
    {
        // 没有打开项目：按照要求不写入任何东西，使用默认配置即可
        LOG_INFO(QStringLiteral("未保存 ipfind 设置：没有打开的项目，保留默认设置"));
    }
}

QJsonObject ProjectData::loadIpfindSettings() const
{
    return m_configManager.workflowSettings("ipfind");
}

void ProjectData::setIpmatchSettings(const QJsonObject &settings)
{
    m_configManager.setWorkflowSettings("ipmatch", settings);
    updateConfig(m_configManager.data(), false);
}

QJsonObject ProjectData::getIpmatchSettings() const
{
    return m_configManager.workflowSettings("ipmatch");
}

void ProjectData::saveUiSettings(const QJsonObject &settings)
{
    m_configManager.setUiSettings(settings);
    updateConfig(m_configManager.data());
    saveTemporaryMetadata();

    // 如果有打开的项目，立即写入 project_config.json 到归档，避免只保存在内存/临时文件中
    if (!m_projectPath.isEmpty()) {
        PlascanArchive archive(m_projectPath);
        if (archive.isValid()) {
            QString err;
            QJsonDocument configDoc(m_configManager.data());
            if (!archive.writeEntry("project_config.json", configDoc.toJson(QJsonDocument::Compact), &err)) {
                LOG_WARN(QStringLiteral("保存 UI 设置到项目归档失败: %1").arg(err));
            }
        } else {
            LOG_WARN(QStringLiteral("无法打开项目归档以保存 UI 设置"));
        }
    } else {
        // 未打开项目：保持写入临时元数据的行为
        LOG_INFO(QStringLiteral("UI 设置已保存到临时元数据（未打开项目）"));
    }
}

QJsonObject ProjectData::loadUiSettings() const
{
    return m_configManager.uiSettings();
}

void ProjectData::appendIpfindResult(const QString &input, const QString &output, const QJsonObject &settings)
{
    ensureResultsLoaded();
    m_filesManager.appendIpfindResult(input, output, settings);

    // 延迟写入 .plascan 归档（防抖 2s），避免逐次打开 ZIP
    m_resultsDirtyForArchive = true;
    m_archiveSyncTimer->start(2000);
    saveTemporaryMetadata();   // 保留临时文件写入用于崩溃恢复
}

void ProjectData::appendIpmatchResult(const QStringList &outputs, const QJsonObject &settings)
{
    ensureResultsLoaded();
    m_filesManager.appendIpmatchResult(outputs, settings);

    // 延迟写入 .plascan 归档（防抖 2s），避免逐次打开 ZIP
    m_resultsDirtyForArchive = true;
    m_archiveSyncTimer->start(2000);
    // 注：不在此调用 saveTemporaryMetadata（每对调用一次过于频繁），定时器触发时统一处理
}

// 辅助方法实现 — 通过 ProjectIO 解耦路径计算逻辑

// 返回项目根目录（即 .plascan 文件所在目录）
QString ProjectData::projectDir() const
{
    return ProjectIO::projectRootFromPlascan(m_projectPath);
}

// 返回临时 project_files.json 的完整磁盘路径（.plascan_tmp/project_files.json）
QString ProjectData::tempFilesPath() const
{
    return ProjectIO::tempFilesPath(m_projectPath);
}

// 返回临时 project_config.json 的完整磁盘路径（.plascan_tmp/project_config.json）
QString ProjectData::tempConfigPath() const
{
    return ProjectIO::tempConfigPath(m_projectPath);
}

// 返回临时 project_results.json 的完整磁盘路径（.plascan_tmp/project_results.json）
QString ProjectData::tempResultsPath() const
{
    return ProjectIO::tempResultsPath(m_projectPath);
}

// 返回 assets/ 目录路径
QString ProjectData::assetsDir() const
{
    return ProjectIO::projectAssetsDir(m_projectPath);
}

// 返回 assets/images/ 目录路径
QString ProjectData::imagesDir() const
{
    return ProjectIO::projectImagesDir(m_projectPath);
}
