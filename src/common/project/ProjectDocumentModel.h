// =============================================================================
// 文件名: ProjectDocumentModel.h
// 描述:   管理项目文件元数据的内存模型与访问接口。
//
//         数据在内存中拆分为两个对象，对应 Chunk doc.json 中的两个字段：
//
//         【project_files】—— 轻量核心数据，每次打开项目都会立即加载
//           - images[]：所有影像的路径、类型、相机参数等（常用，必须快速可用）
//
//         【project_results】—— 惰性加载的结果数据，仅在需要时读取
//           - *_results[]：逐影像匹配、稀疏/密集重建、模型、DEM/DOM 等产物记录
//
//         拆分动机：
//           image_match_results 对 N 张影像固定最多 N 条；每条记录指向该影像唯一
//           `.pimatch` 分片，并列出邻接影像，避免工程元数据随像对数平方增长。
//
//         本类只维护轻量索引，不解析匹配二进制；`.pimatch` 的格式校验统一由
//         image_matching::ImageMatchFile 负责。
// =============================================================================
#pragma once

#include <QJsonObject>
#include <QJsonArray>
#include <QMap>
#include <QStringList>
#include <QVector>

struct ProjectImageMatchResultRecord
{
    QString image;
    QString output;
    QStringList neighbors;
    QJsonObject settings;
};

class ProjectFilesManager
{
public:
    ProjectFilesManager() = default;

    // ── 核心数据（images）── 项目打开时立即可用 ──────────────────────────
    QJsonObject coreData() const { return _coreFiles; }
    void setCoreData(const QJsonObject &data) { _coreFiles = data; }

    // ── 结果数据（*_results）── 惰性加载，仅在需要时填充 ─────────────────
    QJsonObject resultsData() const { return _resultFiles; }
    void setResultsData(const QJsonObject &data);

    /// 结果数据是否已被修改（尚未持久化）
    bool isResultsDirty() const { return _resultsDirty; }
    void clearResultsDirty() { _resultsDirty = false; }

    // ── 兼容接口（返回 core + results 合并视图，仅在需要全量数据时使用）──
    QJsonObject data() const;
    /// 将整体 JSON（旧格式）拆分写入 core/results 两个内部对象（迁移用）
    void setData(const QJsonObject &data);
    // 判断某个 key 是否属于 results 域
    static bool isResultKey(const QString &key);

    // ── 默认结构 ─────────────────────────────────────────────────────────
    /// 返回空的默认 project_files 对象
    static QJsonObject defaultFiles();
    /// 返回空的默认 project_results 对象
    static QJsonObject defaultResults();

    // ── 查询接口 ─────────────────────────────────────────────────────────
    QStringList getAllImages() const;
    QStringList getImagesByCategory(const QString &category) const;
    /// 返回“影像路径 -> 逐影像匹配分片”的索引。
    QMap<QString, QString> getImageMatchOutputMap() const;
    /// 在 image_match_results 中查找包含指定像对的逐影像分片。
    QString findMatchFile(const QString &imgA, const QString &imgB) const;

    // ── 修改接口 ─────────────────────────────────────────────────────────
    void setImages(const QJsonArray &images);
    /// 按 owner 影像覆盖写入分片索引；同一影像始终只有一条当前记录。
    void appendImageMatchResult(const ProjectImageMatchResultRecord &record);
    void appendImageMatchResults(const QVector<ProjectImageMatchResultRecord> &records);

private:
    // 核心数据（images 数组）
    QJsonObject _coreFiles;
    // 结果数据（*_results 结果数组）
    QJsonObject _resultFiles;
    // 结果数据脏标记
    bool _resultsDirty = false;

};
