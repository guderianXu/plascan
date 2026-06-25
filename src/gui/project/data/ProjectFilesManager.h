// =============================================================================
// 文件名: ProjectFilesManager.h
// 描述:   管理项目文件元数据的内存模型与访问接口。
//
//         数据被拆分为两个独立的 JSON 文件（对应归档中的两个条目）：
//
//         【project_files.json】—— 轻量核心数据，每次打开项目都会立即加载
//           - images[]：所有影像的路径、类型、相机参数等（常用，必须快速可用）
//
//         【project_results.json】—— 惰性加载的结果数据，仅在需要时读取
//           - *_results[]：特征、匹配、稀疏/密集重建、模型、DEM/DOM 等产物记录
//
//         拆分动机：
//           ipmatch_results 对 N 张影像最多有 C(N,2) 条（例如 1000 张 ≈ 50 万条），
//           若与 images 混放在同一文件，将导致开项目时需要解析整个大文件，严重拖慢加载。
//
//         本类主要负责内存模型。appendIpmatchResult 为提取匹配数量会读取 sidecar/二进制文件；
//         project_files.json / project_results.json 的持久化由 ProjectData 通过 PlascanArchive 完成。
// =============================================================================
#pragma once

#include <QJsonObject>
#include <QJsonArray>
#include <QMap>
#include <QStringList>

class ProjectFilesManager
{
public:
    /// 归档中两个 JSON 条目的名称常量
    static constexpr const char *kArchiveCoreFile        = "project_files.json";
    static constexpr const char *kArchiveResultsFile    = "project_results.json";
    static constexpr const char *kArchiveCborResultsFile = "project_results.cbor"; ///< 预留 CBOR 结果条目（当前未读写）
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
    /// 返回空的默认 core 对象（project_files.json 初始内容）
    static QJsonObject defaultFiles();
    /// 返回空的默认 results 对象（project_results.json 初始内容）
    static QJsonObject defaultResults();

    // ── 查询接口 ─────────────────────────────────────────────────────────
    QStringList getAllImages() const;
    QStringList getImagesByCategory(const QString &category) const;
    QMap<QString, QString> getIpfindOutputMap() const;
    /// 在 ipmatch_results 中查找两张影像对应的匹配文件路径（支持 A/B 两种顺序）
    QString findMatchFile(const QString &imgA, const QString &imgB) const;

    // ── 修改接口 ─────────────────────────────────────────────────────────
    void setImages(const QJsonArray &images);
    void appendIpfindResult(const QString &input, const QString &output, const QJsonObject &settings);
    /// 追加 ipmatch 结果记录（精简格式：仅存路径，去掉冗余 sp0/sp1/pair_name）
    void appendIpmatchResult(const QStringList &outputs, const QJsonObject &settings);

private:
    // 核心数据（images 数组）
    QJsonObject _coreFiles;
    // 结果数据（*_results 结果数组）
    QJsonObject _resultFiles;
    // 结果数据脏标记
    bool _resultsDirty = false;

};
