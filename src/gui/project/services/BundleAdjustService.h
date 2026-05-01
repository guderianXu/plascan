// =============================================================================
// 文件名: BundleAdjustService.h
// 描述:   光束法平差（Bundle Adjustment）服务层。
//
//         本服务负责将光束法平差的纯算法逻辑与 GUI 层（ProjectManager）解耦。
//         调用方（ProjectManager）负责：
//           1. 从项目元数据中加载相机参数 → xjw::Camera 列表
//           2. 从 sidecar JSON 匹配文件构建 BA 轨迹 → xjw::BATrack 列表
//           3. 调用 BundleAdjustService::run() 执行平差
//           4. 将返回的 pendingCamUpdates 通过 ProjectData 写回项目
//
//         BundleAdjustService::run() 本身不依赖 QWidget / QMessageBox，
//         便于未来在无头（headless）环境中复用。
// =============================================================================
#pragma once

#include "BundleAdjust.h"
#include "Camera.h"

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

#include <vector>

namespace xjw
{
namespace gui
{

// ──────────────────────────────────────────────────────────────────────────────
// BaServiceOptions  — 光束法平差服务输入选项
// ──────────────────────────────────────────────────────────────────────────────
struct BaServiceOptions
{
    // ── 匹配 BA 求解器参数 ─────────────────────────────────────────────────
    xjw::BAOptions      baOpt;              ///< BA 求解器配置（迭代次数、鲁棒核、收敛阈值等）

    // ── 输入数据 ───────────────────────────────────────────────────────────
    QStringList         imagePathByIndex;   ///< 与 cameras 列表一一对应的影像绝对路径
    QStringList         selectedImages;     ///< 用户选中的全部影像路径（用于输出 JSON）
    QMap<QString, QJsonObject> beforeCamMeta; ///< 平差前各影像的相机 JSON，用于度量位移

    // ── 输出目录 ───────────────────────────────────────────────────────────
    QString             outputDir;          ///< 所有输出文件的根目录

    // ── 执行控制 ───────────────────────────────────────────────────────────
    bool                dryRun = false;     ///< 仅统计轨迹数量，不执行实际优化与文件写出
    int                 threads = 4;        ///< 预留字段（供后续并行 BA 使用）

    // ── 文件导出标志 ───────────────────────────────────────────────────────
    bool    exportTsai       = true;        ///< 是否导出平差后的 .tsai 相机文件
    bool    exportSummaryTxt = true;        ///< 是否生成文字摘要报告
    bool    exportPointsCsv  = true;        ///< 是否导出点位精度 CSV
    bool    exportCameraCsv  = true;        ///< 是否导出相机位移 CSV
    bool    exportRunJson    = true;        ///< 是否将完整结果写入 JSON
    bool    exportEvalPlot   = true;        ///< 是否生成评估对比图（PNG）
};

// ──────────────────────────────────────────────────────────────────────────────
// BaServiceResult  — 光束法平差服务输出结果
// ──────────────────────────────────────────────────────────────────────────────
struct BaServiceResult
{
    bool    success  = false;               ///< 是否成功完成平差
    QString errorMessage;                   ///< 失败时的错误描述

    // 平差后相机 JSON 映射，键为影像绝对路径。
    // 由调用方决定是否通过 ProjectData::setImageCameras() 提交到项目。
    QMap<QString, QJsonObject> pendingCamUpdates;

    // 完整的 BA 运行结果 JSON（含统计信息、文件路径列表等），
    // 可直接作为预览数据发给 GUI 展示。
    QJsonObject         resultJson;
};

// ──────────────────────────────────────────────────────────────────────────────
// BundleAdjustService  — 光束法平差服务（无状态静态方法）
// ──────────────────────────────────────────────────────────────────────────────
class BundleAdjustService
{
public:
    // 执行完整的光束法平差流程：
    //   1. 若 dryRun=true，只统计轨迹数量并直接返回
    //   2. 调用 xjw::BundleAdjust::optimizePoints 执行非线性最小二乘
    //   3. 生成各类输出文件（tsai / csv / txt / json / png 评估图）
    //   4. 将平差后相机 JSON 写入 result.pendingCamUpdates
    //      （不直接写回项目数据，由调用方决定是否应用）
    //
    // @param cameras   预先从项目 JSON 加载的相机列表（与 opts.imagePathByIndex 一一对应）
    // @param tracks    预先构建的 BA 轨迹列表（含初始三角化点 + 观测）
    // @param opts      平差选项（见 BaServiceOptions 各字段说明）
    // @return          BaServiceResult（含平差结果 JSON 与待提交相机参数）
    static BaServiceResult run(
        const std::vector<xjw::Camera>&  cameras,
        std::vector<xjw::BATrack>&       tracks,
        const BaServiceOptions&          opts
    );
};

} // namespace gui
} // namespace xjw
