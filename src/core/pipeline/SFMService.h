// =============================================================================
// 文件名: SFMService.h
// 描述:   增量式 SFM（运动恢复结构）一站式服务层。
//
//         本服务实现了"傻瓜式"空中三角测量：
//           - 自动检查并补全 DISK 特征提取
//           - 优先复用已有特征匹配（可选自动补全缺失匹配）
//           - 执行增量式 SFM 重建
//           - 回收恢复的相机参数
//
//         调用方只需填写 SFMServiceOptions 并调用 SFMService::run()，
//         即使用户此前未执行过任何特征提取或匹配操作，服务也能完成重建。
//
//         SFMService::run() 本身不依赖 QWidget / QMessageBox，
//         并且不直接访问 ProjectData / ProjectManager，
//         便于未来在无头（headless）环境中复用。
//         调用方负责根据 SFMServiceResult 更新项目元数据。
// =============================================================================
#pragma once

#include "Camera.h"
#include "pipeline/IncrementalSfm.h"

#include <QJsonObject>
#include <QJsonArray>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVector>
#include <atomic>
#include <functional>
#include <memory>

namespace xjw
{
namespace gui
{

// ──────────────────────────────────────────────────────────────────────────────
// SFMServiceOptions  — SFM 服务输入选项
// ──────────────────────────────────────────────────────────────────────────────
struct SFMServiceOptions
{
    // ── 影像列表 ───────────────────────────────────────────────────────────
    QStringList         images;             ///< 参与 SFM 的影像完整路径列表

    // ── 项目路径（用于查找已有特征 / .match 文件）────────────────────────
    QString             plascanPath;        ///< 当前 .plascan 项目路径

    // ── 项目元数据（备用；目前优先目录扫描）─────────────────────────────
    QJsonObject         projectMeta;        ///< project_files.json 内容

    // ── 输出目录 ───────────────────────────────────────────────────────────
    QString             outputDir;          ///< 稀疏点云等输出文件的根目录

    // ── 精度等级（0=低, 1=中, 2=高, 3=最高）────────────────────────────────
    int                 quality = 3;

    // ── 执行控制 ───────────────────────────────────────────────────────────
    int                 threads = 4;

    /// 仅执行 Phase 1（特征检测）+ Phase 2（特征匹配），不做 SFM 重建。
    /// 用于光束法平差模式：先自动保证特征/匹配存在，再由外部做 BA。
    bool                baOnly  = false;

    // ── 进度回调（在调用线程中同步调用，UI层应使用 Qt::QueuedConnection 转发）────
    /// @param stage    当前阶段名称，例如 "特征提取 3/20"
    /// @param percent  0~100 整数进度
    std::function<void(const QString &stage, int percent)> progressFn;

    // ── 自动化流水线配置 ───────────────────────────────────────────────────
    /// 设备偏好: "auto"（自动检测 CUDA）, "cuda", "cpu"
    QString             device = "auto";
    /// 一键空三默认特征提取算法。默认 DISK，不再使用 SuperPoint。
    QString             featureAlgorithm = QStringLiteral("disk");
    /// 一键空三默认特征匹配算法。默认 LightGlue，不再使用 SuperGlue。
    QString             matchAlgorithm = QStringLiteral("lightglue");
    /// 旧 SuperGlue 模型类型配置，仅保留兼容旧调用方。
    QString             sgModelType = "outdoor";

    /// CUDA 模式下并行处理的影像对数（每对独立持有一个 Matcher 实例）
    /// 默认: 1（串行）；增大前确认 GPU 显存充裕（每实例约占 200-400 MB）
    int                 cudaParallelPairs = 1;

    /// 深度特征提取输入图像最长边限制。
    /// >0: 使用调用方指定初始值；0: 使用质量档位，最高质量档先尝试不缩放；
    /// <0: 从不缩放开始。CUDA OOM 时内部会自动降低尺寸并重试。
    int                 featureMaxImageDim = 0;

    /// 特征点灰度过滤范围。默认过滤近黑背景，保留有效地物纹理。
    float               featureGrayscaleMin = 5.0f / 255.0f;
    float               featureGrayscaleMax = 1.0f;

    /// 若为 true，则 Phase 2 只处理 allowedPairs 中显式给出的影像对。
    /// 若 allowedPairs 为空，则视为没有可用配对约束并直接失败。
    bool                restrictPairs = false;

    /// 允许参与匹配的影像对集合。每项为按路径规范化后、稳定排序的 pair key。
    /// key 格式: minPath + "\n" + maxPath。
    QStringList         allowedPairs;

    /// 已知相机文件与影像一一对应且影像数量较大时，自动只匹配相邻若干张。
    /// 这避免航拍/有序采集数据退化为 O(N^2) 全配对。显式 restrictPairs 优先。
    bool                autoRestrictKnownCameraPairs = true;

    /// 自动配对裁剪的邻域窗口。4 表示第 i 张最多匹配 i+1 到 i+4。
    int                 knownCameraPairWindow = 4;

    /// 已知相机自动配对时，每张影像额外选择的空间最近邻数量。
    /// 与顺序邻域取并集，用于航带换行、跨航带和文件顺序不可靠的数据。
    int                 knownCameraSpatialNeighborCount = 8;

    /// 已知相机自动配对时，优先根据相机足迹重叠生成候选对。
    /// 成功时会替代仅按顺序/相机中心距离的裁剪，减少无重叠影像的无效匹配。
    bool                useKnownCameraOverlapPairs = true;

    /// 相机足迹重叠的邻域宽松系数。值越大，候选对越多。
    double              knownCameraOverlapNeighborFactor = 2.0;

    /// 影像数大于该阈值时才启用已知相机自动配对裁剪，小数据集保留全配对。
    int                 knownCameraAllPairsMaxImages = 20;

    /// 每完成一对影像的匹配后立即调用此回调（在工作线程中调用）
    /// 可用于实时更新 UI（调用方需通过 Qt::QueuedConnection 转回主线程）
    /// img0/img1: 影像绝对路径; matchPath: .match 文件路径; numMatches: 内点数
    std::function<void(const QString &img0, const QString &img1,
                       const QString &matchPath, int numMatches)> pairMatchedFn;

    /// 是否自动补全缺失匹配对。
    /// true: 对缺失匹配自动调用配置的匹配器生成；false: 仅复用已有 .match 文件。
    bool                autoGenerateMissingMatches = true;

    /// 取消标志（由调用方提供的原子标志，置 true 则流水线尽早中止）
    std::shared_ptr<std::atomic<bool>> cancelFlag;

    // ── 可选：用户提供的相机内参（优先于自动估算）───────────────────────
    /// 若 > 0 则使用用户提供的值；否则自动估算
    double              userFu = 0.0;
    double              userFv = 0.0;
    double              userCu = 0.0;
    double              userCv = 0.0;
    /// 像元大小 (mm)，结合焦距(mm) 可计算焦距(px): f_px = f_mm / pitch
    double              userPitch = 0.0;

    // ── 可选：相机文件路径列表（与 images 一一对应）─────────────────────
    /// 每项为对应影像的 .tsai 相机文件路径；为空则跳过该影像的文件加载。
    /// 当某影像同时有 cameraPaths 和 userFu/userFv 时，优先使用 cameraPaths。
    QStringList         cameraPaths;
};

// ──────────────────────────────────────────────────────────────────────────────
// SpFileRecord  — 服务自动生成的单张影像特征文件记录
// ──────────────────────────────────────────────────────────────────────────────
struct SpFileRecord
{
    QString imagePath;              ///< 原始影像路径
    QString spPath;                 ///< 生成的特征文件路径（字段名保留兼容旧调用方）
};

// ──────────────────────────────────────────────────────────────────────────────
// FailedPairRecord  — 影像对匹配失败（内点数不足）的记录
// ──────────────────────────────────────────────────────────────────────────────
struct FailedPairRecord
{
    QString     imagePath0;         ///< image0 的完整路径
    QString     imagePath1;         ///< image1 的完整路径
};

// ──────────────────────────────────────────────────────────────────────────────
// MatchFileRecord  — 服务自动生成的单对影像匹配推记录
// ──────────────────────────────────────────────────────────────────────────────
struct MatchFileRecord
{
    QString     pairName;           ///< 匹配对名称 (baseA__baseB)
    QString     matchPath;          ///< 生成的 .match 文件路径
    QString     sidecarPath;        ///< sidecar JSON 路径
    QJsonObject settings;           ///< 匹配设置（适用于 appendIpmatchResult）
};

// ──────────────────────────────────────────────────────────────────────────────
// SFMServiceResult  — SFM 服务输出结果
// ──────────────────────────────────────────────────────────────────────────────
struct SFMServiceResult
{
    bool    success = false;                ///< 是否成功完成重建
    QString errorMessage;                   ///< 失败时的错误描述
    QString summary;                        ///< 可读摘要（成功或失败均有）

    /// baOnly 模式下：特征/匹配已准备完毕，BA 可以运行（此时 success=false, SFM 未执行）
    bool    featureMatchesReady = false;

    int     numRegisteredImages = 0;        ///< 已注册图像数
    int     numPoints3D = 0;                ///< 三维点数
    double  meanReprojError = 0.0;          ///< 平均重投影误差（像素）

    /// 重建后相机 JSON 映射，键为影像绝对路径。
    QMap<QString, QJsonObject> pendingCamUpdates;

    /// 稀疏点云输出路径（若成功则非空）
    QString sparseCloudPath;

    // ── 光束法平差统计（供报告使用） ──────────────────────────────────────
    double  baRmsBefore        = 0.0;  ///< BA 优化前平均重投影 RMS（px）
    double  baRmsAfter         = 0.0;  ///< BA 优化后平均重投影 RMS（px）
    int     baTracksTotal      = 0;    ///< 参与 BA 的总轨迹数
    int     baTracksOptimized  = 0;    ///< 成功优化的轨迹数
    int     baTracksFiltered   = 0;    ///< 被过滤的离群点数
    double  durationSeconds    = -1.0; ///< 总耗时（秒）
    /// 逐相机残差列表，每项: {path, registered, residual_px}
    QJsonArray perCameraResiduals;

    // ── 自动生成的中间文件记录（供调用方更新项目元数据）───────────────────
    QVector<SpFileRecord>    newSpFiles;     ///< 自动提取的特征文件（字段名保留兼容旧调用方）
    QVector<MatchFileRecord> newMatchFiles;  ///< 自动生成的匹配文件

    /// 本次运行中内点数不足（无有效匹配）的影像对
    /// 已记录到 matchDir/no_match_pairs.json，下次运行时自动跳过（除非特征文件更新）
    QVector<FailedPairRecord> failedPairs;
};

// ──────────────────────────────────────────────────────────────────────────────
// SFMService  — 一站式增量式 SFM 服务（无状态静态方法）
// ──────────────────────────────────────────────────────────────────────────────
class SFMService
{
public:
    /// 执行完整的增量式 SFM 全自动流程（同步阻塞，应在后台线程中调用）：
    ///
    ///   Phase 1 — 确保特征:
    ///     检查每张影像是否已有对应算法特征文件，缺失的自动提取。
    ///
    ///   Phase 2 — 确保匹配:
    ///     优先复用已有 .match 文件；当 autoGenerateMissingMatches=true 时，
    ///     对缺失配对自动调用配置的匹配器补齐。
    ///
    ///   Phase 3 — SFM 重建:
    ///     将特征点和匹配传入 IncrementalSfm 执行增量式重建。
    ///
    ///   Phase 4 — 结果收集:
    ///     恢复的相机参数序列化为 JSON，导出稀疏点云。
    ///
    /// @param opts  输入选项
    /// @return      SFMServiceResult（含相机参数 + 自动生成的文件清单）
    static SFMServiceResult run(const SFMServiceOptions &opts);
};

} // namespace gui
} // namespace xjw
