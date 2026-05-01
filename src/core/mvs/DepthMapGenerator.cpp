// =============================================================================
// 文件: DepthMapGenerator.cpp
// 模块: MVS - Qt 封装的深度图生成 + COLMAP BFS 融合
// =============================================================================

#include "DepthMapGenerator.h"
#include "DenseCloudBuilder.h"
#include "EpipolarRectifier.h"
#include "Logger.h"
#include <QtConcurrent/QtConcurrent>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <thread>
#include <mutex>
#include <deque>
#include <chrono>

namespace xjw
{
namespace mvs
{

namespace
{

bool writeCvMatStorage(const std::string &path, const cv::Mat &matrix, std::string *errorMsg)
{
    cv::FileStorage storage(path, cv::FileStorage::WRITE);
    if (!storage.isOpened())
    {
        if (errorMsg)
        {
            *errorMsg = "无法写入矩阵文件: " + path;
        }
        return false;
    }

    storage << "mat" << matrix;
    storage.release();
    return true;
}

} // namespace

// =============================================================================
// 辅助：旋转矩阵行列式
// =============================================================================
static float det3(const float *R)
{
    return R[0] * (R[4] * R[8] - R[5] * R[7])
           - R[1] * (R[3] * R[8] - R[5] * R[6])
           + R[2] * (R[3] * R[7] - R[4] * R[6]);
}

// =============================================================================
/**
 * @brief 默认构造深度图生成器。
 *
 * 仅初始化 Qt 元类型注册；实际数据通过 `setViews()`、`setSparseCloud()`
 * 与 `setConfig()` 注入。
 */
DepthMapGenerator::DepthMapGenerator(QObject *parent)
    : QObject(parent)
{
    qRegisterMetaType<DepthFrameResult>("DepthFrameResult");
    qRegisterMetaType<QSharedPointer<cv::Mat>>("QSharedPointer<cv::Mat>");
    qRegisterMetaType<std::vector<DensePoint>>("std::vector<DensePoint>");
}

/**
 * @brief 使用输入视图、稀疏点云与配置构造深度图生成器。
 */
DepthMapGenerator::DepthMapGenerator(const std::vector<CameraView> &views,
                                     const PreprocessResult        &ppResult,
                                     const DepthGenConfig          &config,
                                     QObject                       *parent)
    : QObject(parent)
    , m_views(views)
    , m_sparse(ppResult.cloud)
    , m_config(config)
{
    qRegisterMetaType<DepthFrameResult>("DepthFrameResult");
    qRegisterMetaType<QSharedPointer<cv::Mat>>("QSharedPointer<cv::Mat>");
    qRegisterMetaType<std::vector<DensePoint>>("std::vector<DensePoint>");
}

DepthMapGenerator::~DepthMapGenerator()
{
}

// =============================================================================
void DepthMapGenerator::setViews(const std::vector<CameraView> &views)
{
    m_views = views;
    m_skipFrameMask.assign(m_views.size(), 0);
}

void DepthMapGenerator::setSparseCloud(const SparseCloud &sparse)
{
    m_sparse = sparse;
}

void DepthMapGenerator::setConfig(const DepthGenConfig &config)
{
    m_config = config;
}

void DepthMapGenerator::setSkippedFrameIndices(const std::vector<int> &indices)
{
    m_skipFrameMask.assign(m_views.size(), 0);
    for (int index : indices)
    {
        if (index >= 0 && index < static_cast<int>(m_skipFrameMask.size()))
        {
            m_skipFrameMask[static_cast<size_t>(index)] = 1;
        }
    }
}

// =============================================================================
// 预加载所有图像到灰度缓存，避免逐帧重复从磁盘读取
// =============================================================================
void DepthMapGenerator::preloadImages()
{
    const int NV = static_cast<int>(m_views.size());
    m_grayCache.resize(NV);
    m_contentMasks.resize(NV);
    for (int i = 0; i < NV; ++i)
    {
        m_grayCache[i] = cv::imread(m_views[i].imagePath, cv::IMREAD_GRAYSCALE);
        if (m_grayCache[i].empty())
        {
            LOG_WARN(QStringLiteral("[MVS] 警告: 无法读取图像 %1: %2").arg(i).arg(QString::fromStdString(m_views[i].imagePath)));
            continue;
        }

        // ── 在 CLAHE 增强之前计算内容区域掩码 ─────────────────────────
        // 关键：必须在原始图像上计算暗区掩码！
        // gamma(0.4) 会将 gray=3 提升到 37，gray=5 提升到 46，
        // 使得本应被屏蔽的黑边像素通过暗区阈值，
        // 导致 PatchMatch 在无纹理黑边区域生成大量噪声深度。
        {
            cv::Mat blurOrig;
            cv::GaussianBlur(m_grayCache[i], blurOrig, cv::Size(15, 15), 0);
            // 原始图像上的暗区阈值：内容区像素通常 > 20，黑边 < 5
            // 使用 Otsu 自动阈值以自适应不同亮度的图像
            cv::Mat otsuBin;
            double otsuThresh = cv::threshold(blurOrig, otsuBin, 0, 255,
                                              cv::THRESH_BINARY | cv::THRESH_OTSU);
            // 取 Otsu 阈值的 30%（更保守）与固定下限(8)中较大的
            int adaptiveThresh = std::max(8, static_cast<int>(otsuThresh * 0.3));
            m_contentMasks[i] = (blurOrig > adaptiveThresh);

            // 形态学闭操作填补内容区域中的小孔洞
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
            cv::morphologyEx(m_contentMasks[i], m_contentMasks[i], cv::MORPH_CLOSE, kernel);

            int contentPx = cv::countNonZero(m_contentMasks[i]);
            int totalPx = m_grayCache[i].rows * m_grayCache[i].cols;
            fprintf(stderr, "[MVS] 图像 %d: 内容掩码 (Otsu=%.0f adaptiveThresh=%d): %d/%d (%.1f%%)\n",
                    i, otsuThresh, adaptiveThresh, contentPx, totalPx, 100.f*contentPx/totalPx);
        }

        // ── 自适应对比度增强 (CLAHE) ──────────────────────────────────
        // 暗场/低对比度图像（均值 < 40）的像素方差极小，NCC 分母接近 0，
        // 导致 PatchMatch 无法区分正确/错误深度假设 → 全图噪声。
        // CLAHE 局部均衡化可在不影响已有纹理的前提下大幅提升暗区对比度。
        double imgMean = cv::mean(m_grayCache[i])[0];
        if (imgMean < 80.0)
        {
            cv::Mat enhanced;
            if (imgMean < 30.0)
            {
                // 极暗图像（航空影像常见）：先做 gamma 校正提升暗区可见度，
                // 再用高 clipLimit CLAHE 进一步增强局部对比度。
                // gamma=0.4 将 [0,255] 非线性映射，使暗部细节显现。
                cv::Mat floatImg;
                m_grayCache[i].convertTo(floatImg, CV_32F, 1.0 / 255.0);
                cv::pow(floatImg, 0.4, floatImg);
                floatImg.convertTo(enhanced, CV_8U, 255.0);
                auto clahe = cv::createCLAHE(8.0, cv::Size(8, 8));
                clahe->apply(enhanced, enhanced);
            }
            else
            {
                // 中等暗度：标准 CLAHE
                auto clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
                clahe->apply(m_grayCache[i], enhanced);
            }
            double newMean = cv::mean(enhanced)[0];
            fprintf(stderr, "[MVS] 图像 %d: 低对比度 (mean=%.1f)，应用 CLAHE → mean=%.1f\n",
                    i, imgMean, newMean);
            m_grayCache[i] = enhanced;
        }
        else
        {
            fprintf(stderr, "[MVS] 预加载图像 %d/%d: %dx%d (mean=%.1f)\n",
                    i+1, NV, m_grayCache[i].cols, m_grayCache[i].rows, imgMean);
        }
    }
}

// =============================================================================
void DepthMapGenerator::start()
{
    m_cancelled = false;
    m_depthFrames.clear();
    (void)QtConcurrent::run([this]()
    {
        runInBackground();
    });
}

// =============================================================================
bool DepthMapGenerator::estimateDepthRange(int refIdx, float &zNear, float &zFar) const
{
    const CameraView &ref = m_views[refIdx];
    PositiveDepthCameraModel cam = ref.positiveDepthModel();

    std::vector<float> depths;
    depths.reserve(m_sparse.points.size());

    for (const auto &pt : m_sparse.points)
    {
        float Zc = cam.R_cw[6]*pt[0] + cam.R_cw[7]*pt[1] + cam.R_cw[8]*pt[2] + cam.T[2];
        if (Zc > 0.f)
        {
            depths.push_back(Zc);
        }
    }

    if (depths.size() < 5)
    {
        // 没有足够的稀疏点——使用全局最大相机基线估算深度范围。
        // 关键：使用「全局最大基线」（所有相机对之间的最大距离），
        // 而非 per-camera 最大基线，以保证所有帧使用一致的 zNear/zFar，
        // 防止深度图均值差异过大导致融合一致性检查全部失败。
        float maxBaseline = 0.f;
        const int NVall = static_cast<int>(m_views.size());
        for (int ia = 0; ia < NVall; ++ia)
        {
            PositiveDepthCameraModel ca = m_views[ia].positiveDepthModel();
            for (int ib = ia + 1; ib < NVall; ++ib)
            {
                PositiveDepthCameraModel cb = m_views[ib].positiveDepthModel();
                float dx = ca.C[0]-cb.C[0], dy = ca.C[1]-cb.C[1], dz = ca.C[2]-cb.C[2];
                maxBaseline = std::max(maxBaseline, std::sqrt(dx*dx + dy*dy + dz*dz));
            }
        }
        if (maxBaseline > 1e-3f)
        {
            // 航空摄影测量：典型场景深度 ≈ 基线的 0.5× ~ 100×
            zNear = maxBaseline * 0.1f;
            zFar  = maxBaseline * 100.f;
            fprintf(stderr, "[MVS] 深度范围回退(全局基线): global_baseline=%.4f → zNear=%.4f zFar=%.4f\n",
                    maxBaseline, zNear, zFar);
        } else {
            // 实在无法估计
            float dx = m_sparse.maxPt[0] - m_sparse.minPt[0];
            float dy = m_sparse.maxPt[1] - m_sparse.minPt[1];
            float dz = m_sparse.maxPt[2] - m_sparse.minPt[2];
            float diag = std::sqrt(dx*dx + dy*dy + dz*dz);
            zNear = 0.1f;
            zFar  = diag > 0 ? diag * 3.f : 100.f;
            fprintf(stderr, "[MVS] 深度范围回退(AABB): diag=%.4f → zNear=%.4f zFar=%.4f\n",
                    diag, zNear, zFar);
        }
        return true;
    }

    std::sort(depths.begin(), depths.end());
    size_t n = depths.size();

    // 使用 IQR (四分位距) 剔除离群深度值，比固定百分位更鲁棒
    float Q1 = depths[n / 4];
    float Q3 = depths[n * 3 / 4];
    float IQR = Q3 - Q1;
    float lowerFence = Q1 - 1.5f * IQR;
    float upperFence = Q3 + 1.5f * IQR;

    // 在 fence 范围内重新取 5%/95% 分位
    std::vector<float> inlierDepths;
    inlierDepths.reserve(n);
    for (float d : depths) {
        if (d >= lowerFence && d <= upperFence)
            inlierDepths.push_back(d);
    }
    if (inlierDepths.size() < 3) inlierDepths = depths; // fallback

    size_t ni = inlierDepths.size();
    zNear = inlierDepths[static_cast<size_t>(ni * 0.02f)] * m_config.zNearScale;
    zFar  = inlierDepths[static_cast<size_t>(ni * 0.98f)] * m_config.zFarScale;
    zNear = std::max(zNear, 0.01f);
    zFar  = std::max(zFar,  zNear + 0.1f);

    fprintf(stderr, "[MVS] 深度范围 IQR: Q1=%.4f Q3=%.4f IQR=%.4f fence=[%.4f, %.4f] inliers=%zu/%zu\n",
            Q1, Q3, IQR, lowerFence, upperFence, inlierDepths.size(), n);
    return true;
}

// =============================================================================
cv::Mat DepthMapGenerator::buildHintDepth(int refIdx, int W, int H) const
{
    if (m_sparse.points.empty()) return cv::Mat();

    const CameraView &ref = m_views[refIdx];
    PositiveDepthCameraModel cam = ref.positiveDepthModel();

    // 先计算深度的 IQR fence，过滤离群稀疏点的深度提示
    std::vector<float> allZc;
    allZc.reserve(m_sparse.points.size());
    for (const auto &pt : m_sparse.points) {
        float Zc = cam.R_cw[6]*pt[0] + cam.R_cw[7]*pt[1] + cam.R_cw[8]*pt[2] + cam.T[2];
        if (Zc > 0.f) allZc.push_back(Zc);
    }
    float depthLo = 0.f, depthHi = 1e30f;
    if (allZc.size() >= 4) {
        std::sort(allZc.begin(), allZc.end());
        float Q1 = allZc[allZc.size() / 4];
        float Q3 = allZc[allZc.size() * 3 / 4];
        float IQR = Q3 - Q1;
        depthLo = Q1 - 1.5f * IQR;
        depthHi = Q3 + 1.5f * IQR;
    }

    cv::Mat hint(H, W, CV_32F, cv::Scalar(0.f));

    // 第一步：将稀疏点投影到 hint 图，每点周围 ±3px 小区域赋深度
    for (const auto &pt : m_sparse.points) {
        float u, v;
        if (!cam.project(pt[0], pt[1], pt[2], u, v)) continue;
        int iu = static_cast<int>(std::round(u));
        int iv = static_cast<int>(std::round(v));
        if (iu < 0 || iu >= W || iv < 0 || iv >= H) continue;

        float Zc = cam.R_cw[6]*pt[0] + cam.R_cw[7]*pt[1] + cam.R_cw[8]*pt[2] + cam.T[2];
        if (Zc <= 0.f || Zc < depthLo || Zc > depthHi) continue;

        for (int dv = -3; dv <= 3; ++dv) {
            for (int du = -3; du <= 3; ++du) {
                int nu = iu+du, nv = iv+dv;
                if (nu<0||nu>=W||nv<0||nv>=H) continue;
                float &h = hint.at<float>(nv, nu);
                if (h == 0.f || Zc < h) h = Zc;
            }
        }
    }

    // 第二步：限距离膨胀——仅将稀疏种子传播到 maxHintRadius 像素范围内
    // 不做全图扫线传播，以免把远离稀疏点的像素也强制初始化为"错误 hint"：
    //   GPU 初始化对有 hint 的像素使用 hint±30% 的窄范围，
    //   若 hint 覆盖了距离真实深度很远的区域，PatchMatch 将无法逃脱。
    // 超出 maxHintRadius 的像素保持 hint=0 → GPU 用全范围随机初始化。
    const int maxHintRadius = 64; // 像素
    // BFS 限距离膨胀：每步把所有"边界上有 hint 且邻居无 hint"的像素填充
    {
        // distMap: 到最近稀疏种子的曼哈顿（棋盘）距离近似（使用 2 次线性扫描）
        cv::Mat distMap(H, W, CV_32S, cv::Scalar(INT_MAX / 2));
        // 初始化种子
        for (int row = 0; row < H; ++row)
            for (int col = 0; col < W; ++col)
                if (hint.at<float>(row, col) > 0)
                    distMap.at<int>(row, col) = 0;
        // 正向扫描（左上→右下）
        for (int row = 0; row < H; ++row) {
            for (int col = 0; col < W; ++col) {
                int &d = distMap.at<int>(row, col);
                if (row > 0) d = std::min(d, distMap.at<int>(row-1, col) + 1);
                if (col > 0) d = std::min(d, distMap.at<int>(row, col-1) + 1);
            }
        }
        // 反向扫描（右下→左上）
        for (int row = H-1; row >= 0; --row) {
            for (int col = W-1; col >= 0; --col) {
                int &d = distMap.at<int>(row, col);
                if (row < H-1) d = std::min(d, distMap.at<int>(row+1, col) + 1);
                if (col < W-1) d = std::min(d, distMap.at<int>(row, col+1) + 1);
            }
        }
        // BFS 传播：在 distMap 限制内，将最近稀疏点深度写入 hint（近似最近邻）
        // 只填充距离 <= maxHintRadius 且原本为 0 的像素
        // 思路：再做一次双向扫描，但只在接受范围内传播
        cv::Mat hintDilated = hint.clone();
        for (int row = 0; row < H; ++row) {
            for (int col = 0; col < W; ++col) {
                if (distMap.at<int>(row, col) <= maxHintRadius && hintDilated.at<float>(row, col) == 0) {
                    // 从上方或左方最近的已填充像素传播
                    float v = 0.f;
                    if (row > 0 && hintDilated.at<float>(row-1, col) > 0
                        && distMap.at<int>(row-1, col) < distMap.at<int>(row, col))
                        v = hintDilated.at<float>(row-1, col);
                    if (col > 0 && hintDilated.at<float>(row, col-1) > 0
                        && distMap.at<int>(row, col-1) < distMap.at<int>(row, col))
                        v = hintDilated.at<float>(row, col-1);
                    hintDilated.at<float>(row, col) = v;
                }
            }
        }
        for (int row = H-1; row >= 0; --row) {
            for (int col = W-1; col >= 0; --col) {
                if (distMap.at<int>(row, col) <= maxHintRadius && hintDilated.at<float>(row, col) == 0) {
                    float v = 0.f;
                    if (row < H-1 && hintDilated.at<float>(row+1, col) > 0
                        && distMap.at<int>(row+1, col) < distMap.at<int>(row, col))
                        v = hintDilated.at<float>(row+1, col);
                    if (col < W-1 && hintDilated.at<float>(row, col+1) > 0
                        && distMap.at<int>(row, col+1) < distMap.at<int>(row, col))
                        v = hintDilated.at<float>(row, col+1);
                    if (hintDilated.at<float>(row, col) == 0) hintDilated.at<float>(row, col) = v;
                }
            }
        }
        // 只把扩散结果写入原来为 0 的像素（不覆盖已有稀疏点深度）
        hintDilated.copyTo(hint, hint == 0);
    }

    int hintCnt = cv::countNonZero(hint > 0);
    fprintf(stderr, "[Hint] hint 覆盖像素: %d / %d (%.1f%%)\n",
            hintCnt, W*H, 100.f*hintCnt/(W*H));
    return hint;
}

// =============================================================================
DepthFrameResult DepthMapGenerator::computeDepthForView(int refIdx, const DepthGenConfig *configOverride)
{
    DepthFrameResult result;
    result.refViewIdx = refIdx;
    result.imageIndex = refIdx;
    result.success = false;

    const DepthGenConfig &config = configOverride ? *configOverride : m_config;

    const CameraView &refView = m_views[refIdx];

    // 使用预加载的灰度图缓存（无磁盘 I/O）
    cv::Mat refImg;
    if (refIdx >= 0 && refIdx < (int)m_grayCache.size() && !m_grayCache[refIdx].empty()) {
        refImg = m_grayCache[refIdx];  // 浅拷贝，零开销
    } else {
        refImg = cv::imread(refView.imagePath, cv::IMREAD_GRAYSCALE);
    }
    if (refImg.empty()) {
        result.errorMsg = "无法读取参考帧图像: " + refView.imagePath;
        return result;
    }
    const int W = refImg.cols, H = refImg.rows;
    PositiveDepthCameraModel refCam = refView.positiveDepthModel();

    // 选择源帧（从缓存中取，省去重复加载）
    const int NV = static_cast<int>(m_views.size());
    int numSrc = std::min(config.numSourceViews, NV - 1);
    std::vector<cv::Mat> srcGrays;
    std::vector<PositiveDepthCameraModel> srcCams;

    for (int delta = 1; delta <= NV && static_cast<int>(srcGrays.size()) < numSrc; ++delta) {
        for (int sign : {-1, 1}) {
            int si = refIdx + sign * delta;
            if (si < 0 || si >= NV) continue;
            cv::Mat srcImg;
            if (si >= 0 && si < (int)m_grayCache.size() && !m_grayCache[si].empty()) {
                srcImg = m_grayCache[si];
            } else {
                srcImg = cv::imread(m_views[si].imagePath, cv::IMREAD_GRAYSCALE);
            }
            if (srcImg.empty()) continue;
            if (srcImg.cols != W || srcImg.rows != H)
                cv::resize(srcImg, srcImg, cv::Size(W, H));
            srcGrays.push_back(srcImg);
            srcCams.push_back(m_views[si].positiveDepthModel());
            if (static_cast<int>(srcGrays.size()) >= numSrc) break;
        }
    }

    if (srcGrays.empty()) {
        result.errorMsg = "没有可用的源帧";
        return result;
    }

    // 自适应置信度阈值：源视图越少，NCC 方差越大，需适当降低阈值
    PatchMatchConfig pmCfg = config.patchMatch;
    pmCfg.cpuThreadCount = std::max(1, config.cpuWorkerCount);
    if ((int)srcGrays.size() == 1) {
        // 单源视图：使用较低但非零的置信度阈值。
        // 完全归零会保留所有随机初始化深度（90% 以上暗像素 NCC=0 → conf=0），
        // 导致可视化和 crossCheck 被海量噪声淹没。
        // 阈值 0.10 ≈ NCC > 0.1，可过滤纯随机噪声但保留弱匹配。
        pmCfg.confidenceThresh = 0.10f;
        fprintf(stderr, "[MVS] 帧 %d: 单源视图模式，GPU confidenceThresh=0.10\n",
                refIdx);
    } else if ((int)srcGrays.size() <= 2) {
        // 2 源视图：适当降低阈值但不可过低，0.10 会保留大量低质量匹配→噪声
        pmCfg.confidenceThresh = std::min(pmCfg.confidenceThresh, 0.20f);
    }

    // 诊断输出
    fprintf(stderr, "\n[MVS] 帧 %d: 图像 %dx%d, 源帧 %d 个, det(R)=%.4f\n",
            refIdx, W, H, (int)srcCams.size(), det3(refCam.R_cw));
    fprintf(stderr, "[MVS] 帧 %d: C=[%.3f, %.3f, %.3f]\n",
            refIdx, refCam.C[0], refCam.C[1], refCam.C[2]);

    // 深度范围
    float zNear, zFar;
    estimateDepthRange(refIdx, zNear, zFar);
    fprintf(stderr, "[MVS] 帧 %d: zNear=%.4f  zFar=%.4f\n", refIdx, zNear, zFar);

    // 提示深度
    cv::Mat hintDepth = buildHintDepth(refIdx, W, H);

    // =========================================================================
    // ★ 极线校正（仅双目立体对时启用）
    //   将两张图像校正到极线对齐状态，使 PatchMatch 的搜索从 2D 降为近似 1D，
    //   显著降低匹配噪声。
    //   始终以较小索引为 left 进行校正，避免不同帧顺序产生不同校正几何。
    // =========================================================================
    mvs::EpipolarRectifier::RectifiedPair rectPair;
    bool useRectified = false;
    cv::Mat workRefImg = refImg;
    std::vector<cv::Mat> workSrcGrays = srcGrays;
    PositiveDepthCameraModel workRefCam = refCam;
    std::vector<PositiveDepthCameraModel> workSrcCams = srcCams;

    if (srcGrays.size() == 1)
    {
        int srcIdx = -1;
        for (int delta = 1; delta <= NV; ++delta)
            for (int sign : {-1, 1}) {
                int si = refIdx + sign * delta;
                if (si >= 0 && si < NV) { srcIdx = si; goto found_src; }
            }
        found_src:

        bool refIsCanonicalLeft = (srcIdx < 0 || refIdx < srcIdx);

        cv::Mat canonLeft  = refIsCanonicalLeft ? refImg      : srcGrays[0];
        cv::Mat canonRight = refIsCanonicalLeft ? srcGrays[0] : refImg;
        auto    camL       = refIsCanonicalLeft ? refCam      : srcCams[0];
        auto    camR       = refIsCanonicalLeft ? srcCams[0]  : refCam;

        std::string rectErr;
        if (mvs::EpipolarRectifier::rectify(
                canonLeft, canonRight, camL, camR, rectPair, &rectErr))
        {
            if (refIsCanonicalLeft) {
                workRefImg = rectPair.rectLeft;
                workSrcGrays = { rectPair.rectRight };
                workRefCam = rectPair.rectCamLeft;
                workSrcCams = { rectPair.rectCamRight };
                rectPair.refIsRight = false;
            } else {
                workRefImg = rectPair.rectRight;
                workSrcGrays = { rectPair.rectLeft };
                workRefCam = rectPair.rectCamRight;
                workSrcCams = { rectPair.rectCamLeft };
                rectPair.refIsRight = true;
            }
            useRectified = true;
            fprintf(stderr, "[MVS] 帧 %d: 极线校正成功 (ref=%s)\n",
                    refIdx, refIsCanonicalLeft ? "left" : "right");
        }
        else
        {
            fprintf(stderr, "[MVS] 帧 %d: 极线校正失败(%s)，使用原始图像\n",
                    refIdx, rectErr.c_str());
        }
    }

    // =========================================================================
    // ★ 多尺度 PatchMatch（粗到精）
    //   Pass 1: ds=4 粗分辨率 (1006×759)，NCC patch 等效 44×44 原始像素，
    //           覆盖更大空间范围 → 在低纹理/暗场景下也能获得正确初始深度。
    //   Pass 2: ds=2 精细分辨率 (2012×1518)，以粗结果为 hint 精化。
    //           只需较少迭代，因为初始深度已经接近正确值。
    // =========================================================================
    cv::Mat depthMap, confMap;
    std::string errMsg;

    // ── Pass 1: 粗分辨率 ────────────────────────────────────────────────
    PatchMatchConfig coarseCfg = pmCfg;
    coarseCfg.downsampleFactor = std::max(pmCfg.downsampleFactor * 2, 4); // 通常 ds=4
    coarseCfg.numIterations    = 12;         // 粗分辨率像素少，12 轮足够
    coarseCfg.confidenceThresh = 0.08f;      // 低阈值获取更多种子点
    coarseCfg.epipolarRectified = useRectified;

    cv::Mat coarseDepth, coarseConf;
    bool coarseOk = PatchMatchDepthEstimator::estimate(
        workRefImg, workSrcGrays, workRefCam, workSrcCams,
        zNear, zFar, coarseCfg,
        coarseDepth, &coarseConf, &errMsg,
        hintDepth.empty() ? nullptr : &hintDepth);

    if (coarseOk) {
        int coarseValid = cv::countNonZero(coarseDepth > 0);
        fprintf(stderr, "[MVS] 帧 %d: 粗分辨率(ds=%d) 完成, 有效像素=%d/%d (%.1f%%)\n",
                refIdx, coarseCfg.downsampleFactor, coarseValid, W*H,
                100.f * coarseValid / (W*H));

        // 合并 hint：粗分辨率结果 + 稀疏点（稀疏点更精确，优先保留）
        cv::Mat fineHint = coarseDepth.clone();
        if (!hintDepth.empty()) {
            hintDepth.copyTo(fineHint, hintDepth > 0);
        }

        // ── Pass 2: 精细分辨率 ──────────────────────────────────────
        PatchMatchConfig fineCfg = pmCfg;
        fineCfg.numIterations = useRectified ? 6 : 8;
        fineCfg.epipolarRectified = useRectified;

        const int hintValid = cv::countNonZero(fineHint > 0);
        const float hintCoverage = static_cast<float>(hintValid) / std::max(1, W * H);
        if (hintCoverage > 0.35f)
        {
            fineCfg.numIterations = std::min(fineCfg.numIterations, useRectified ? 5 : 6);
        }
        if (hintCoverage > 0.55f)
        {
            fineCfg.numIterations = std::min(fineCfg.numIterations, useRectified ? 4 : 5);
        }

        bool fineOk = PatchMatchDepthEstimator::estimate(
            workRefImg, workSrcGrays, workRefCam, workSrcCams,
            zNear, zFar, fineCfg,
            depthMap, &confMap, &errMsg,
            &fineHint);

        if (!fineOk) {
            depthMap = coarseDepth;
            confMap  = coarseConf;
            fprintf(stderr, "[MVS] 帧 %d: 精细分辨率失败，回退粗分辨率\n", refIdx);
        }
    } else {
        fprintf(stderr, "[MVS] 帧 %d: 粗分辨率失败，回退单尺度\n", refIdx);
        PatchMatchConfig fallbackCfg = pmCfg;
        fallbackCfg.epipolarRectified = useRectified;
        bool ok = PatchMatchDepthEstimator::estimate(
            workRefImg, workSrcGrays, workRefCam, workSrcCams,
            zNear, zFar, fallbackCfg,
            depthMap, &confMap, &errMsg,
            hintDepth.empty() ? nullptr : &hintDepth);
        if (!ok) {
            result.errorMsg = errMsg;
            return result;
        }
    }

    // ── 极线校正反变换：将校正空间的深度图映射回原始图像空间 ──────────────
    if (useRectified && !depthMap.empty())
    {
        depthMap = mvs::EpipolarRectifier::unrectifyDepth(
            depthMap, rectPair, W, H);
        if (!confMap.empty())
            confMap = mvs::EpipolarRectifier::unrectifyDepth(
                confMap, rectPair, W, H);
        fprintf(stderr, "[MVS] 帧 %d: 深度图已从校正空间映射回原始空间\n", refIdx);
    }

    // ── 暗区掩码：使用 preloadImages() 中基于原始图像计算的内容掩码 ──────
    // 必须使用 CLAHE 增强前的掩码！gamma(0.4) 将黑边 gray=3 提升到 37,
    // 使得基于增强后图像的暗区阈值失效，导致 PatchMatch 在无纹理黑边
    // 区域产生大量噪声深度（如 44% 有效像素 vs 实际仅 10% 内容区域）。
    {
        cv::Mat brightMask;
        if (refIdx >= 0 && refIdx < (int)m_contentMasks.size() && !m_contentMasks[refIdx].empty()) {
            brightMask = m_contentMasks[refIdx];
        } else {
            // 回退：直接用原始阈值（不应走到这里）
            cv::Mat blurred;
            cv::GaussianBlur(refImg, blurred, cv::Size(15, 15), 0);
            brightMask = (blurred > 15);
            fprintf(stderr, "[MVS] 帧 %d: 警告 - 内容掩码不可用，回退默认阈值\n", refIdx);
        }

        // 适配深度图尺寸（PatchMatch 可能有上采样）
        if (brightMask.size() != depthMap.size()) {
            cv::resize(brightMask, brightMask, depthMap.size(), 0, 0, cv::INTER_NEAREST);
        }

        int beforeMask = cv::countNonZero(depthMap > 0);
        depthMap.setTo(0, ~brightMask);
        if (!confMap.empty())
            confMap.setTo(0, ~brightMask);
        int afterMask = cv::countNonZero(depthMap > 0);

        if (afterMask < beforeMask) {
            fprintf(stderr, "[MVS] 帧 %d: 内容掩码过滤 %d→%d 有效像素\n",
                    refIdx, beforeMask, afterMask);
        }
    }

    // 统计
    int validCnt = cv::countNonZero(depthMap > 0);
    fprintf(stderr, "[MVS] 帧 %d: PatchMatch 完成, 有效深度像素=%d/%d (%.1f%%)\n",
            refIdx, validCnt, W*H, 100.f * validCnt / (W*H));

    result.depthMap   = QSharedPointer<cv::Mat>::create(depthMap);
    result.confidence = QSharedPointer<cv::Mat>::create(confMap);
    result.success    = true;
    return result;
}

// =============================================================================
FusionFrameInput DepthMapGenerator::buildFusionFrame(const DepthFrameResult &res) const
{
    FusionFrameInput frame;
    frame.cameraModel = m_views[res.refViewIdx].positiveDepthModel();
    frame.imgW = res.depthMap ? res.depthMap->cols : 0;
    frame.imgH = res.depthMap ? res.depthMap->rows : 0;
    frame.imagePath = m_views[res.refViewIdx].imagePath;

    if (!res.depthMap || res.depthMap->empty()) return frame;

    // 置信度过滤：将低置信度像素的深度置 0，防止随机初始化深度进入融合
    const cv::Mat &rawDepth = *res.depthMap;
    // 少视图场景使用适当的置信度阈值
    // 注：PatchMatch GPU 已在 kernelFinalizeDepth 用 0.20 过滤，
    //     这里再次过滤用更高阈值去除低质量匹配
    const int NV = static_cast<int>(m_views.size());
    float confThresh = m_config.fusion.confidenceThresh;
    if (NV <= 2) {
        // 少视图：GPU 已用 confidenceThresh=0.10 过滤纯噪声，
        // 此处不做额外二次过滤，crossCheck 负责清除不一致深度
        confThresh = 0.0f;
    }

    cv::Mat filteredDepth = rawDepth.clone();

    if (res.confidence && !res.confidence->empty()) {
        const cv::Mat &conf = *res.confidence;
        // 先统计置信度分布（用于诊断）
        double cMin, cMax;
        cv::minMaxLoc(conf, &cMin, &cMax);
        cv::Scalar cMean = cv::mean(conf, rawDepth > 0);
        fprintf(stderr, "[MVS] 帧%d 置信度统计: min=%.4f max=%.4f mean=%.4f (thresh=%.4f)\n",
                res.imageIndex, cMin, cMax, cMean[0], confThresh);

        for (int v = 0; v < rawDepth.rows; ++v) {
            const float *pd = rawDepth.ptr<float>(v);
            const float *pc = conf.ptr<float>(v);
            float       *pf = filteredDepth.ptr<float>(v);
            for (int u = 0; u < rawDepth.cols; ++u) {
                if (pc[u] < confThresh) pf[u] = 0.f;
            }
        }
        int validBefore = cv::countNonZero(rawDepth > 0);
        int validAfter  = cv::countNonZero(filteredDepth > 0);
        fprintf(stderr, "[MVS] 帧%d 置信度过滤: %d→%d 有效像素 (thresh=%.4f)\n",
                res.imageIndex, validBefore, validAfter, confThresh);

        // 如果过滤后像素太少，自动回退为不过滤
        if (validAfter < validBefore / 20) { // 保留不足 5%
            fprintf(stderr, "[MVS] 帧%d 置信度过滤后像素太少，回退为不过滤\n", res.imageIndex);
            filteredDepth = rawDepth.clone();
        }
    }

    frame.depthMap   = filteredDepth;
    frame.confidence = res.confidence ? *res.confidence : cv::Mat();
    return frame;
}

// =============================================================================
// 双视图深度图左右一致性检查
// 对每个深度像素：反投影到 3D → 投影到另一视图 → 比较另一视图的深度值
//
// 策略:
//   多视图 (≥3): "需要确认" — 像素必须得到至少一个其他视图的深度一致性确认
//   少视图 (≤2): "仅移除矛盾" — 仅移除被其他视图明确否定的像素；
//                 对方深度为 0 或超出投影范围时，保留原像素（疑罪从无）
// =============================================================================
void DepthMapGenerator::crossCheckDepthConsistency()
{
    const int NV = static_cast<int>(m_views.size());
    if (NV < 2) return;

    // 相对深度误差阈值：少视图放宽至 25%，多视图 5%
    const float relThresh = (NV <= 2) ? 0.25f : 0.05f;
    const bool  fewViews  = (NV <= 2);

    // ── 先保存所有帧的原始深度图拷贝，避免顺序处理的级联清除问题 ──────
    std::vector<cv::Mat> origDepths(NV);
    for (int i = 0; i < NV; ++i)
    {
        if (m_depthFrames[i].success && m_depthFrames[i].depthMap)
        {
            origDepths[i] = m_depthFrames[i].depthMap->clone();
        }
    }

    for (int i = 0; i < NV; ++i)
    {
        if (!m_depthFrames[i].success || !m_depthFrames[i].depthMap)
        {
            continue;
        }
        cv::Mat &depthI = *m_depthFrames[i].depthMap;
        PositiveDepthCameraModel camI = m_views[i].positiveDepthModel();

        // 备份，万一过滤太激进需要回退
        cv::Mat depthBackup = depthI.clone();

        // ─── 少视图模式: 初始化为全部保留, 仅标记"被明确否定"的像素 ────
        // ─── 多视图模式: 初始化为全部移除, 需要"被确认"才保留 ──────────
        cv::Mat consistentMask(depthI.size(), CV_8U,
                               fewViews ? cv::Scalar(255) : cv::Scalar(0));

        for (int j = 0; j < NV; ++j)
        {
            if (j == i)
            {
                continue;
            }
            if (origDepths[j].empty())
            {
                continue;
            }
            const cv::Mat &depthJ = origDepths[j];
            PositiveDepthCameraModel camJ = m_views[j].positiveDepthModel();

            for (int v = 0; v < depthI.rows; ++v)
            {
                const float *pdi = depthI.ptr<float>(v);
                uint8_t     *pMask = consistentMask.ptr<uint8_t>(v);
                for (int u = 0; u < depthI.cols; ++u)
                {
                    float di = pdi[u];
                    if (di <= 0.f)
                    {
                        continue;
                    }

                    if (fewViews)
                    {
                        // 少视图: 只在被明确否定时移除
                        if (pMask[u] == 0)
                        {
                            continue; // 已被其他视图否定
                        }
                    }
                    else
                    {
                        // 多视图: 已确认的跳过
                        if (pMask[u])
                        {
                            continue;
                        }
                    }

                    // 反投影到世界坐标
                    float Xw, Yw, Zw;
                    camI.unproject(static_cast<float>(u), static_cast<float>(v), di, Xw, Yw, Zw);

                    // 投影到帧 j
                    float uj, vj;
                    if (!camJ.project(Xw, Yw, Zw, uj, vj))
                    {
                        continue;
                    }
                    int iuj = static_cast<int>(std::round(uj));
                    int ivj = static_cast<int>(std::round(vj));
                    if (iuj < 0 || iuj >= depthJ.cols || ivj < 0 || ivj >= depthJ.rows)
                    {
                        continue;
                    }

                    float dj = depthJ.at<float>(ivj, iuj);
                    if (dj <= 0.f)
                    {
                        continue; // 对方无深度 → 少视图保留，多视图无变化
                    }

                    // 计算预期深度与实测深度的相对误差
                    float Zc_j = camJ.R_cw[6]*Xw + camJ.R_cw[7]*Yw + camJ.R_cw[8]*Zw + camJ.T[2];
                    if (Zc_j <= 0.f)
                    {
                        continue;
                    }

                    float relErr = std::fabs(dj - Zc_j) / Zc_j;

                    if (fewViews)
                    {
                        // 少视图: 明确矛盾 → 标记移除
                        if (relErr >= relThresh)
                        {
                            pMask[u] = 0;
                        }
                    }
                    else
                    {
                        // 多视图: 一致 → 标记保留
                        if (relErr < relThresh)
                        {
                            pMask[u] = 255;
                        }
                    }
                }
            }
        }

        // 剔除不一致像素
        int beforeValid = cv::countNonZero(depthI > 0);
        depthI.setTo(0, consistentMask == 0);
        int afterValid = cv::countNonZero(depthI > 0);
        float keepRate = beforeValid > 0 ? 100.f * afterValid / beforeValid : 0.f;
        fprintf(stderr, "[MVS] 帧%d 一致性检查(%s): %d→%d 有效像素 (保留 %.1f%%)\n",
                i, fewViews ? "仅移除矛盾" : "需要确认",
                beforeValid, afterValid, keepRate);

        // 安全回退：如果保留率过低（< 10%），回退使用原始深度图
        if (afterValid < beforeValid / 10 && beforeValid > 100) 
        {
            fprintf(stderr, "[MVS] 帧%d 一致性过滤后像素太少 (%.1f%%)，回退为原始深度图\n",
                    i, keepRate);
            depthBackup.copyTo(depthI);
        }
    }
}

// =============================================================================
void DepthMapGenerator::runInBackground()
{
    bool allOk = true;
    const int NV = static_cast<int>(m_views.size());

    if (!m_config.runDepthEstimation)
    {
        emit errorOccurred(QStringLiteral("当前生成器配置未启用深度估计阶段"));
        emit finished(false);
        return;
    }

    // ── 预加载所有图像（一次性磁盘 I/O，后续全从内存读取）────────────────────
    emit progressChanged(QString("预加载 %1 张图像...").arg(NV), 0.f);
    preloadImages();
    m_depthFrames.resize(NV);

    // ── 阶段一：优先级队列并行估计深度图（GPU 优先高价值帧，CPU 处理其余帧）────
    int skippedFrames = 0;
    for (size_t i = 0; i < m_skipFrameMask.size(); ++i)
    {
        if (m_skipFrameMask[i] != 0)
        {
            ++skippedFrames;
        }
    }

    std::atomic<int> completedTasks{skippedFrames};
    std::atomic<int> activeGpuTasks{0};
    std::atomic<int> activeCpuTasks{0};
    std::atomic<bool> anyFailure{false};
    std::mutex saveMutex;
    std::mutex taskMutex;

    struct FramePriority
    {
        int viewIndex = -1;
        float score = 0.f;
    };

    const bool cudaAvailable = m_config.patchMatch.useCuda && PatchMatchDepthEstimator::isCudaAvailable();
    const int cpuThreadCount = std::max(1, m_config.cpuWorkerCount);
    const int cpuWorkers = cudaAvailable ? 0 : 1;

    std::vector<FramePriority> framePriorities;
    framePriorities.reserve(static_cast<size_t>(NV));
    for (int i = 0; i < NV; ++i)
    {
        if (i >= 0 && i < static_cast<int>(m_skipFrameMask.size()) && m_skipFrameMask[static_cast<size_t>(i)] != 0)
        {
            continue;
        }

        float resolutionScore = 0.f;
        if (i >= 0 && i < static_cast<int>(m_grayCache.size()) && !m_grayCache[i].empty())
        {
            resolutionScore = static_cast<float>(m_grayCache[i].cols * m_grayCache[i].rows);
        }

        float contentRatio = 0.5f;
        if (i >= 0 && i < static_cast<int>(m_contentMasks.size()) && !m_contentMasks[i].empty())
        {
            const int totalPx = m_contentMasks[i].rows * m_contentMasks[i].cols;
            if (totalPx > 0)
            {
                contentRatio = static_cast<float>(cv::countNonZero(m_contentMasks[i])) / static_cast<float>(totalPx);
            }
        }

        const int leftNeighbors = i;
        const int rightNeighbors = (NV - 1) - i;
        const int localSupport = std::min(leftNeighbors, rightNeighbors);

        FramePriority priority;
        priority.viewIndex = i;
        priority.score = resolutionScore * (0.70f + 0.30f * contentRatio)
                       + static_cast<float>(localSupport) * 100000.0f;
        framePriorities.push_back(priority);
    }

    std::sort(framePriorities.begin(), framePriorities.end(), [](const FramePriority &a, const FramePriority &b)
    {
        return a.score > b.score;
    });

    std::deque<int> gpuQueue;
    std::deque<int> cpuQueue;
    if (cudaAvailable)
    {
        for (const FramePriority &priority : framePriorities)
        {
            gpuQueue.push_back(priority.viewIndex);
        }
    }
    else
    {
        for (const FramePriority &priority : framePriorities)
        {
            cpuQueue.push_back(priority.viewIndex);
        }
    }

    LOG_INFO(QStringLiteral("[MVS] 深度估计调度: cuda=%1 gpu_frame_workers=%2 cpu_frame_workers=%3 cpu_pixel_threads=%4 views=%5 gpuQueue=%6 cpuQueue=%7")
                 .arg(cudaAvailable ? QStringLiteral("on") : QStringLiteral("off"))
                 .arg(cudaAvailable ? 1 : 0)
                 .arg(cpuWorkers)
                 .arg(cpuThreadCount)
                 .arg(NV)
                 .arg(static_cast<qulonglong>(gpuQueue.size()))
                 .arg(static_cast<qulonglong>(cpuQueue.size())));
    if (skippedFrames > 0)
    {
        LOG_INFO(QStringLiteral("[MVS] 续跑模式：跳过已存在深度图 %1 帧").arg(skippedFrames));
    }
    if (cudaAvailable)
    {
        LOG_INFO(QStringLiteral("[MVS] CUDA 已启用，最终深度图统一由 GPU 生成；CPU 仅保留为无 CUDA 场景的像素级并行回退"));
    }

    auto emitDepthProgress = [this, NV, &completedTasks, &activeGpuTasks, &activeCpuTasks, &taskMutex, &gpuQueue, &cpuQueue](
                                 const QString &workerTag,
                                 int frameIndex,
                                 bool pickedTask)
    {
        int gpuPending = 0;
        int cpuPending = 0;
        {
            std::lock_guard<std::mutex> lock(taskMutex);
            gpuPending = static_cast<int>(gpuQueue.size());
            cpuPending = static_cast<int>(cpuQueue.size());
        }

        const int done = completedTasks.load();
        const int gpuActive = activeGpuTasks.load();
        const int cpuActive = activeCpuTasks.load();
        const float ratio = static_cast<float>(done) / (NV + 2);

        QString stage = QStringLiteral("深度估计: 已完成 %1/%2, 运行中 GPU %3 CPU %4, 待处理 GPU %5 CPU %6")
                            .arg(done)
                            .arg(NV)
                            .arg(gpuActive)
                            .arg(cpuActive)
                            .arg(gpuPending)
                            .arg(cpuPending);

        if (pickedTask && frameIndex >= 0)
        {
            stage += QStringLiteral(", 当前启动帧 %1 [%2]")
                         .arg(frameIndex + 1)
                         .arg(workerTag);
        }
        else if (frameIndex >= 0)
        {
            stage += QStringLiteral(", 最新完成帧 %1 [%2]")
                         .arg(frameIndex + 1)
                         .arg(workerTag);
        }

        emit progressChanged(stage, ratio);
    };

    auto workerFunc = [this, NV, &completedTasks, &activeGpuTasks, &activeCpuTasks, &anyFailure,
                       &saveMutex, &taskMutex, &gpuQueue, &cpuQueue, &emitDepthProgress](bool useGpu) {
        DepthGenConfig workerConfig = m_config;
        workerConfig.patchMatch.useCuda = useGpu;
        const QString workerTag = useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU");

        while (!m_cancelled)
        {
            int i = -1;
            {
                std::lock_guard<std::mutex> lock(taskMutex);
                if (useGpu)
                {
                    if (!gpuQueue.empty())
                    {
                        i = gpuQueue.front();
                        gpuQueue.pop_front();
                    }
                    else if (!cpuQueue.empty())
                    {
                        i = cpuQueue.front();
                        cpuQueue.pop_front();
                    }
                }
                else
                {
                    if (!cpuQueue.empty())
                    {
                        i = cpuQueue.front();
                        cpuQueue.pop_front();
                    }
                    else if (!gpuQueue.empty())
                    {
                        i = gpuQueue.front();
                        gpuQueue.pop_front();
                    }
                }
            }

            if (i < 0 || i >= NV)
            {
                break;
            }

            if (useGpu)
            {
                activeGpuTasks.fetch_add(1);
            }
            else
            {
                activeCpuTasks.fetch_add(1);
            }
            emitDepthProgress(workerTag, i, true);

            const auto frameStart = std::chrono::steady_clock::now();
            LOG_INFO(QStringLiteral("[MVS] 帧 %1 开始深度估计: device=%2")
                         .arg(i)
                         .arg(useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU")));

            DepthFrameResult res = computeDepthForView(i, &workerConfig);
            m_depthFrames[i] = res;
            const auto frameEnd = std::chrono::steady_clock::now();
            const double elapsedMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

            if (!res.success)
            {
                LOG_WARN(QStringLiteral("[MVS] 帧 %1 深度估计失败: device=%2 elapsed=%3 ms error=%4")
                             .arg(i)
                             .arg(useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU"))
                             .arg(elapsedMs, 0, 'f', 1)
                             .arg(QString::fromStdString(res.errorMsg)));
                anyFailure = true;
            }
            else
            {
                const int depthWidth = (res.depthMap && !res.depthMap->empty()) ? res.depthMap->cols : 0;
                const int depthHeight = (res.depthMap && !res.depthMap->empty()) ? res.depthMap->rows : 0;
                LOG_INFO(QStringLiteral("[MVS] 帧 %1 深度估计完成: device=%2 elapsed=%3 ms size=%4x%5")
                             .arg(i)
                             .arg(useGpu ? QStringLiteral("GPU") : QStringLiteral("CPU"))
                             .arg(elapsedMs, 0, 'f', 1)
                             .arg(depthWidth)
                             .arg(depthHeight));
            }

            if (res.success && res.depthMap && !res.depthMap->empty() && !m_outputDir.empty())
            {
                std::lock_guard<std::mutex> guard(saveMutex);
                const cv::Mat &dm = *res.depthMap;
                double dMin, dMax;
                cv::Mat validMask = dm > 0;
                cv::minMaxLoc(dm, &dMin, &dMax, nullptr, nullptr, validMask);
                cv::Mat vis;
                if (dMax > dMin)
                {
                    dm.convertTo(vis, CV_8U, 255.0 / (dMax - dMin), -255.0 * dMin / (dMax - dMin));
                }
                else
                {
                    vis = cv::Mat::zeros(dm.size(), CV_8U);
                }
                vis.setTo(0, dm <= 0);
                cv::Mat colorVis;
                cv::applyColorMap(vis, colorVis, cv::COLORMAP_TURBO);
                colorVis.setTo(cv::Scalar(0, 0, 0), dm <= 0);

                std::string pngPath = m_outputDir + "/depth_" + std::to_string(i) + ".png";
                cv::imwrite(pngPath, colorVis);
                LOG_INFO(QStringLiteral("[MVS] 帧 %1 深度图已保存: %2 (%3x%4)")
                         .arg(i)
                         .arg(QString::fromStdString(pngPath))
                         .arg(dm.cols)
                         .arg(dm.rows));
                emit depthMapSaved(
                    QString::fromStdString(pngPath),
                    dm.cols, dm.rows,
                    QString::fromStdString(m_views[i].imagePath));
            }

            emit depthMapReady(res);
            const int done = completedTasks.fetch_add(1) + 1;
            if (useGpu)
            {
                activeGpuTasks.fetch_sub(1);
            }
            else
            {
                activeCpuTasks.fetch_sub(1);
            }
            Q_UNUSED(done);
            emitDepthProgress(workerTag, i, false);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(cpuWorkers + (cudaAvailable ? 1 : 0)));

    if (cudaAvailable)
    {
        workers.emplace_back(workerFunc, true);
    }
    for (int workerIndex = 0; workerIndex < cpuWorkers; ++workerIndex)
    {
        workers.emplace_back(workerFunc, false);
    }

    for (std::thread &worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }

    allOk = !anyFailure.load();

    // 释放图像缓存（深度估计完毕后不再需要）
    m_grayCache.clear();
    m_grayCache.shrink_to_fit();
    m_contentMasks.clear();
    m_contentMasks.shrink_to_fit();

    if (m_cancelled) {
        emit finished(false);
        return;
    }

    // ── 阶段 1.5：双视图深度图左右一致性检查 ────────────────────────────────
    // 在融合前剔除 PatchMatch 产生的幽灵深度（两视图深度互不一致的像素）
    if (NV >= 2) {
        emit progressChanged("深度一致性检查...", static_cast<float>(NV) / (NV + 3));
        crossCheckDepthConsistency();
    }

    if (m_config.saveIntermediateDepthMaps && !m_config.intermediateDir.empty())
    {
        for (int i = 0; i < NV; ++i)
        {
            const DepthFrameResult &res = m_depthFrames[i];
            if (!res.success || !res.depthMap || res.depthMap->empty())
            {
                continue;
            }

            const std::string depthPath = m_config.intermediateDir + "/depth_" + std::to_string(i) + ".yml.gz";
            std::string saveErr;
            if (!writeCvMatStorage(depthPath, *res.depthMap, &saveErr))
            {
                LOG_WARN(QStringLiteral("[MVS] 保存原始深度失败: %1")
                             .arg(QString::fromStdString(saveErr)));
            }

            if (res.confidence && !res.confidence->empty())
            {
                const std::string confPath = m_config.intermediateDir + "/depth_" + std::to_string(i) + "_conf.yml.gz";
                if (!writeCvMatStorage(confPath, *res.confidence, &saveErr))
                {
                    LOG_WARN(QStringLiteral("[MVS] 保存置信图失败: %1")
                                 .arg(QString::fromStdString(saveErr)));
                }
            }
        }
    }

    if (!m_config.runFusion)
    {
        emit progressChanged("完成", 1.f);
        emit finished(allOk);
        return;
    }

    // ── 阶段二：COLMAP BFS 深度图融合 → 直接输出 3D 点 ──────────────────────
    emit progressChanged("深度图融合...", static_cast<float>(NV) / (NV + 2));

    std::vector<FusionFrameInput> frames;
    for (const auto &fr : m_depthFrames) {
        if (fr.success && fr.depthMap && !fr.depthMap->empty())
            frames.push_back(buildFusionFrame(fr));
    }

    if (frames.empty()) {
        emit errorOccurred("没有有效的深度帧，融合失败");
        emit finished(false);
        return;
    }

    fprintf(stderr, "[MVS] COLMAP BFS 融合: %d 帧参与\n", (int)frames.size());

    // 使用新的 StereoFusionConfig
    StereoFusionConfig fusionCfg;
    fusionCfg.minNumPixels   = m_config.fusion.minConsistentViews;
    fusionCfg.maxReprojError = m_config.fusion.pixelThresh;
    fusionCfg.maxDepthError  = m_config.fusion.relDepthThresh;
    fusionCfg.checkNumImages = std::min(50, NV);

    // 少视图场景（≤2张）：crossCheck 已保证深度一致性，
    // 融合只需 1 个观测即可通过（避免 BFS 因级联过滤找不到第二观测而全部拒绝）
    if (NV <= 2) {
        fusionCfg.minNumPixels   = 1;   // 单观测即可，crossCheck 已保证质量
        fusionCfg.maxDepthError  = std::max(fusionCfg.maxDepthError, 0.08f); // 放宽至 8%
        fusionCfg.maxReprojError = std::max(fusionCfg.maxReprojError, 3.0f); // 3 像素重投影误差
        fprintf(stderr, "[MVS] 少视图模式 (%d 帧)，minNumPixels=1, 放宽融合阈值\n", NV);
    }

    // 如果有稀疏云 AABB，设置包围盒过滤离群点
    // 但必须检查稀疏云与相机是否在同一坐标系（防止 GPS 稀疏云 vs SFM 相机失配）
    if (m_sparse.points.size() > 10) {
        // 计算相机中心的最大分量，作为坐标系尺度参考
        float camExtent = 1.0f;
        for (const auto &v : m_views) {
            PositiveDepthCameraModel vc = v.positiveDepthModel();
            for (int k = 0; k < 3; ++k)
                camExtent = std::max(camExtent, std::fabs(vc.C[k]));
        }
        // 稀疏云 AABB 最大分量
        float cloudExtent = 0.0f;
        for (int k = 0; k < 3; ++k)
            cloudExtent = std::max(cloudExtent,
                std::max(std::fabs(m_sparse.maxPt[k]), std::fabs(m_sparse.minPt[k])));
        // 坐标系兼容判断：尺度差不超过 50 倍才启用包围盒
        const float scaleRatio = cloudExtent / std::max(camExtent, 1.0f);
        if (scaleRatio < 50.0f)
        {
            fusionCfg.useBoundingBox = true;
            float pad = (NV <= 2) ? 0.5f : 0.2f;
            for (int k = 0; k < 3; ++k)
            {
                float range = m_sparse.maxPt[k] - m_sparse.minPt[k];
                fusionCfg.bboxMin[k] = m_sparse.minPt[k] - range * pad;
                fusionCfg.bboxMax[k] = m_sparse.maxPt[k] + range * pad;
            }
            fprintf(stderr, "[MVS] 包围盒: [%.2f,%.2f,%.2f] ~ [%.2f,%.2f,%.2f]\n",
                    fusionCfg.bboxMin[0], fusionCfg.bboxMin[1], fusionCfg.bboxMin[2],
                    fusionCfg.bboxMax[0], fusionCfg.bboxMax[1], fusionCfg.bboxMax[2]);
        }
        else
        {
            fprintf(stderr, "[MVS] 稀疏云与相机坐标系不匹配 (cloudExtent=%.1f camExtent=%.1f ratio=%.1f)"
                            "，禁用包围盒过滤\n", cloudExtent, camExtent, scaleRatio);
            // 坐标系不兼容时深度初始化无先验，放宽一致性阈值至 10%
            fusionCfg.maxDepthError = std::max(fusionCfg.maxDepthError, 0.10f);
            fprintf(stderr, "[MVS] 坐标系不兼容，自动放宽 maxDepthError → %.2f\n",
                    fusionCfg.maxDepthError);
        }
    }

    DepthMapFusion fusion(fusionCfg);
    std::vector<DensePoint> cloud;
    std::string fuseErr;

    bool fuseOk = fusion.fuse(frames, cloud,
        [this, NV](const std::string &msg, float ratio)
        {
            emit progressChanged(
                QString::fromStdString(msg),
                static_cast<float>(NV + ratio) / (NV + 2));
        },
        &fuseErr);

    if (!fuseOk)
    {
        fprintf(stderr, "[MVS] 融合失败: %s\n", fuseErr.c_str());
        emit errorOccurred(QString::fromStdString(fuseErr));
        emit finished(false);
        return;
    }

    // 保存每帧一致性过滤深度图（加锁，防止 GUI 线程并发读取）
    {
        std::lock_guard<std::mutex> lock(m_filteredDepthsMutex);
        m_filteredDepths = fusion.filteredDepths();
    }

    fprintf(stderr, "[MVS] 融合完成: %d 个稠密点\n", (int)cloud.size());

    if (cloud.empty())
    {
        fprintf(stderr, "[MVS] 警告: 融合产出 0 个点，可能原因：\n"
                        "  - 深度图质量不足（过少有效像素）\n"
                        "  - 视图数量太少（当前 %d 帧，建议 >=3）\n"
                        "  - 深度一致性阈值过严\n",
                NV);
        emit errorOccurred(QStringLiteral("深度图融合产出 0 个点，请尝试增加影像数量或降低融合阈值"));
    }

    // ── 阶段三：离群点过滤 ──────────────────────────────────────────────
    if (!cloud.empty())
    {
        emit progressChanged("离群点过滤...", 0.90f);

        const std::size_t initialCount = cloud.size();
        const float maxStageRemovalRatio = 0.45f;
        const float minOverallRetentionRatio = (initialCount > 500000) ? 0.50f : 0.40f;

        auto applyFilterWithGuard = [&](const char *stageName, auto &&filterOp)
        {
            if (cloud.size() < 100)
            {
                return;
            }

            const std::size_t beforeCount = cloud.size();
            std::vector<DensePoint> filtered = filterOp(cloud);
            if (filtered.empty())
            {
                fprintf(stderr, "[MVS] %s 结果为空，跳过该阶段以保护点云\n", stageName);
                return;
            }

            const float stageRemovedRatio = static_cast<float>(beforeCount - filtered.size())
                                            / static_cast<float>(beforeCount);
            const float overallRetentionRatio = static_cast<float>(filtered.size())
                                              / static_cast<float>(initialCount);

            if (stageRemovedRatio > maxStageRemovalRatio || overallRetentionRatio < minOverallRetentionRatio)
            {
                fprintf(stderr,
                        "[MVS] %s 触发过滤护栏: stageRemoved=%.1f%%(上限%.1f%%), overallRetention=%.1f%%(下限%.1f%%)，保留上一阶段结果\n",
                        stageName,
                        stageRemovedRatio * 100.0f,
                        maxStageRemovalRatio * 100.0f,
                        overallRetentionRatio * 100.0f,
                        minOverallRetentionRatio * 100.0f);
                return;
            }

            cloud.swap(filtered);
        };

        // 第 1 遍：统计离群点过滤（SOR）— 移除 kNN 距离异常的点
        applyFilterWithGuard("SOR-1", [](const std::vector<DensePoint> &inputCloud) {
            return DenseCloudBuilder::statisticalOutlierRemoval(inputCloud, 30, 1.2f);
        });

        // 第 2 遍：半径过滤 — 基于点云密度估算搜索半径
        if (cloud.size() > 100)
        {
            // 估算平均点间距来确定半径
            float minX=cloud[0].x, maxX=cloud[0].x;
            float minY=cloud[0].y, maxY=cloud[0].y;
            float minZ=cloud[0].z, maxZ=cloud[0].z;
            for (const auto &p : cloud) {
                minX=std::min(minX,p.x); maxX=std::max(maxX,p.x);
                minY=std::min(minY,p.y); maxY=std::max(maxY,p.y);
                minZ=std::min(minZ,p.z); maxZ=std::max(maxZ,p.z);
            }
            float volume = (maxX-minX) * (maxY-minY) * (maxZ-minZ);
            volume = std::max(volume, 1e-12f);
            float avgSpacing = std::cbrt(volume / (float)cloud.size());
            float searchRadius = std::max(avgSpacing * 4.0f, 1e-4f); // 4 倍平均间距内至少要有邻居
            applyFilterWithGuard("RadiusOR", [searchRadius](const std::vector<DensePoint> &inputCloud) {
                return DenseCloudBuilder::radiusOutlierRemoval(inputCloud, searchRadius, 5);
            });

            // 第 3 遍：再做一轮 SOR 清除残余离群
            if (cloud.size() > 100) {
                applyFilterWithGuard("SOR-2", [](const std::vector<DensePoint> &inputCloud) {
                    return DenseCloudBuilder::statisticalOutlierRemoval(inputCloud, 20, 1.5f);
                });
            }
        }

        fprintf(stderr, "[MVS] 过滤后剩余: %d 个稠密点\n", (int)cloud.size());
    }

    emit pointCloudReady(cloud);
    emit progressChanged("完成", 1.f);
    // 只要最终生成了有效点云就算成功（部分帧深度估计失败不影响最终结果）
    emit finished(!cloud.empty());
}

} // namespace mvs
} // namespace xjw
