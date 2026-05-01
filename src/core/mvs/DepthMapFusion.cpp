// =============================================================================
// 文件: DepthMapFusion.cpp
// 模块: MVS - COLMAP 风格多视深度图融合
// 说明:
//   参考 COLMAP fusion.cc 的 BFS 多视图一致性融合算法。
//   每个参考像素向重叠视图投影，检查深度/重投影/法线一致性，
//   满足 minNumPixels 的像素使用中位数聚合生成一个 3D 点。
// =============================================================================

#include "DepthMapFusion.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <queue>

namespace xjw
{
namespace mvs
{

// =============================================================================
DepthMapFusion::DepthMapFusion(const StereoFusionConfig &config)
    : m_config(config)
{
}

// =============================================================================
float DepthMapFusion::median(std::vector<float> &v)
{
    if (v.empty())
    {
        return 0.f;
    }
    size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n/2, v.end());
    return v[n/2];
}

// =============================================================================
// 预计算投影几何
// =============================================================================
void DepthMapFusion::prepareGeometry(
    const std::vector<FusionFrameInput> &frames,
    std::vector<FrameGeometry> &geom)
{
    const int NF = static_cast<int>(frames.size());
    geom.resize(NF);

    for (int fi = 0; fi < NF; ++fi)
    {
        const PositiveDepthCameraModel &cam = frames[fi].cameraModel;
        FrameGeometry &g = geom[fi];
        g.cameraModel = cam;
        g.W = frames[fi].imgW;
        g.H = frames[fi].imgH;

        // 构造 3×4 投影矩阵 P = K * [R | T]
        // K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                float kRow[3] = {0, 0, 0};
                if (r == 0)
                {
                    kRow[0] = cam.fx;
                    kRow[2] = cam.cx;
                }
                else if (r == 1)
                {
                    kRow[1] = cam.fy;
                    kRow[2] = cam.cy;
                }
                else
                {
                    kRow[2] = 1.0f;
                }

                g.P[r*4+c] = kRow[0]*cam.R_cw[0*3+c] +
                              kRow[1]*cam.R_cw[1*3+c] +
                              kRow[2]*cam.R_cw[2*3+c];
            }
            // T 列
            float kRow[3] = {0, 0, 0};
            if (r == 0)
            {
                kRow[0] = cam.fx;
                kRow[2] = cam.cx;
            }
            else if (r == 1)
            {
                kRow[1] = cam.fy;
                kRow[2] = cam.cy;
            }
            else
            {
                kRow[2] = 1.0f;
            }

            g.P[r*4+3] = kRow[0]*cam.T[0] + kRow[1]*cam.T[1] + kRow[2]*cam.T[2];
        }

        // 逆旋转 R_wc = R_cw^T
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                g.invR[r*3+c] = cam.R_cw[c*3+r];
            }
        }

        // 逆投影矩阵（简化表达 — 直接用 cameraModel.unproject）
        // invP 不需要显式存储，我们直接调用正深度模型的反投影接口
    }
}

// =============================================================================
// 计算重叠图像：基于相机中心距离和朝向
// =============================================================================
void DepthMapFusion::computeOverlappingImages(
    const std::vector<FusionFrameInput> &frames,
    const std::vector<FrameGeometry> &geom,
    std::vector<std::vector<int>> &overlapping)
{
    const int NF = static_cast<int>(frames.size());
    overlapping.resize(NF);

    for (int fi = 0; fi < NF; ++fi)
    {
        // 计算到所有其他帧的距离，取最近的 checkNumImages 个
        struct OverlapInfo
        {
            int idx;
            float dist;
        };
        std::vector<OverlapInfo> candidates;
        candidates.reserve(NF - 1);

        const float *C_fi = frames[fi].cameraModel.C;
        for (int fj = 0; fj < NF; ++fj)
        {
            if (fj == fi)
            {
                continue;
            }
            const float *C_fj = frames[fj].cameraModel.C;
            float dx = C_fj[0] - C_fi[0];
            float dy = C_fj[1] - C_fi[1];
            float dz = C_fj[2] - C_fi[2];
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            candidates.push_back({fj, dist});
        }

        // 按距离排序，取最近的 N 个
        std::sort(candidates.begin(), candidates.end(),
                  [](const OverlapInfo &a, const OverlapInfo &b)
                  {
                      return a.dist < b.dist;
                  });

        int numOverlap = std::min(m_config.checkNumImages, (int)candidates.size());
        overlapping[fi].resize(numOverlap);
        for (int i = 0; i < numOverlap; ++i)
        {
            overlapping[fi][i] = candidates[i].idx;
        }
    }
}

// =============================================================================
// BFS 单像素融合（COLMAP StereoFusion::Fuse 风格）
// =============================================================================
bool DepthMapFusion::fusePixel(
    int imageIdx, int row, int col,
    const std::vector<FusionFrameInput> &frames,
    const std::vector<FrameGeometry> &geom,
    const std::vector<std::vector<int>> &overlapping,
    const std::vector<cv::Mat> &colorImages,
    std::vector<std::vector<char>> &fusedMask,
    FusedPoint &outPoint)
{
    const float maxReprojSq = m_config.maxReprojError * m_config.maxReprojError;
    const float maxDepthErr = m_config.maxDepthError;
    const float cosMaxNormErr = std::cos(m_config.maxNormalError * (float)M_PI / 180.f);

    // BFS 数据
    struct QueueItem
    {
        int imgIdx;
        int row;
        int col;
        int depth;
    };
    std::queue<QueueItem> bfsQueue;
    bfsQueue.push({imageIdx, row, col, 0});

    // 收集的观测
    std::vector<float> xs, ys, zs;
    std::vector<float> nxs, nys, nzs;
    std::vector<uint8_t> rs_v, gs_v, bs_v;

    // 参考点
    float refX = 0.f, refY = 0.f, refZ = 0.f;
    bool hasRef = false;

    while (!bfsQueue.empty() && (int)xs.size() < m_config.maxNumPixels)
    {
        QueueItem item = bfsQueue.front();
        bfsQueue.pop();

        const int fi = item.imgIdx;
        const int r = item.row;
        const int c = item.col;
        const int td = item.depth;
        const FrameGeometry &g = geom[fi];

        // 检查边界
        if (r < 0 || r >= g.H || c < 0 || c >= g.W)
        {
            continue;
        }

        // 已融合检查
        if (fusedMask[fi][r * g.W + c])
        {
            continue;
        }

        // 读深度
        float d = frames[fi].depthMap.at<float>(r, c);
        if (d <= 0.f)
        {
            continue;
        }

        // 非首次遍历需要做一致性检查
        if (td > 0 && hasRef)
        {
            // 将参考点投影到当前帧
            float u_proj, v_proj;
            if (!g.cameraModel.project(refX, refY, refZ, u_proj, v_proj))
            {
                continue;
            }

            // 重投影误差
            float du = u_proj - c;
            float dv = v_proj - r;
            if (du*du + dv*dv > maxReprojSq)
            {
                continue;
            }

            // 深度一致性：计算期望深度 vs 实测深度
            float Zc_expect = g.cameraModel.R_cw[6]*refX + g.cameraModel.R_cw[7]*refY +
                              g.cameraModel.R_cw[8]*refZ + g.cameraModel.T[2];
            if (Zc_expect <= 0.f)
            {
                continue;
            }
            float relDepthErr = std::fabs(d - Zc_expect) / Zc_expect;
            if (relDepthErr > maxDepthErr)
            {
                continue;
            }

            // 法线一致性（如果有法线图）
            if (!frames[fi].normalMap.empty() && !frames[imageIdx].normalMap.empty())
            {
                const cv::Vec3f &nCurr = frames[fi].normalMap.at<cv::Vec3f>(r, c);
                const cv::Vec3f &nRef  = frames[imageIdx].normalMap.at<cv::Vec3f>(row, col);

                // 法线从相机坐标系转到世界坐标系
                const float *invR_curr = geom[fi].invR;
                const float *invR_ref  = geom[imageIdx].invR;

                float nw_curr[3], nw_ref[3];
                for (int k = 0; k < 3; ++k)
                {
                    nw_curr[k] = invR_curr[k*3+0]*nCurr[0] +
                                 invR_curr[k*3+1]*nCurr[1] +
                                 invR_curr[k*3+2]*nCurr[2];
                    nw_ref[k]  = invR_ref[k*3+0]*nRef[0] +
                                 invR_ref[k*3+1]*nRef[1] +
                                 invR_ref[k*3+2]*nRef[2];
                }

                float cosAngle = nw_curr[0]*nw_ref[0] + nw_curr[1]*nw_ref[1] +
                                 nw_curr[2]*nw_ref[2];
                if (cosAngle < cosMaxNormErr)
                {
                    continue;
                }
            }
        }

        // 计算 3D 坐标
        float Xw, Yw, Zw;
        g.cameraModel.unproject(static_cast<float>(c), static_cast<float>(r), d, Xw, Yw, Zw);

        // 包围盒检查
        if (m_config.useBoundingBox)
        {
            if (Xw < m_config.bboxMin[0] || Xw > m_config.bboxMax[0] ||
                Yw < m_config.bboxMin[1] || Yw > m_config.bboxMax[1] ||
                Zw < m_config.bboxMin[2] || Zw > m_config.bboxMax[2])
            {
                continue;
            }
        }

        // 标记为已融合
        fusedMask[fi][r * g.W + c] = 1;

        // 记录参考点
        if (!hasRef)
        {
            refX = Xw;
            refY = Yw;
            refZ = Zw;
            hasRef = true;
        }

        // 累积观测
        xs.push_back(Xw); ys.push_back(Yw); zs.push_back(Zw);

        // 累积法线
        if (!frames[fi].normalMap.empty())
        {
            const cv::Vec3f &n = frames[fi].normalMap.at<cv::Vec3f>(r, c);
            const float *invR = geom[fi].invR;
            float nw[3];
            for (int k = 0; k < 3; ++k)
            {
                nw[k] = invR[k*3+0]*n[0] + invR[k*3+1]*n[1] + invR[k*3+2]*n[2];
            }
            nxs.push_back(nw[0]);
            nys.push_back(nw[1]);
            nzs.push_back(nw[2]);
        }

        // 累积颜色：直接使用当前通过一致性检查的观测颜色，避免后续按“最近相机”重着色。
        if (fi >= 0 && fi < static_cast<int>(colorImages.size()) && !colorImages[fi].empty())
        {
            const cv::Mat &img = colorImages[fi];
            if (r >= 0 && r < img.rows && c >= 0 && c < img.cols)
            {
                if (img.channels() == 3)
                {
                    const cv::Vec3b &bgr = img.at<cv::Vec3b>(r, c);
                    rs_v.push_back(bgr[2]);
                    gs_v.push_back(bgr[1]);
                    bs_v.push_back(bgr[0]);
                }
                else if (img.channels() == 1)
                {
                    const uint8_t gray = img.at<uint8_t>(r, c);
                    rs_v.push_back(gray);
                    gs_v.push_back(gray);
                    bs_v.push_back(gray);
                }
                else
                {
                    rs_v.push_back(128);
                    gs_v.push_back(128);
                    bs_v.push_back(128);
                }
            }
            else
            {
                rs_v.push_back(128);
                gs_v.push_back(128);
                bs_v.push_back(128);
            }
        }
        else
        {
            rs_v.push_back(128);
            gs_v.push_back(128);
            bs_v.push_back(128);
        }

        // BFS 扩展到重叠视图
        if ((int)xs.size() < m_config.maxNumPixels && td < 100)
        {
            for (int overlapIdx : overlapping[fi])
            {
                const FrameGeometry &gO = geom[overlapIdx];
                // 将当前 3D 点投影到重叠视图
                float u_o, v_o;
                if (!gO.cameraModel.project(Xw, Yw, Zw, u_o, v_o))
                {
                    continue;
                }
                int iu = static_cast<int>(std::round(u_o));
                int iv = static_cast<int>(std::round(v_o));
                if (iu < 0 || iu >= gO.W || iv < 0 || iv >= gO.H)
                {
                    continue;
                }
                if (fusedMask[overlapIdx][iv * gO.W + iu])
                {
                    continue;
                }
                bfsQueue.push({overlapIdx, iv, iu, td + 1});
            }
        }
    }

    // 检查最少观测数
    if ((int)xs.size() < m_config.minNumPixels)
    {
        return false;
    }

    // 中位数聚合（COLMAP 关键设计：鲁棒于离群值）
    outPoint.x = median(xs);
    outPoint.y = median(ys);
    outPoint.z = median(zs);

    if (!nxs.empty())
    {
        outPoint.nx = median(nxs);
        outPoint.ny = median(nys);
        outPoint.nz = median(nzs);
        float nLen = std::sqrt(outPoint.nx*outPoint.nx +
                               outPoint.ny*outPoint.ny +
                               outPoint.nz*outPoint.nz);
        if (nLen > 1e-6f)
        {
            outPoint.nx /= nLen;
            outPoint.ny /= nLen;
            outPoint.nz /= nLen;
        }
    }

    // 颜色中位数
    if (!rs_v.empty())
    {
        std::vector<float> rf(rs_v.begin(), rs_v.end());
        std::vector<float> gf(gs_v.begin(), gs_v.end());
        std::vector<float> bf(bs_v.begin(), bs_v.end());
        outPoint.r = static_cast<uint8_t>(std::clamp(median(rf), 0.f, 255.f));
        outPoint.g = static_cast<uint8_t>(std::clamp(median(gf), 0.f, 255.f));
        outPoint.b = static_cast<uint8_t>(std::clamp(median(bf), 0.f, 255.f));
    }

    return true;
}

// =============================================================================
// 主融合接口：COLMAP StereoFusion::Run 风格
// =============================================================================
bool DepthMapFusion::fuse(
    const std::vector<FusionFrameInput> &frames,
    std::vector<FusedPoint>             &fusedPoints,
    MvsProgressCallback                  progressCb,
    std::string                         *errorMsg)
{
    if (frames.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "输入帧为空";
        }
        return false;
    }

    const int NF = static_cast<int>(frames.size());

    // 检查深度图
    for (int fi = 0; fi < NF; ++fi)
    {
        if (frames[fi].depthMap.empty())
        {
            if (errorMsg)
            {
                *errorMsg = "帧 " + std::to_string(fi) + " 深度图为空";
            }
            return false;
        }
    }

    fprintf(stderr, "[StereoFusion] 开始融合 %d 帧\n", NF);
    fprintf(stderr, "[StereoFusion] config: minNumPixels=%d maxReprojError=%.1f "
            "maxDepthError=%.3f maxNormalError=%.1f\n",
            m_config.minNumPixels, m_config.maxReprojError,
            m_config.maxDepthError, m_config.maxNormalError);

    // 1. 预计算投影几何
    std::vector<FrameGeometry> geom;
    prepareGeometry(frames, geom);

    // 2. 计算重叠图像
    std::vector<std::vector<int>> overlapping;
    computeOverlappingImages(frames, geom, overlapping);

    // 3. 初始化融合掩码
    std::vector<std::vector<char>> fusedMask(NF);
    for (int fi = 0; fi < NF; ++fi)
    {
        int W = geom[fi].W, H = geom[fi].H;
        fusedMask[fi].assign(W * H, 0);
    }

    // 4. 加载颜色图像（缓存避免重复读取）
    std::vector<cv::Mat> colorImages(NF);
    for (int fi = 0; fi < NF; ++fi)
    {
        if (!frames[fi].imagePath.empty())
        {
            colorImages[fi] = cv::imread(frames[fi].imagePath, cv::IMREAD_COLOR);
        }
    }

    // 5. 初始化每帧一致性过滤深度
    m_filteredDepths.resize(NF);
    for (int fi = 0; fi < NF; ++fi)
    {
        m_filteredDepths[fi] = cv::Mat::zeros(geom[fi].H, geom[fi].W, CV_32F);
    }

    // 6. 逐帧逐像素融合
    fusedPoints.clear();
    fusedPoints.reserve(100000);

    int totalProcessed = 0;

    for (int fi = 0; fi < NF; ++fi)
    {
        if (progressCb)
        {
            progressCb("融合 " + std::to_string(fi+1) + "/" + std::to_string(NF),
                       static_cast<float>(fi) / NF);
        }

        const int W = geom[fi].W;
        const int H = geom[fi].H;

        for (int r = 0; r < H; ++r)
        {
            for (int c = 0; c < W; ++c)
            {
                // 跳过已融合像素
                if (fusedMask[fi][r * W + c])
                {
                    continue;
                }

                // 跳过无效深度
                float d = frames[fi].depthMap.at<float>(r, c);
                if (d <= 0.f)
                {
                    continue;
                }

                // BFS 融合
                FusedPoint fp;
                if (!fusePixel(fi, r, c, frames, geom, overlapping, colorImages, fusedMask, fp))
                {
                    continue;
                }

                fusedPoints.push_back(fp);
                ++totalProcessed;

                // 记录该像素通过一致性
                m_filteredDepths[fi].at<float>(r, c) = d;
            }
        }

        int frameValid = cv::countNonZero(frames[fi].depthMap > 0);
        int frameFused = cv::countNonZero(m_filteredDepths[fi] > 0);
        fprintf(stderr, "[StereoFusion] 帧 %d: 有效深度=%d 融合点=%d (%.1f%%)\n",
                fi, frameValid, frameFused, 100.f * frameFused / std::max(1, frameValid));
    }

    fprintf(stderr, "[StereoFusion] 融合完成：总点数=%d\n", (int)fusedPoints.size());

    if (progressCb)
    {
        progressCb("融合完毕", 1.0f);
    }
    return true;
}

// =============================================================================
// 兼容旧接口：输出 DensePoint
// =============================================================================
bool DepthMapFusion::fuse(
    const std::vector<FusionFrameInput> &frames,
    std::vector<DensePoint>             &densePoints,
    MvsProgressCallback                  progressCb,
    std::string                         *errorMsg)
{
    std::vector<FusedPoint> fusedPts;
    if (!fuse(frames, fusedPts, progressCb, errorMsg))
        return false;

    densePoints.resize(fusedPts.size());
    for (size_t i = 0; i < fusedPts.size(); ++i) {
        densePoints[i].x = fusedPts[i].x;
        densePoints[i].y = fusedPts[i].y;
        densePoints[i].z = fusedPts[i].z;
        densePoints[i].r = fusedPts[i].r;
        densePoints[i].g = fusedPts[i].g;
        densePoints[i].b = fusedPts[i].b;
    }
    return true;
}

} // namespace mvs
} // namespace xjw
