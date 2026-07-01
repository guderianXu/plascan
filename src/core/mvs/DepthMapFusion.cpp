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
#include <array>
#include <numeric>
#include <cstdio>
#include <queue>
#include <atomic>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace xjw
{
namespace mvs
{

namespace
{

int resolveFusionWorkerCount(int requestedWorkers, int rowCount)
{
    if (rowCount <= 0)
    {
        return 1;
    }
    int desired = requestedWorkers;
    if (desired <= 0)
    {
        desired = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (desired <= 0)
    {
        desired = 1;
    }
    return std::clamp(desired, 1, rowCount);
}

bool isFusionCancelled(const StereoFusionConfig &config)
{
    return config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed);
}

uint8_t medianByte2(uint8_t first, uint8_t second, bool hasSecond)
{
    return hasSecond ? std::max(first, second) : first;
}

float medianScalar2(float first, float second, bool hasSecond)
{
    return hasSecond ? std::max(first, second) : first;
}

class ColorImageCache
{
public:
    ColorImageCache(const std::vector<FusionFrameInput> &frames,
                    bool useColor,
                    int capacity)
        : m_frames(frames)
        , m_useColor(useColor)
        , m_capacity(std::max(0, capacity))
    {
    }

    cv::Mat get(int frameIdx)
    {
        if (!m_useColor || frameIdx < 0 || frameIdx >= static_cast<int>(m_frames.size()))
        {
            return cv::Mat();
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache.find(frameIdx);
        if (it != m_cache.end())
        {
            touch(it);
            return it->second.image;
        }

        cv::Mat image;
        const std::string &path = m_frames[frameIdx].imagePath;
        if (!path.empty())
        {
            image = cv::imread(path, cv::IMREAD_COLOR);
        }
        if (!image.empty()
            && m_frames[frameIdx].imgW > 0
            && m_frames[frameIdx].imgH > 0
            && (image.cols != m_frames[frameIdx].imgW || image.rows != m_frames[frameIdx].imgH))
        {
            cv::resize(image, image,
                       cv::Size(m_frames[frameIdx].imgW, m_frames[frameIdx].imgH),
                       0.0,
                       0.0,
                       cv::INTER_AREA);
        }

        if (m_capacity > 0)
        {
            m_lru.push_front(frameIdx);
            CacheEntry entry;
            entry.image = image;
            entry.lruIt = m_lru.begin();
            m_cache.emplace(frameIdx, std::move(entry));
            evictIfNeeded();
        }
        return image;
    }

private:
    struct CacheEntry
    {
        cv::Mat image;
        std::list<int>::iterator lruIt;
    };

    void touch(std::unordered_map<int, CacheEntry>::iterator it)
    {
        m_lru.erase(it->second.lruIt);
        m_lru.push_front(it->first);
        it->second.lruIt = m_lru.begin();
    }

    void evictIfNeeded()
    {
        while (m_capacity > 0 && static_cast<int>(m_cache.size()) > m_capacity)
        {
            const int victim = m_lru.back();
            m_lru.pop_back();
            m_cache.erase(victim);
        }
    }

    const std::vector<FusionFrameInput> &m_frames;
    bool m_useColor = true;
    int m_capacity = 4;
    std::mutex m_mutex;
    std::list<int> m_lru;
    std::unordered_map<int, CacheEntry> m_cache;
};

} // namespace

// =============================================================================
DepthMapFusion::DepthMapFusion(const StereoFusionConfig &config)
    : _config(config)
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
        if (!frames[fi].sourceImageIndices.empty())
        {
            overlapping[fi].reserve(std::min(_config.checkNumImages,
                                             static_cast<int>(frames[fi].sourceImageIndices.size())));
            for (int sourceIdx : frames[fi].sourceImageIndices)
            {
                if (sourceIdx < 0 || sourceIdx >= NF || sourceIdx == fi)
                {
                    continue;
                }
                if (std::find(overlapping[fi].begin(), overlapping[fi].end(), sourceIdx) != overlapping[fi].end())
                {
                    continue;
                }
                overlapping[fi].push_back(sourceIdx);
                if (static_cast<int>(overlapping[fi].size()) >= _config.checkNumImages)
                {
                    break;
                }
            }
            if (!overlapping[fi].empty())
            {
                continue;
            }
        }

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

        int numOverlap = std::min(_config.checkNumImages, (int)candidates.size());
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
    const std::function<cv::Mat(int)> &colorProvider,
    std::vector<std::vector<char>> &fusedMask,
    FusedPoint &outPoint)
{
    const float maxReprojSq = _config.maxReprojError * _config.maxReprojError;
    const float maxDepthErr = _config.maxDepthError;
    const float cosMaxNormErr = std::cos(_config.maxNormalError * (float)M_PI / 180.f);

    // BFS 数据
    struct QueueItem
    {
        int imgIdx;
        int row;
        int col;
        int depth;
    };
    struct AcceptedPixel
    {
        int frameIdx;
        int row;
        int col;
    };
    std::queue<QueueItem> bfsQueue;
    bfsQueue.push({imageIdx, row, col, 0});

    // 收集的观测
    std::vector<float> xs, ys, zs;
    std::vector<float> nxs, nys, nzs;
    std::vector<uint8_t> rs_v, gs_v, bs_v;
    std::vector<AcceptedPixel> acceptedPixels;

    // 参考点
    float refX = 0.f, refY = 0.f, refZ = 0.f;
    bool hasRef = false;

    while (!bfsQueue.empty() && (int)xs.size() < _config.maxNumPixels)
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

        const int pixelIdx = r * g.W + c;

        // 已融合检查
        if (fusedMask[fi][pixelIdx])
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
        if (_config.useBoundingBox)
        {
            if (Xw < _config.bboxMin[0] || Xw > _config.bboxMax[0] ||
                Yw < _config.bboxMin[1] || Yw > _config.bboxMax[1] ||
                Zw < _config.bboxMin[2] || Zw > _config.bboxMax[2])
            {
                continue;
            }
        }

        acceptedPixels.push_back({fi, r, c});
        fusedMask[fi][pixelIdx] = 2;

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
        cv::Mat img = colorProvider ? colorProvider(fi) : cv::Mat();
        if (!img.empty())
        {
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
        if ((int)xs.size() < _config.maxNumPixels && td < 100)
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
    if ((int)xs.size() < _config.minNumPixels)
    {
        for (const AcceptedPixel &pixel : acceptedPixels)
        {
            const FrameGeometry &g = geom[pixel.frameIdx];
            fusedMask[pixel.frameIdx][pixel.row * g.W + pixel.col] = 0;
        }
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

    for (const AcceptedPixel &pixel : acceptedPixels)
    {
        const FrameGeometry &g = geom[pixel.frameIdx];
        fusedMask[pixel.frameIdx][pixel.row * g.W + pixel.col] = 1;
        if (pixel.frameIdx >= 0 && pixel.frameIdx < static_cast<int>(_filteredDepths.size()))
        {
            _filteredDepths[pixel.frameIdx].at<float>(pixel.row, pixel.col) =
                frames[pixel.frameIdx].depthMap.at<float>(pixel.row, pixel.col);
        }
    }

    return true;
}

bool DepthMapFusion::fuseTwoViewSingleObservationFast(
    const std::vector<FusionFrameInput> &frames,
    const std::vector<FrameGeometry> &geom,
    const std::function<cv::Mat(int)> &colorProvider,
    std::vector<FusedPoint> &fusedPoints,
    MvsProgressCallback progressCb)
{
    constexpr int kFrameCount = 2;
    const int maxRows = std::max(geom[0].H, geom[1].H);
    const int workerCount = resolveFusionWorkerCount(_config.workerCount, maxRows);
    const float maxReprojSq = _config.maxReprojError * _config.maxReprojError;
    const float cosMaxNormErr = std::cos(_config.maxNormalError * static_cast<float>(M_PI) / 180.f);

    fprintf(stderr,
            "[StereoFusion] 快速并行融合: frames=2 workers=%d minNumPixels=%d\n",
            workerCount,
            _config.minNumPixels);

    std::size_t reserveCount = 0;
    for (int fi = 0; fi < kFrameCount; ++fi)
    {
        reserveCount += static_cast<std::size_t>(cv::countNonZero(frames[fi].depthMap > 0));
    }
    fusedPoints.clear();
    fusedPoints.reserve(reserveCount);

    std::vector<std::unique_ptr<std::atomic<unsigned char>[]>> claimed(kFrameCount);
    for (int fi = 0; fi < kFrameCount; ++fi)
    {
        const std::size_t pixelCount = static_cast<std::size_t>(geom[fi].W) *
                                       static_cast<std::size_t>(geom[fi].H);
        claimed[fi] = std::make_unique<std::atomic<unsigned char>[]>(pixelCount);
        for (std::size_t idx = 0; idx < pixelCount; ++idx)
        {
            claimed[fi][idx].store(0, std::memory_order_relaxed);
        }
    }

    auto tryClaim = [&](int frameIdx, int pixelIdx) {
        unsigned char expected = 0;
        return claimed[frameIdx][static_cast<std::size_t>(pixelIdx)].compare_exchange_strong(
            expected,
            1,
            std::memory_order_relaxed,
            std::memory_order_relaxed);
    };

    auto inBounds = [&](int frameIdx, int row, int col) {
        return row >= 0 && row < geom[frameIdx].H && col >= 0 && col < geom[frameIdx].W;
    };

    auto inBoundingBox = [&](float x, float y, float z) {
        if (!_config.useBoundingBox)
        {
            return true;
        }
        return x >= _config.bboxMin[0] && x <= _config.bboxMax[0] &&
               y >= _config.bboxMin[1] && y <= _config.bboxMax[1] &&
               z >= _config.bboxMin[2] && z <= _config.bboxMax[2];
    };

    auto readColor = [&](int frameIdx, int row, int col, uint8_t &r, uint8_t &g, uint8_t &b) {
        r = 128;
        g = 128;
        b = 128;
        cv::Mat img = colorProvider ? colorProvider(frameIdx) : cv::Mat();
        if (img.empty() || row < 0 || row >= img.rows || col < 0 || col >= img.cols)
        {
            return;
        }
        if (img.channels() == 3)
        {
            const cv::Vec3b &bgr = img.at<cv::Vec3b>(row, col);
            r = bgr[2];
            g = bgr[1];
            b = bgr[0];
        }
        else if (img.channels() == 1)
        {
            const uint8_t gray = img.at<uint8_t>(row, col);
            r = gray;
            g = gray;
            b = gray;
        }
    };

    auto readWorldNormal = [&](int frameIdx, int row, int col, float &nx, float &ny, float &nz) {
        if (frames[frameIdx].normalMap.empty())
        {
            return false;
        }
        const cv::Vec3f &n = frames[frameIdx].normalMap.at<cv::Vec3f>(row, col);
        const float *invR = geom[frameIdx].invR;
        nx = invR[0] * n[0] + invR[1] * n[1] + invR[2] * n[2];
        ny = invR[3] * n[0] + invR[4] * n[1] + invR[5] * n[2];
        nz = invR[6] * n[0] + invR[7] * n[1] + invR[8] * n[2];
        return true;
    };

    for (int fi = 0; fi < kFrameCount; ++fi)
    {
        if (isFusionCancelled(_config))
        {
            return false;
        }
        if (progressCb)
        {
            progressCb("快速并行融合 " + std::to_string(fi + 1) + "/2",
                       static_cast<float>(fi) / static_cast<float>(kFrameCount));
        }

        const int other = 1 - fi;
        const int W = geom[fi].W;
        const int H = geom[fi].H;
        const int frameWorkers = resolveFusionWorkerCount(workerCount, H);
        std::atomic<int> nextRow{0};
        std::vector<std::vector<FusedPoint>> workerOutputs(static_cast<std::size_t>(frameWorkers));
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(frameWorkers));

        for (int worker = 0; worker < frameWorkers; ++worker)
        {
            threads.emplace_back([&, worker]() {
                auto &localPoints = workerOutputs[static_cast<std::size_t>(worker)];
                localPoints.reserve(static_cast<std::size_t>(W) *
                                    static_cast<std::size_t>(std::max(1, H / frameWorkers + 1)));

                for (;;)
                {
                    if (isFusionCancelled(_config))
                    {
                        break;
                    }
                    const int row = nextRow.fetch_add(1, std::memory_order_relaxed);
                    if (row >= H)
                    {
                        break;
                    }
                    const float *depthRow = frames[fi].depthMap.ptr<float>(row);
                    for (int col = 0; col < W; ++col)
                    {
                        const float depth = depthRow[col];
                        if (depth <= 0.f)
                        {
                            continue;
                        }

                        float x0, y0, z0;
                        geom[fi].cameraModel.unproject(
                            static_cast<float>(col),
                            static_cast<float>(row),
                            depth,
                            x0,
                            y0,
                            z0);
                        if (!inBoundingBox(x0, y0, z0))
                        {
                            continue;
                        }

                        const int pixelIdx = row * W + col;
                        if (!tryClaim(fi, pixelIdx))
                        {
                            continue;
                        }

                        uint8_t r0, g0, b0;
                        readColor(fi, row, col, r0, g0, b0);

                        float nx0 = 0.f, ny0 = 0.f, nz0 = 0.f;
                        const bool hasNormal0 = readWorldNormal(fi, row, col, nx0, ny0, nz0);

                        bool hasMatch = false;
                        float x1 = 0.f, y1 = 0.f, z1 = 0.f;
                        float nx1 = 0.f, ny1 = 0.f, nz1 = 0.f;
                        bool hasNormal1 = false;
                        uint8_t r1 = 128, g1 = 128, b1 = 128;

                        float uOther = 0.f;
                        float vOther = 0.f;
                        if (geom[other].cameraModel.project(x0, y0, z0, uOther, vOther))
                        {
                            const int otherCol = static_cast<int>(std::round(uOther));
                            const int otherRow = static_cast<int>(std::round(vOther));
                            if (inBounds(other, otherRow, otherCol))
                            {
                                const int otherIdx = otherRow * geom[other].W + otherCol;
                                const float du = uOther - static_cast<float>(otherCol);
                                const float dv = vOther - static_cast<float>(otherRow);
                                const float otherDepth = frames[other].depthMap.at<float>(otherRow, otherCol);
                                const float zExpected = geom[other].cameraModel.R_cw[6] * x0 +
                                                        geom[other].cameraModel.R_cw[7] * y0 +
                                                        geom[other].cameraModel.R_cw[8] * z0 +
                                                        geom[other].cameraModel.T[2];
                                bool consistent = otherDepth > 0.f &&
                                                  zExpected > 0.f &&
                                                  du * du + dv * dv <= maxReprojSq &&
                                                  std::fabs(otherDepth - zExpected) / zExpected <=
                                                      _config.maxDepthError;
                                if (consistent && hasNormal0 && !frames[other].normalMap.empty())
                                {
                                    float tx = 0.f, ty = 0.f, tz = 0.f;
                                    if (readWorldNormal(other, otherRow, otherCol, tx, ty, tz))
                                    {
                                        const float cosAngle = nx0 * tx + ny0 * ty + nz0 * tz;
                                        consistent = cosAngle >= cosMaxNormErr;
                                    }
                                }
                                if (consistent)
                                {
                                    geom[other].cameraModel.unproject(
                                        static_cast<float>(otherCol),
                                        static_cast<float>(otherRow),
                                        otherDepth,
                                        x1,
                                        y1,
                                        z1);
                                    consistent = inBoundingBox(x1, y1, z1);
                                }
                                if (consistent && tryClaim(other, otherIdx))
                                {
                                    hasMatch = true;
                                    readColor(other, otherRow, otherCol, r1, g1, b1);
                                    hasNormal1 = readWorldNormal(other, otherRow, otherCol, nx1, ny1, nz1);
                                }
                            }
                        }

                        FusedPoint fp;
                        fp.x = medianScalar2(x0, x1, hasMatch);
                        fp.y = medianScalar2(y0, y1, hasMatch);
                        fp.z = medianScalar2(z0, z1, hasMatch);
                        fp.r = medianByte2(r0, r1, hasMatch);
                        fp.g = medianByte2(g0, g1, hasMatch);
                        fp.b = medianByte2(b0, b1, hasMatch);

                        if (hasNormal0 || hasNormal1)
                        {
                            if (hasNormal0 && hasNormal1)
                            {
                                fp.nx = medianScalar2(nx0, nx1, true);
                                fp.ny = medianScalar2(ny0, ny1, true);
                                fp.nz = medianScalar2(nz0, nz1, true);
                            }
                            else if (hasNormal0)
                            {
                                fp.nx = nx0;
                                fp.ny = ny0;
                                fp.nz = nz0;
                            }
                            else
                            {
                                fp.nx = nx1;
                                fp.ny = ny1;
                                fp.nz = nz1;
                            }
                            const float nLen = std::sqrt(fp.nx * fp.nx + fp.ny * fp.ny + fp.nz * fp.nz);
                            if (nLen > 1e-6f)
                            {
                                fp.nx /= nLen;
                                fp.ny /= nLen;
                                fp.nz /= nLen;
                            }
                        }

                        localPoints.push_back(fp);
                        _filteredDepths[fi].at<float>(row, col) = depth;
                    }
                }
            });
        }

        for (std::thread &thread : threads)
        {
            thread.join();
        }
        if (isFusionCancelled(_config))
        {
            return false;
        }

        for (auto &localPoints : workerOutputs)
        {
            fusedPoints.insert(fusedPoints.end(),
                               std::make_move_iterator(localPoints.begin()),
                               std::make_move_iterator(localPoints.end()));
        }

        const int frameValid = cv::countNonZero(frames[fi].depthMap > 0);
        const int frameFused = cv::countNonZero(_filteredDepths[fi] > 0);
        fprintf(stderr,
                "[StereoFusion] 快速并行帧 %d: 有效深度=%d 融合点=%d (%.1f%%)\n",
                fi,
                frameValid,
                frameFused,
                100.f * frameFused / std::max(1, frameValid));
    }

    fprintf(stderr, "[StereoFusion] 快速并行融合完成：总点数=%d\n", (int)fusedPoints.size());
    if (progressCb)
    {
        progressCb("快速并行融合完毕", 1.0f);
    }
    return true;
}

bool DepthMapFusion::fuseFirstFrameObservationsFast(
    const std::vector<FusionFrameInput> &frames,
    const std::vector<FrameGeometry> &geom,
    const std::function<cv::Mat(int)> &colorProvider,
    std::vector<FusedPoint> &fusedPoints,
    MvsProgressCallback progressCb)
{
    if (frames.empty() || geom.empty())
    {
        return false;
    }

    const int fi = 0;
    const int W = geom[fi].W;
    const int H = geom[fi].H;
    if (W <= 0 || H <= 0 || frames[fi].depthMap.empty())
    {
        fusedPoints.clear();
        return true;
    }

    const int workerCount = resolveFusionWorkerCount(_config.workerCount, H);
    const cv::Mat colorImage = colorProvider ? colorProvider(fi) : cv::Mat();
    const bool hasColor = !colorImage.empty();
    const bool hasNormals = !frames[fi].normalMap.empty();
    std::atomic<int> nextRow{0};
    const int requiredObservations = std::max(1, _config.minNumPixels);
    const bool requireNeighborAgreement = requiredObservations > 1 && frames.size() > 1;
    const float maxReprojSq = _config.maxReprojError * _config.maxReprojError;
    const float cosMaxNormErr = std::cos(_config.maxNormalError * static_cast<float>(M_PI) / 180.0f);

    std::vector<int> neighborFrames;
    neighborFrames.reserve(frames.size() > 0 ? frames.size() - 1 : 0);
    if (requireNeighborAgreement && !frames[fi].sourceImageIndices.empty())
    {
        for (int plannedViewIndex : frames[fi].sourceImageIndices)
        {
            for (int localIndex = 1; localIndex < static_cast<int>(frames.size()); ++localIndex)
            {
                if (frames[localIndex].viewIndex == plannedViewIndex)
                {
                    neighborFrames.push_back(localIndex);
                    break;
                }
            }
        }
    }
    if (requireNeighborAgreement && neighborFrames.empty())
    {
        for (int localIndex = 1; localIndex < static_cast<int>(frames.size()); ++localIndex)
        {
            neighborFrames.push_back(localIndex);
        }
    }
    if (requireNeighborAgreement && _config.checkNumImages > 0 &&
        static_cast<int>(neighborFrames.size()) > _config.checkNumImages)
    {
        neighborFrames.resize(static_cast<std::size_t>(_config.checkNumImages));
    }

    const int validCount = cv::countNonZero(frames[fi].depthMap > 0);
    fusedPoints.clear();
    fusedPoints.reserve(static_cast<std::size_t>(std::max(0, validCount)));
    _filteredDepths.resize(frames.size());
    _filteredDepths[fi] = cv::Mat::zeros(H, W, CV_32F);

    fprintf(stderr,
            "[StereoFusion] 使用已过滤深度图快速反投影: frame=0 size=%dx%d valid=%d workers=%d minNumPixels=%d neighbors=%d\n",
            W,
            H,
            validCount,
            workerCount,
            requiredObservations,
            static_cast<int>(neighborFrames.size()));

    auto readWorldNormal = [&](int frameIdx, int row, int col, float &nx, float &ny, float &nz) {
        if (frameIdx < 0 || frameIdx >= static_cast<int>(frames.size()) ||
            frames[frameIdx].normalMap.empty())
        {
            return false;
        }
        const cv::Vec3f &normal = frames[frameIdx].normalMap.at<cv::Vec3f>(row, col);
        const float *frameInvR = geom[frameIdx].invR;
        nx = frameInvR[0] * normal[0] + frameInvR[1] * normal[1] + frameInvR[2] * normal[2];
        ny = frameInvR[3] * normal[0] + frameInvR[4] * normal[1] + frameInvR[5] * normal[2];
        nz = frameInvR[6] * normal[0] + frameInvR[7] * normal[1] + frameInvR[8] * normal[2];
        const float normLen = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (normLen <= 1e-6f)
        {
            nx = 0.0f;
            ny = 0.0f;
            nz = 0.0f;
            return false;
        }
        nx /= normLen;
        ny /= normLen;
        nz /= normLen;
        return true;
    };

    auto estimateWorldNormalFromDepth = [&](int frameIdx, int row, int col, float &nx, float &ny, float &nz) {
        if (frameIdx < 0 || frameIdx >= static_cast<int>(frames.size()) ||
            frames[frameIdx].depthMap.empty())
        {
            return false;
        }
        const FrameGeometry &frameGeom = geom[frameIdx];
        if (frameGeom.W < 2 || frameGeom.H < 2 ||
            row < 0 || row >= frameGeom.H || col < 0 || col >= frameGeom.W)
        {
            return false;
        }
        const int leftCol = std::max(0, col - 1);
        const int rightCol = std::min(frameGeom.W - 1, col + 1);
        const int upRow = std::max(0, row - 1);
        const int downRow = std::min(frameGeom.H - 1, row + 1);
        if (leftCol == rightCol || upRow == downRow)
        {
            return false;
        }

        auto unprojectPixel = [&](int sampleRow, int sampleCol, float &x, float &y, float &z) {
            const float sampleDepth = frames[frameIdx].depthMap.at<float>(sampleRow, sampleCol);
            if (sampleDepth <= 0.0f)
            {
                return false;
            }
            frameGeom.cameraModel.unproject(
                static_cast<float>(sampleCol),
                static_cast<float>(sampleRow),
                sampleDepth,
                x,
                y,
                z);
            return true;
        };

        float xl = 0.0f, yl = 0.0f, zl = 0.0f;
        float xr = 0.0f, yr = 0.0f, zr = 0.0f;
        float xu = 0.0f, yu = 0.0f, zu = 0.0f;
        float xd = 0.0f, yd = 0.0f, zd = 0.0f;
        if (!unprojectPixel(row, leftCol, xl, yl, zl) ||
            !unprojectPixel(row, rightCol, xr, yr, zr) ||
            !unprojectPixel(upRow, col, xu, yu, zu) ||
            !unprojectPixel(downRow, col, xd, yd, zd))
        {
            return false;
        }

        const float ax = xr - xl;
        const float ay = yr - yl;
        const float az = zr - zl;
        const float bx = xd - xu;
        const float by = yd - yu;
        const float bz = zd - zu;
        nx = ay * bz - az * by;
        ny = az * bx - ax * bz;
        nz = ax * by - ay * bx;
        const float normLen = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (normLen <= 1e-6f)
        {
            nx = 0.0f;
            ny = 0.0f;
            nz = 0.0f;
            return false;
        }
        nx /= normLen;
        ny /= normLen;
        nz /= normLen;
        return true;
    };

    auto medianArray = [](std::array<float, 64> values, int count) {
        count = std::clamp(count, 1, static_cast<int>(values.size()));
        const int mid = count / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.begin() + count);
        return values[static_cast<std::size_t>(mid)];
    };

    std::vector<std::vector<FusedPoint>> workerOutputs(static_cast<std::size_t>(workerCount));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workerCount));

    for (int worker = 0; worker < workerCount; ++worker)
    {
        threads.emplace_back([&, worker]() {
            auto &localPoints = workerOutputs[static_cast<std::size_t>(worker)];
            localPoints.reserve(static_cast<std::size_t>(W) *
                                static_cast<std::size_t>(std::max(1, H / workerCount + 1)));

            for (;;)
            {
                if (isFusionCancelled(_config))
                {
                    break;
                }

                const int row = nextRow.fetch_add(1, std::memory_order_relaxed);
                if (row >= H)
                {
                    break;
                }

                const float *depthRow = frames[fi].depthMap.ptr<float>(row);
                float *filteredRow = _filteredDepths[fi].ptr<float>(row);
                for (int col = 0; col < W; ++col)
                {
                    const float depth = depthRow[col];
                    if (depth <= 0.0f)
                    {
                        continue;
                    }

                    FusedPoint point;
                    geom[fi].cameraModel.unproject(
                        static_cast<float>(col),
                        static_cast<float>(row),
                        depth,
                        point.x,
                        point.y,
                        point.z);

                    if (_config.useBoundingBox &&
                        (point.x < _config.bboxMin[0] || point.x > _config.bboxMax[0] ||
                         point.y < _config.bboxMin[1] || point.y > _config.bboxMax[1] ||
                         point.z < _config.bboxMin[2] || point.z > _config.bboxMax[2]))
                    {
                        continue;
                    }

                    bool hasPointNormal = false;
                    if (hasNormals)
                    {
                        hasPointNormal = readWorldNormal(fi, row, col, point.nx, point.ny, point.nz);
                    }
                    if (!hasPointNormal)
                    {
                        hasPointNormal = estimateWorldNormalFromDepth(fi, row, col, point.nx, point.ny, point.nz);
                    }

                    point.r = 128;
                    point.g = 128;
                    point.b = 128;
                    if (hasColor && row < colorImage.rows && col < colorImage.cols)
                    {
                        if (colorImage.channels() == 3)
                        {
                            const cv::Vec3b &bgr = colorImage.at<cv::Vec3b>(row, col);
                            point.r = bgr[2];
                            point.g = bgr[1];
                            point.b = bgr[0];
                        }
                        else if (colorImage.channels() == 1)
                        {
                            const uint8_t gray = colorImage.at<uint8_t>(row, col);
                            point.r = gray;
                            point.g = gray;
                            point.b = gray;
                        }
                    }

                    if (requireNeighborAgreement)
                    {
                        constexpr int kMaxObservations = 64;
                        std::array<float, kMaxObservations> xs{};
                        std::array<float, kMaxObservations> ys{};
                        std::array<float, kMaxObservations> zs{};
                        std::array<float, kMaxObservations> nxs{};
                        std::array<float, kMaxObservations> nys{};
                        std::array<float, kMaxObservations> nzs{};
                        int observationCount = 1;
                        int normalCount = 0;
                        xs[0] = point.x;
                        ys[0] = point.y;
                        zs[0] = point.z;
                        if (hasPointNormal)
                        {
                            nxs[0] = point.nx;
                            nys[0] = point.ny;
                            nzs[0] = point.nz;
                            normalCount = 1;
                        }

                        for (int otherFrame : neighborFrames)
                        {
                            if (observationCount >= kMaxObservations)
                            {
                                break;
                            }
                            const FrameGeometry &otherGeom = geom[otherFrame];
                            float uOther = 0.0f;
                            float vOther = 0.0f;
                            if (!otherGeom.cameraModel.project(point.x, point.y, point.z, uOther, vOther))
                            {
                                continue;
                            }

                            const int otherCol = static_cast<int>(std::round(uOther));
                            const int otherRow = static_cast<int>(std::round(vOther));
                            if (otherRow < 0 || otherRow >= otherGeom.H ||
                                otherCol < 0 || otherCol >= otherGeom.W)
                            {
                                continue;
                            }
                            const float du = uOther - static_cast<float>(otherCol);
                            const float dv = vOther - static_cast<float>(otherRow);
                            if (du * du + dv * dv > maxReprojSq)
                            {
                                continue;
                            }

                            const float otherDepth =
                                frames[otherFrame].depthMap.at<float>(otherRow, otherCol);
                            const float expectedDepth =
                                otherGeom.cameraModel.R_cw[6] * point.x +
                                otherGeom.cameraModel.R_cw[7] * point.y +
                                otherGeom.cameraModel.R_cw[8] * point.z +
                                otherGeom.cameraModel.T[2];
                            if (otherDepth <= 0.0f || expectedDepth <= 0.0f ||
                                std::fabs(otherDepth - expectedDepth) / expectedDepth > _config.maxDepthError)
                            {
                                continue;
                            }

                            float otherNx = 0.0f;
                            float otherNy = 0.0f;
                            float otherNz = 0.0f;
                            bool hasOtherNormal =
                                readWorldNormal(otherFrame, otherRow, otherCol, otherNx, otherNy, otherNz);
                            if (!hasOtherNormal)
                            {
                                hasOtherNormal = estimateWorldNormalFromDepth(
                                    otherFrame,
                                    otherRow,
                                    otherCol,
                                    otherNx,
                                    otherNy,
                                    otherNz);
                            }
                            if (hasPointNormal && hasOtherNormal)
                            {
                                const float cosAngle =
                                    point.nx * otherNx + point.ny * otherNy + point.nz * otherNz;
                                if (cosAngle < cosMaxNormErr)
                                {
                                    continue;
                                }
                            }

                            float otherX = 0.0f;
                            float otherY = 0.0f;
                            float otherZ = 0.0f;
                            otherGeom.cameraModel.unproject(
                                static_cast<float>(otherCol),
                                static_cast<float>(otherRow),
                                otherDepth,
                                otherX,
                                otherY,
                                otherZ);
                            if (_config.useBoundingBox &&
                                (otherX < _config.bboxMin[0] || otherX > _config.bboxMax[0] ||
                                 otherY < _config.bboxMin[1] || otherY > _config.bboxMax[1] ||
                                 otherZ < _config.bboxMin[2] || otherZ > _config.bboxMax[2]))
                            {
                                continue;
                            }

                            const int obs = observationCount++;
                            xs[static_cast<std::size_t>(obs)] = otherX;
                            ys[static_cast<std::size_t>(obs)] = otherY;
                            zs[static_cast<std::size_t>(obs)] = otherZ;
                            if (hasOtherNormal && normalCount < kMaxObservations)
                            {
                                nxs[static_cast<std::size_t>(normalCount)] = otherNx;
                                nys[static_cast<std::size_t>(normalCount)] = otherNy;
                                nzs[static_cast<std::size_t>(normalCount)] = otherNz;
                                ++normalCount;
                            }
                        }

                        if (observationCount < requiredObservations)
                        {
                            continue;
                        }

                        point.x = medianArray(xs, observationCount);
                        point.y = medianArray(ys, observationCount);
                        point.z = medianArray(zs, observationCount);
                        if (normalCount > 0)
                        {
                            point.nx = medianArray(nxs, normalCount);
                            point.ny = medianArray(nys, normalCount);
                            point.nz = medianArray(nzs, normalCount);
                            const float normLen =
                                std::sqrt(point.nx * point.nx + point.ny * point.ny + point.nz * point.nz);
                            if (normLen > 1e-6f)
                            {
                                point.nx /= normLen;
                                point.ny /= normLen;
                                point.nz /= normLen;
                            }
                        }
                    }

                    localPoints.push_back(point);
                    filteredRow[col] = depth;
                }

                if (worker == 0 && progressCb && (row % 128 == 0))
                {
                    progressCb("快速反投影已过滤深度图",
                               static_cast<float>(row) / static_cast<float>(std::max(1, H)));
                }
            }
        });
    }

    for (std::thread &thread : threads)
    {
        thread.join();
    }

    if (isFusionCancelled(_config))
    {
        return false;
    }

    for (auto &localPoints : workerOutputs)
    {
        fusedPoints.insert(fusedPoints.end(),
                           std::make_move_iterator(localPoints.begin()),
                           std::make_move_iterator(localPoints.end()));
    }

    const float yieldRatio = validCount > 0
        ? static_cast<float>(fusedPoints.size()) / static_cast<float>(validCount)
        : 1.0f;
    if (_config.enableLowYieldFallback
        && requireNeighborAgreement
        && requiredObservations > 2
        && validCount > 0
        && yieldRatio < std::max(0.0f, _config.lowYieldFallbackMinRatio)
        && !isFusionCancelled(_config))
    {
        const int fallbackRequiredObservations =
            std::clamp(_config.lowYieldFallbackMinNumPixels, 2, requiredObservations - 1);
        fprintf(stderr,
                "[StereoFusion] 严格流式融合产出过低: points=%d valid=%d ratio=%.4f, "
                "fallback minNumPixels=%d\n",
                static_cast<int>(fusedPoints.size()),
                validCount,
                yieldRatio,
                fallbackRequiredObservations);

        StereoFusionConfig savedConfig = _config;
        _config.minNumPixels = fallbackRequiredObservations;
        _config.enableLowYieldFallback = false;

        std::vector<FusedPoint> fallbackPoints;
        const bool fallbackOk = fuseFirstFrameObservationsFast(frames,
                                                               geom,
                                                               colorProvider,
                                                               fallbackPoints,
                                                               progressCb);
        _config = savedConfig;

        if (!fallbackOk)
        {
            return false;
        }
        if (fallbackPoints.size() > fusedPoints.size())
        {
            fusedPoints = std::move(fallbackPoints);
            fprintf(stderr,
                    "[StereoFusion] 已采用双视一致 fallback: points=%d\n",
                    static_cast<int>(fusedPoints.size()));
        }
    }

    fprintf(stderr,
            "[StereoFusion] 快速反投影完成: points=%d\n",
            static_cast<int>(fusedPoints.size()));
    if (progressCb)
    {
        progressCb("快速反投影已过滤深度图完成", 1.0f);
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
    fusedPoints.clear();
    _filteredDepths.clear();

    if (frames.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "输入帧为空";
        }
        return false;
    }

    if (isFusionCancelled(_config))
    {
        if (errorMsg)
        {
            *errorMsg = "用户取消深度图融合";
        }
        return false;
    }

    const int NF = static_cast<int>(frames.size());

    // 检查深度图
    for (int fi = 0; fi < NF; ++fi)
    {
        if (isFusionCancelled(_config))
        {
            if (errorMsg)
            {
                *errorMsg = "用户取消深度图融合";
            }
            return false;
        }
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
            "maxDepthError=%.3f maxNormalError=%.1f workerCount=%d\n",
            _config.minNumPixels, _config.maxReprojError,
            _config.maxDepthError, _config.maxNormalError,
            resolveFusionWorkerCount(_config.workerCount, frames.front().depthMap.rows));

    // 1. 预计算投影几何
    std::vector<FrameGeometry> geom;
    prepareGeometry(frames, geom);
    if (isFusionCancelled(_config))
    {
        if (errorMsg)
        {
            *errorMsg = "用户取消深度图融合";
        }
        return false;
    }

    if (_config.fuseOnlyFirstFrame)
    {
        ColorImageCache colorCache(frames, _config.useColor, _config.colorCacheCapacity);
        auto colorProvider = [&colorCache](int frameIdx) {
            return colorCache.get(frameIdx);
        };
        const bool fastOk = fuseFirstFrameObservationsFast(frames,
                                                           geom,
                                                           colorProvider,
                                                           fusedPoints,
                                                           progressCb);
        if (!fastOk && isFusionCancelled(_config) && errorMsg)
        {
            *errorMsg = "用户取消深度图融合";
        }
        return fastOk;
    }

    // 2. 计算重叠图像
    std::vector<std::vector<int>> overlapping;
    computeOverlappingImages(frames, geom, overlapping);
    if (isFusionCancelled(_config))
    {
        if (errorMsg)
        {
            *errorMsg = "用户取消深度图融合";
        }
        return false;
    }

    // 3. 初始化融合掩码
    std::vector<std::vector<char>> fusedMask(NF);
    for (int fi = 0; fi < NF; ++fi)
    {
        if (isFusionCancelled(_config))
        {
            if (errorMsg)
            {
                *errorMsg = "用户取消深度图融合";
            }
            return false;
        }
        int W = geom[fi].W, H = geom[fi].H;
        fusedMask[fi].assign(W * H, 0);
    }

    // 4. 彩色图懒加载；只有融合点需要赋色时才读取原图，并通过 LRU 控制内存。
    ColorImageCache colorCache(frames, _config.useColor, _config.colorCacheCapacity);
    auto colorProvider = [&colorCache](int frameIdx) {
        return colorCache.get(frameIdx);
    };

    // 5. 初始化每帧一致性过滤深度
    _filteredDepths.resize(NF);
    for (int fi = 0; fi < NF; ++fi)
    {
        if (isFusionCancelled(_config))
        {
            if (errorMsg)
            {
                *errorMsg = "用户取消深度图融合";
            }
            return false;
        }
        _filteredDepths[fi] = cv::Mat::zeros(geom[fi].H, geom[fi].W, CV_32F);
    }

    if (!_config.fuseOnlyFirstFrame && NF == 2 && _config.minNumPixels <= 1)
    {
        const bool fastOk = fuseTwoViewSingleObservationFast(frames,
                                                             geom,
                                                             colorProvider,
                                                             fusedPoints,
                                                             progressCb);
        if (!fastOk && isFusionCancelled(_config) && errorMsg)
        {
            *errorMsg = "用户取消深度图融合";
        }
        return fastOk;
    }

    // 6. 逐帧逐像素融合
    fusedPoints.clear();
    fusedPoints.reserve(100000);

    int totalProcessed = 0;

    const int fusionStartFrame = 0;
    const int fusionEndFrame = _config.fuseOnlyFirstFrame ? std::min(1, NF) : NF;
    const int fusionFrameCount = std::max(1, fusionEndFrame - fusionStartFrame);
    for (int fi = fusionStartFrame; fi < fusionEndFrame; ++fi)
    {
        if (isFusionCancelled(_config))
        {
            if (errorMsg)
            {
                *errorMsg = "用户取消深度图融合";
            }
            return false;
        }
        if (progressCb)
        {
            progressCb("融合 " + std::to_string(fi - fusionStartFrame + 1) + "/" +
                           std::to_string(fusionFrameCount),
                       static_cast<float>(fi - fusionStartFrame) / static_cast<float>(fusionFrameCount));
        }

        const int W = geom[fi].W;
        const int H = geom[fi].H;

        for (int r = 0; r < H; ++r)
        {
            if (isFusionCancelled(_config))
            {
                if (errorMsg)
                {
                    *errorMsg = "用户取消深度图融合";
                }
                return false;
            }
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
                if (!fusePixel(fi, r, c, frames, geom, overlapping, colorProvider, fusedMask, fp))
                {
                    continue;
                }

                fusedPoints.push_back(fp);
                ++totalProcessed;

                // 记录该像素通过一致性
                _filteredDepths[fi].at<float>(r, c) = d;
            }
        }

        int frameValid = cv::countNonZero(frames[fi].depthMap > 0);
        int frameFused = cv::countNonZero(_filteredDepths[fi] > 0);
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

} // namespace mvs
} // namespace xjw
