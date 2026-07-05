/**
 * @file  LightGlueMatcher.cpp
 * @brief LightGlue 匹配器实现
 *
 * 支持两种工作模式：
 *  1. match(FeatureData, FeatureData) —— 使用已有特征数据（如从 .sp 文件读入的 FeatureOutput）。
 *  2. matchImages(cv::Mat, cv::Mat)   —— 通过内置 SP 模型端到端匹配。
 */

#include "LightGlueMatcher.h"

#include "../../feature_extractors/FeatureData.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace xjw::feature_match
{

// ---------------------------------------------------------------------------
// 构造函数
// ---------------------------------------------------------------------------
LightGlueMatcher::LightGlueMatcher(const LightGlueConfig &config)
    : _torchDevice(config.useCuda && torch::cuda::is_available()
                       ? torch::Device(torch::kCUDA, std::max(0, config.cudaDevice))
                       : torch::Device(torch::kCPU)),
      _config(config)
{
    // 加载 LightGlue 匹配器模型
    if (!config.matcherModelPath.empty())
    {
        try
        {
            _lgModel = torch::jit::load(config.matcherModelPath, _torchDevice);
            _lgModel.eval();
            _matcherLoaded = true;
        }
        catch (const c10::Error &e)
        {
            throw std::runtime_error(
                std::string("[LightGlueMatcher] LightGlue 模型加载失败: ") + e.what());
        }
    }

    // 可选：加载 SP 提取器模型
    if (!config.spModelPath.empty())
    {
        try
        {
            _spModel = torch::jit::load(config.spModelPath, _torchDevice);
            _spModel.eval();
            _spLoaded = true;
        }
        catch (const c10::Error &e)
        {
            // SP 模型可选，不中断构造
        }
    }
}

// ---------------------------------------------------------------------------
// 辅助：BGR Mat → float32 Tensor [1,3,H,W] on device
// ---------------------------------------------------------------------------
torch::Tensor LightGlueMatcher::_matToTensor(const cv::Mat &bgr) const
{
    cv::Mat rgb, rgbF;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgbF, CV_32F, 1.0 / 255.0);

    auto t = torch::from_blob(
        rgbF.data,
        {rgbF.rows, rgbF.cols, 3},
        torch::kFloat32).clone();
    return t.permute({2, 0, 1}).unsqueeze(0).to(_torchDevice);
}

// ---------------------------------------------------------------------------
// 辅助：从 log assignment scores 提取 mutual-NN 匹配索引及置信度
// 返回 vector<{idx0, score}>，其中 idx1 = m0[idx0]
// ---------------------------------------------------------------------------
std::vector<std::pair<int64_t, float>>
LightGlueMatcher::_filterScores(const torch::Tensor &scores, int64_t expectedN0, int64_t expectedN1) const
{
    // scores: [1, M(+1), N(+1)]
    // 若模型输出含 dustbin 行/列（尺寸比关键点数多 1），则裁掉最后一行/列
    const int64_t dim1 = scores.size(1);
    const int64_t dim2 = scores.size(2);
    const int64_t endRow = (dim1 == expectedN0 + 1) ? dim1 - 1 : dim1;
    const int64_t endCol = (dim2 == expectedN1 + 1) ? dim2 - 1 : dim2;

    auto s = scores[0].slice(0, 0, endRow).slice(1, 0, endCol); // [M, N]

    auto max0 = s.max(1);
    auto max1 = s.max(0);

    auto m0 = std::get<1>(max0).cpu();       // [M] 每个 kpt0 最配的 kpt1 索引
    auto m1 = std::get<1>(max1).cpu();       // [N] 每个 kpt1 最配的 kpt0 索引
    auto v0 = std::get<0>(max0).exp().cpu(); // [M] 置信度

    const int64_t M  = m0.size(0);
    const int64_t N1 = m1.size(0);
    auto m0Acc = m0.accessor<int64_t, 1>();
    auto m1Acc = m1.accessor<int64_t, 1>();
    auto v0Acc = v0.accessor<float, 1>();

    std::vector<std::pair<int64_t, float>> result;
    result.reserve(static_cast<size_t>(M / 4));
    for (int64_t i = 0; i < M; ++i)
    {
        const int64_t j = m0Acc[i];
        if (j < 0 || j >= N1) continue;
        if (m1Acc[j] == i && v0Acc[i] > _config.scoreThreshold)
        {
            result.emplace_back(i, v0Acc[i]);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// IFeatureMatcher::match —— 从 FeatureData 输入匹配
//
// 设计思路：
//   FeatureData 在项目中存储的是像素坐标（原图尺度）和 [N,D] CV_32F 描述子。
//   LightGlue 模型需要：
//     kpts_t  [1,N,2] 坐标（fixed-size 空间）；SIFT 额外传 [scale, orientation]
//     descs_t [1,N,D] 描述子（DISK/ALIKED 为 128 维，SuperPoint 为 256 维）
//     fixed_wh [1,2]   fixed 宽高
//   将 imageWidth/imageHeight 设为 fixed_wh，则 kpts_t = 像素坐标，无需坐标变换。
// ---------------------------------------------------------------------------
xjw::feature_match::MatchResult
LightGlueMatcher::match(const xjw::feature_extractors::FeatureData &feat0,
                        const xjw::feature_extractors::FeatureData &feat1)
{
    if (!_matcherLoaded)
    {
        throw std::runtime_error("[LightGlueMatcher] 匹配器模型未加载");
    }
    if (feat0.empty() || feat1.empty())
    {
        return {};
    }

    const int N0 = feat0.size();
    const int N1 = feat1.size();

    // 确定 fixed_wh：使用 FeatureData 中存储的图像宽高，没有则尝试从关键点最大坐标推断
    auto inferSize = [](const xjw::feature_extractors::FeatureData &fd) -> std::pair<float, float>
    {
        if (fd.imageWidth > 0 && fd.imageHeight > 0)
        {
            return {static_cast<float>(fd.imageWidth), static_cast<float>(fd.imageHeight)};
        }
        float maxX = 1.0f, maxY = 1.0f;
        for (const auto &kp : fd.keypoints)
        {
            maxX = std::max(maxX, kp.pt.x);
            maxY = std::max(maxY, kp.pt.y);
        }
        // 添加 10% 边距，避免坐标归一化时超出范围
        return {maxX * 1.1f + 1.0f, maxY * 1.1f + 1.0f};
    };

    auto [fw0, fh0] = inferSize(feat0);
    auto [fw1, fh1] = inferSize(feat1);

    // 构造 kpts_t [1,N,2]；SIFT LightGlue 需要 OpenCV SIFT 的尺度和方向参与位置编码。
    auto buildKpts = [&](const xjw::feature_extractors::FeatureData &fd) -> torch::Tensor
    {
        const int N = fd.size();
        const bool useSiftGeometry = fd.sourceAlgorithm == "sift";
        const int kptDim = useSiftGeometry ? 4 : 2;
        torch::Tensor t = torch::zeros({1, N, kptDim}, torch::kFloat32);
        auto acc = t.accessor<float, 3>();
        for (int i = 0; i < N; ++i)
        {
            const cv::KeyPoint &keypoint = fd.keypoints[i];
            acc[0][i][0] = keypoint.pt.x;
            acc[0][i][1] = keypoint.pt.y;
            if (useSiftGeometry)
            {
                acc[0][i][2] = std::max(1.0f, keypoint.size);
                acc[0][i][3] = (std::isfinite(keypoint.angle) && keypoint.angle >= 0.0f)
                    ? keypoint.angle * static_cast<float>(CV_PI / 180.0)
                    : 0.0f;
            }
        }
        return t.to(_torchDevice);
    };

    // 构造 descs_t [1,N,D]
    auto buildDescs = [&](const xjw::feature_extractors::FeatureData &fd) -> torch::Tensor
    {
        torch::Tensor nd = fd.toTensorND(); // [N, D]
        const int64_t D = nd.size(1);
        if (D != 128 && D != 256)
        {
            throw std::runtime_error(
                "[LightGlueMatcher] 描述子维度不支持: " + std::to_string(D) +
                " (期望 128 或 256)");
        }
        return nd.unsqueeze(0).to(_torchDevice); // [1, N, D]
    };

    // 构造 fixed_wh [1,2]
    auto buildWH = [&](float w, float h) -> torch::Tensor
    {
        return torch::tensor({{w, h}},
                             torch::TensorOptions().dtype(torch::kFloat32).device(_torchDevice));
    };

    auto kpts0   = buildKpts(feat0);
    auto kpts1   = buildKpts(feat1);
    auto descs0  = buildDescs(feat0);
    auto descs1  = buildDescs(feat1);
    auto fixedWH0 = buildWH(fw0, fh0);
    auto fixedWH1 = buildWH(fw1, fh1);

    torch::NoGradGuard noGrad;
    torch::Tensor scoresTensor = _lgModel.forward(
        {kpts0, descs0, fixedWH0, kpts1, descs1, fixedWH1}).toTensor();

    if (!scoresTensor.defined() || scoresTensor.numel() == 0)
    {
        throw std::runtime_error("[LightGlueMatcher] 模型推理失败: 返回空张量");
    }
    if (scoresTensor.dim() != 3)
    {
        throw std::runtime_error(
            "[LightGlueMatcher] 模型输出维度错误: 期望 3 维, 实际 " +
            std::to_string(scoresTensor.dim()) + " 维");
    }

    auto validPairs = _filterScores(scoresTensor, N0, N1);

    // 获取 m0[索引]: 每个 kpt0 对应的最佳 kpt1 索引（与 _filterScores 使用相同裁剪逻辑）
    const int64_t endRow = (scoresTensor.size(1) == N0 + 1) ? scoresTensor.size(1) - 1 : scoresTensor.size(1);
    const int64_t endCol = (scoresTensor.size(2) == N1 + 1) ? scoresTensor.size(2) - 1 : scoresTensor.size(2);
    auto sSlice = scoresTensor[0].slice(0, 0, endRow).slice(1, 0, endCol); // [M, N]
    auto m0Tensor = std::get<1>(sSlice.max(1)).cpu();                      // [M]
    auto m0Acc = m0Tensor.accessor<int64_t, 1>();

    // 构造 MatchResult
    xjw::feature_match::MatchResult result;
    result.sourceAlgorithm = "lightglue";
    result.matches0.assign(N0, -1);
    result.matches1.assign(N1, -1);
    result.matchingScores0.assign(N0, 0.0f);
    result.matchingScores1.assign(N1, 0.0f);

    for (auto &[idx0, score] : validPairs)
    {
        const int64_t idx1 = m0Acc[idx0];
        if (idx1 < 0 || idx1 >= N1) continue;

        result.matches0[static_cast<int>(idx0)] = static_cast<int>(idx1);
        result.matches1[static_cast<int>(idx1)] = static_cast<int>(idx0);
        result.matchingScores0[static_cast<int>(idx0)] = score;
        result.matchingScores1[static_cast<int>(idx1)] = score;

        cv::DMatch dm;
        dm.queryIdx = static_cast<int>(idx0);
        dm.trainIdx = static_cast<int>(idx1);
        dm.distance = 1.0f - score;
        result.cvMatches.push_back(dm);
    }
    result.numMatches = static_cast<int>(result.cvMatches.size());
    return result;
}

// ---------------------------------------------------------------------------
// extractNative ：使用内置 SP 模型提取单幅图像特征
// ---------------------------------------------------------------------------
xjw::feature_extractors::FeatureData
LightGlueMatcher::extractNative(const cv::Mat &img) const
{
    if (!_spLoaded)
    {
        throw std::runtime_error("[LightGlueMatcher] SP 模型未加载，请设置 LightGlueConfig::spModelPath");
    }
    if (img.empty())
    {
        throw std::runtime_error("[LightGlueMatcher] extractNative: 输入图像为空");
    }

    const int origH = img.rows, origW = img.cols;
    auto origWh = torch::tensor(
        {{static_cast<float>(origW), static_cast<float>(origH)}},
        torch::TensorOptions().dtype(torch::kFloat32).device(_torchDevice));

    auto imgTensor = _matToTensor(img);

    torch::NoGradGuard noGrad;
    auto spOut = _spModel.forward({imgTensor, origWh}).toTuple();

    auto kptsT   = spOut->elements()[0].toTensor().cpu(); // [1,N,2] fixed-size
    auto descsT  = spOut->elements()[1].toTensor().cpu(); // [1,N,256]
    auto fixedWH = spOut->elements()[2].toTensor().cpu(); // [1,2]

    const float fw = fixedWH[0][0].item<float>();
    const float fh = fixedWH[0][1].item<float>();

    const int64_t N = kptsT.size(1);
    auto kpts0 = kptsT[0].contiguous();
    auto kAcc = kpts0.accessor<float, 2>();

    xjw::feature_extractors::FeatureData fd;
    fd.sourceAlgorithm = "superpoint";
    fd.imageWidth  = origW;
    fd.imageHeight = origH;
    fd.keypoints.reserve(static_cast<size_t>(N));

    for (int64_t i = 0; i < N; ++i)
    {
        // 坐标从 fixed-size 还原到原图尺度
        const float x = (kAcc[i][0] + 0.5f) * (static_cast<float>(origW) / fw) - 0.5f;
        const float y = (kAcc[i][1] + 0.5f) * (static_cast<float>(origH) / fh) - 0.5f;
        fd.keypoints.emplace_back(cv::KeyPoint(x, y, 1.0f));
        fd.scores.push_back(1.0f); // SP 提取器未单独输出分数
    }

    // 描述子从 [1,N,256] 转为 cv::Mat [N,256]
    auto descCont = descsT[0].contiguous(); // [N, 256]
    const int D = static_cast<int>(descCont.size(1));
    fd.descriptors = cv::Mat(static_cast<int>(N), D, CV_32F);
    std::memcpy(fd.descriptors.ptr<float>(0),
                descCont.data_ptr<float>(),
                static_cast<size_t>(N) * D * sizeof(float));

    return fd;
}

// ---------------------------------------------------------------------------
// matchImages ：使用内置 SP 模型端到端匹配
// ---------------------------------------------------------------------------
xjw::feature_match::MatchResult
LightGlueMatcher::matchImages(const cv::Mat &img0, const cv::Mat &img1)
{
    auto fd0 = extractNative(img0);
    auto fd1 = extractNative(img1);
    return match(fd0, fd1);
}

} // namespace xjw::feature_match
