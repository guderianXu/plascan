#include "FeatureData.h"

#include "FeatureOutput.h" // 提供 FeatureOutput 定义
#include "string_utils/StringTransform.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xjw::feature_extractors
{

// ---------------------------------------------------------------------------
// 格式转换
// ---------------------------------------------------------------------------

torch::Tensor FeatureData::toTensorND() const
{
    if (descriptors.empty())
    {
        return torch::Tensor();
    }

    CV_Assert(descriptors.type() == CV_32F);
    const int N = descriptors.rows;
    const int D = descriptors.cols;

    // 直接从 cv::Mat 内存构造 Tensor，拷贝一份以保证生命周期安全
    return torch::from_blob(
               const_cast<float *>(descriptors.ptr<float>(0)),
               {N, D},
               torch::kFloat32)
        .clone();
}

torch::Tensor FeatureData::toTensorSuperGlue() const
{
    torch::Tensor nd = toTensorND(); // [N, D]
    if (!nd.defined() || nd.numel() == 0)
    {
        return torch::Tensor();
    }
    // [N, D] -> [D, N] -> [1, D, N]
    return nd.t().unsqueeze(0).contiguous();
}

// ---------------------------------------------------------------------------
// 工厂方法
// ---------------------------------------------------------------------------

FeatureData FeatureData::fromFeatureOutput(const FeatureOutput &out,
                                              const std::string &algoName)
{
    FeatureData fd;
    fd.keypoints = out.keypoints;
    fd.scores = out.scores;
    fd.sourceAlgorithm = algoName;
    fd.imageWidth = out.imageWidth;
    fd.imageHeight = out.imageHeight;

    // FeatureOutput::descriptors 为 [N, D] float32 CPU Tensor
    if (out.descriptors.defined() && out.descriptors.numel() > 0)
    {
        const auto t = out.descriptors.contiguous().cpu().to(torch::kFloat32);
        const int N = static_cast<int>(t.size(0));
        const int D = static_cast<int>(t.size(1));
        fd.descriptors = cv::Mat(N, D, CV_32F);
        std::memcpy(fd.descriptors.ptr<float>(0),
                    t.data_ptr<float>(),
                    static_cast<size_t>(N) * D * sizeof(float));
    }

    return fd;
}

FeatureData FeatureData::fromOpenCV(const std::vector<cv::KeyPoint> &keypoints,
                                    const cv::Mat &descriptors,
                                    const std::string &algoName)
{
    FeatureData fd;
    fd.keypoints = keypoints;
    fd.sourceAlgorithm = algoName;

    // 从 cv::KeyPoint::response 提取得分
    fd.scores.reserve(keypoints.size());
    for (const auto &kp : keypoints)
    {
        fd.scores.push_back(kp.response);
    }

    if (!descriptors.empty())
    {
        if (descriptors.type() == CV_32F)
        {
            // 直接克隆
            fd.descriptors = descriptors.clone();
        }
        else if (descriptors.type() == CV_8U)
        {
            // Binary 描述子：逐位展开为 float（0.0 / 1.0）
            const int N = descriptors.rows;
            const int bytesPerDesc = descriptors.cols;
            const int D = bytesPerDesc * 8;
            fd.descriptors = cv::Mat(N, D, CV_32F, cv::Scalar(0.0f));
            for (int r = 0; r < N; ++r)
            {
                const uchar *src = descriptors.ptr<uchar>(r);
                float *dst = fd.descriptors.ptr<float>(r);
                for (int b = 0; b < bytesPerDesc; ++b)
                {
                    const uchar byte = src[b];
                    for (int bit = 0; bit < 8; ++bit)
                    {
                        dst[b * 8 + bit] = ((byte >> bit) & 1) ? 1.0f : 0.0f;
                    }
                }
            }
        }
        else
        {
            // 其他类型（如 CV_16U）统一转为 float
            descriptors.convertTo(fd.descriptors, CV_32F);
        }
    }

    return fd;
}

cv::Mat FeatureData::toCvDescriptors(const std::string &matchAlgorithm) const
{
    if (descriptors.empty())
    {
        return cv::Mat();
    }

    CV_Assert(descriptors.type() == CV_32F);

    // 判断是否需要 bit-packed CV_8U（ORB Hamming 匹配）
    const std::string algo = xjw::common::string_utils::asciiLowerCopy(matchAlgorithm);

    const bool needBinary = (algo == "orb" || algo == "orb_bf_hamming" ||
                             algo == "hamming" || algo == "bf_hamming" || algo == "bf-hamming");
    if (needBinary)
    {
        const int N = descriptors.rows;
        const int D = descriptors.cols;
        const int bytesPerRow = (D + 7) / 8;
        cv::Mat packed(N, bytesPerRow, CV_8U, cv::Scalar(0));
        for (int r = 0; r < N; ++r)
        {
            const float *src = descriptors.ptr<float>(r);
            uchar *dst = packed.ptr<uchar>(r);
            for (int b = 0; b < D; ++b)
            {
                if (src[b] > 0.0f)
                {
                    dst[b / 8] = static_cast<uchar>(dst[b / 8] | (1u << (b % 8)));
                }
            }
        }
        return packed;
    }

    // SIFT / FLANN / L2: 直接返回 CV_32F 克隆
    return descriptors.clone();
}

torch::Tensor FeatureData::cvDescriptorsToTensor(const cv::Mat &descriptors,
                                                  int targetDim,
                                                  const std::string &sourceFeatureAlgorithm)
{
    const int safeTargetDim = std::max(1, targetDim);
    if (descriptors.empty() || descriptors.rows <= 0)
    {
        return torch::empty({0, safeTargetDim}, torch::kFloat32);
    }

    const std::string algo = xjw::common::string_utils::asciiLowerCopy(sourceFeatureAlgorithm);

    torch::Tensor output = torch::zeros({static_cast<int64_t>(descriptors.rows), safeTargetDim},
                                        torch::kFloat32);
    auto acc = output.accessor<float, 2>();

    if (descriptors.type() == CV_8U && algo == "orb")
    {
        const int bitDim = descriptors.cols * 8;
        const int copyDim = std::min(bitDim, safeTargetDim);
        for (int row = 0; row < descriptors.rows; ++row)
        {
            const uchar *src = descriptors.ptr<uchar>(row);
            for (int col = 0; col < copyDim; ++col)
            {
                acc[row][col] = ((src[col / 8] >> (col % 8)) & 0x01u) ? 1.0f : -1.0f;
            }
        }
    }
    else
    {
        cv::Mat floatDesc;
        if (descriptors.type() == CV_32F)
            floatDesc = descriptors;
        else
            descriptors.convertTo(floatDesc, CV_32F);

        const int copyDim = std::min(floatDesc.cols, safeTargetDim);
        for (int row = 0; row < floatDesc.rows; ++row)
        {
            const float *src = floatDesc.ptr<float>(row);
            if (algo == "sift")
            {
                float l1Norm = 0.0f;
                for (int col = 0; col < copyDim; ++col)
                {
                    l1Norm += std::max(0.0f, src[col]);
                }
                const float invL1 = l1Norm > 1e-6f ? 1.0f / l1Norm : 0.0f;
                for (int col = 0; col < copyDim; ++col)
                {
                    const float l1Value = std::max(0.0f, src[col]) * invL1;
                    acc[row][col] = std::sqrt(std::max(l1Value, 1e-6f));
                }
            }
            else
            {
                for (int col = 0; col < copyDim; ++col)
                {
                    acc[row][col] = src[col];
                }
            }
        }
    }

    // L2 归一化各行
    for (int64_t row = 0; row < output.size(0); ++row)
    {
        float norm2 = 0.0f;
        for (int64_t col = 0; col < output.size(1); ++col)
        {
            const float v = acc[row][col];
            norm2 += v * v;
        }
        if (norm2 > 1e-12f)
        {
            const float invNorm = 1.0f / std::sqrt(norm2);
            for (int64_t col = 0; col < output.size(1); ++col)
                acc[row][col] *= invNorm;
        }
    }

    return output;
}

FeatureData FeatureData::fromTensorOutput(const std::vector<cv::KeyPoint> &kpts,
                                          const std::vector<float> &scores,
                                          const torch::Tensor &desc,
                                          const std::string &algoName)
{
    FeatureData fd;
    fd.keypoints       = kpts;
    fd.scores          = scores;
    fd.sourceAlgorithm = algoName;

    if (desc.defined() && desc.numel() > 0)
    {
        const auto t = desc.contiguous().cpu().to(torch::kFloat32);
        const int N = static_cast<int>(t.size(0));
        const int D = static_cast<int>(t.size(1));
        fd.descriptors = cv::Mat(N, D, CV_32F);
        std::memcpy(fd.descriptors.ptr<float>(0),
                    t.data_ptr<float>(),
                    static_cast<size_t>(N) * D * sizeof(float));
    }

    return fd;
}

} // namespace xjw::feature_extractors
