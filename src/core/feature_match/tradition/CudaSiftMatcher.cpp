#include "CudaSiftMatcher.h"

#include <torch/torch.h>

#include <algorithm>

namespace
{

torch::Tensor descriptorsToTensor(const cv::Mat &descriptors)
{
    cv::Mat continuous = descriptors.isContinuous() ? descriptors : descriptors.clone();
    return torch::from_blob(continuous.data,
                            {continuous.rows, continuous.cols},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone();
}

} // namespace

namespace xjw::feature_match::tradition
{

bool CudaSiftMatcher::isAvailable()
{
    return torch::cuda::is_available();
}

std::vector<std::vector<cv::DMatch>> CudaSiftMatcher::knnMatchL2(const cv::Mat &queryDescriptors,
                                                                 const cv::Mat &trainDescriptors,
                                                                 int k,
                                                                 int cudaDevice)
{
    if (queryDescriptors.empty() || trainDescriptors.empty())
    {
        return {};
    }

    const int neighbors = std::min(k, trainDescriptors.rows);
    if (neighbors <= 0)
    {
        return {};
    }

    torch::Device device(torch::kCUDA, std::max(0, cudaDevice));
    torch::NoGradGuard noGrad;
    torch::Tensor query = descriptorsToTensor(queryDescriptors).to(device);
    torch::Tensor train = descriptorsToTensor(trainDescriptors).to(device);
    const torch::Tensor trainSquared = train.square().sum(1).unsqueeze(0);

    constexpr int chunkRows = 1024;
    std::vector<std::vector<cv::DMatch>> matches(static_cast<std::size_t>(queryDescriptors.rows));
    for (int start = 0; start < queryDescriptors.rows; start += chunkRows)
    {
        const int count = std::min(chunkRows, queryDescriptors.rows - start);
        torch::Tensor queryChunk = query.narrow(0, start, count);
        torch::Tensor distances = queryChunk.square().sum(1).unsqueeze(1) + trainSquared -
                                  2.0f * torch::matmul(queryChunk, train.transpose(0, 1));
        distances = distances.clamp_min(0.0f);

        auto topk = torch::topk(distances, neighbors, 1, false, true);
        torch::Tensor values = std::get<0>(topk).sqrt().to(torch::kCPU).contiguous();
        torch::Tensor indices = std::get<1>(topk).to(torch::kCPU).contiguous();

        const float *valuePtr = values.data_ptr<float>();
        const int64_t *indexPtr = indices.data_ptr<int64_t>();
        for (int row = 0; row < count; ++row)
        {
            std::vector<cv::DMatch> rowMatches;
            rowMatches.reserve(static_cast<std::size_t>(neighbors));
            for (int neighbor = 0; neighbor < neighbors; ++neighbor)
            {
                const int offset = row * neighbors + neighbor;
                rowMatches.emplace_back(start + row,
                                        static_cast<int>(indexPtr[offset]),
                                        valuePtr[offset]);
            }
            matches[static_cast<std::size_t>(start + row)] = std::move(rowMatches);
        }
    }
    return matches;
}

} // namespace xjw::feature_match::tradition
