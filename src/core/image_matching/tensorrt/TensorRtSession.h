#pragma once

/**
 * @file TensorRtSession.h
 * @brief 影像匹配算法共用的固定形状 TensorRT 执行会话。
 */

#include <NvInferRuntime.h>

#include <memory>
#include <string>
#include <vector>

namespace xjw::image_matching
{

struct TensorRtHostBinding
{
    const char *name = nullptr;
    void *data = nullptr;
    std::size_t bytes = 0;
    bool input = true;
};

class TensorRtSession final
{
public:
    TensorRtSession(const std::string &enginePath, int cudaDevice);
    ~TensorRtSession();

    TensorRtSession(const TensorRtSession &) = delete;
    TensorRtSession &operator=(const TensorRtSession &) = delete;

    void validateTensor(const char *name,
                        nvinfer1::TensorIOMode mode,
                        nvinfer1::DataType type) const;
    nvinfer1::Dims tensorShape(const char *name) const;
    void setInputShape(const char *name, const nvinfer1::Dims &shape);
    void execute(const std::vector<TensorRtHostBinding> &bindings);

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace xjw::image_matching
