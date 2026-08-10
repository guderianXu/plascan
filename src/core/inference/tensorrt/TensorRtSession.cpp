#include "TensorRtSession.h"

#include "TensorRtSessionImpl.h"

namespace xjw::inference
{

    TensorRtSession::TensorRtSession(const std::string& enginePath, int cudaDevice)
        : _impl(std::make_unique<Impl>(enginePath, cudaDevice))
    {
    }

    TensorRtSession::TensorRtSession(const QString& enginePath, int cudaDevice)
        : TensorRtSession(enginePath.toStdString(), cudaDevice)
    {
    }

    TensorRtSession::~TensorRtSession() = default;

    int TensorRtSession::cudaDevice() const
    {
        return _impl->cudaDevice();
    }

    const std::vector<TensorRtTensorInfo>& TensorRtSession::tensors() const
    {
        return _impl->tensors();
    }

    TensorRtTensorInfo TensorRtSession::tensorInfo(const char* name) const
    {
        return _impl->tensorInfo(name);
    }

    bool TensorRtSession::hasTensor(const char* name) const
    {
        return _impl->hasTensor(name);
    }

    void TensorRtSession::validateTensor(const char* name, TensorRtTensorMode mode, TensorRtTensorDataType type) const
    {
        _impl->validate(name, mode, type);
    }

    void TensorRtSession::validateTensor(const char* name, nvinfer1::TensorIOMode mode, nvinfer1::DataType type) const
    {
        _impl->validateNative(name, mode, type);
    }

    nvinfer1::Dims TensorRtSession::tensorShape(const char* name) const
    {
        return _impl->shape(name);
    }

    void TensorRtSession::setInputShape(const char* name, const nvinfer1::Dims& shape)
    {
        _impl->setInputShape(name, shape);
    }

    void TensorRtSession::execute(const std::vector<TensorRtHostBinding>& bindings)
    {
        _impl->execute(bindings);
    }

} // namespace xjw::inference
