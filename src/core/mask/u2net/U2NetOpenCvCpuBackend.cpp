#include "U2NetInferenceBackend.h"

#include <opencv2/dnn.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

namespace xjw::mask
{
    namespace
    {

        class U2NetOpenCvCpuBackend final : public U2NetInferenceBackend
        {
        public:
            explicit U2NetOpenCvCpuBackend(const U2NetMaskGeneratorConfig& config)
            {
                _net = cv::dnn::readNetFromONNX(config.modelPath);
                if (_net.empty())
                {
                    throw std::runtime_error("Failed to load U2Net ONNX model: " + config.modelPath);
                }

                // OpenCV is deliberately a CPU-only fallback. GPU inference belongs to TensorRT.
                _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                _metadata.backend = U2NetBackendType::OpenCvCpu;
                _metadata.precision = U2NetInferencePrecision::Fp32;
                _metadata.deviceLabel = "OpenCV CPU";
                _metadata.fusedOutputName = "first graph output";
                _metadata.environmentSummary = "OpenCV DNN CPU";
            }

            cv::Mat forward(const cv::Mat& inputBlob) override
            {
                _net.setInput(inputBlob);
                const std::vector<std::string> outputNames = _net.getUnconnectedOutLayersNames();
                if (outputNames.empty())
                {
                    return _net.forward();
                }

                std::vector<cv::Mat> outputs;
                _net.forward(outputs, outputNames);
                if (outputs.empty())
                {
                    return {};
                }

                // U2Net exports the fused saliency map first, followed by six side outputs.
                return outputs.front();
            }

            const U2NetBackendMetadata& metadata() const override
            {
                return _metadata;
            }

        private:
            cv::dnn::Net _net;
            U2NetBackendMetadata _metadata;
        };

    } // namespace

    std::unique_ptr<U2NetInferenceBackend> createU2NetOpenCvCpuBackend(const U2NetMaskGeneratorConfig& config)
    {
        return std::make_unique<U2NetOpenCvCpuBackend>(config);
    }

} // namespace xjw::mask
