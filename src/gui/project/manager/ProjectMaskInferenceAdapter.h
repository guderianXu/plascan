#pragma once

#include <QJsonObject>
#include <QString>
#include <opencv2/core.hpp>

#include <functional>
#include <memory>
#include <string>

namespace xjw::mask
{
class BiRefNetMaskGenerator;
class U2NetMaskGenerator;
} // namespace xjw::mask

namespace xjw::gui::project
{

struct ProjectMaskInferenceResult
{
    cv::Mat mask;
    QString modelId;
    QString modelFileName;
    QString modelSha256;
    QString backend;
    QString device;
    QString precision;
    QString environment;
    QString fallbackReason;
    QString enginePath;
    int inputSize = 0;
    bool engineReused = false;
};

class ProjectMaskInferenceAdapter final
{
public:
    using StatusCallback = std::function<void(const std::string&)>;

    ~ProjectMaskInferenceAdapter();
    ProjectMaskInferenceAdapter(const ProjectMaskInferenceAdapter&) = delete;
    ProjectMaskInferenceAdapter& operator=(const ProjectMaskInferenceAdapter&) = delete;

    static std::unique_ptr<ProjectMaskInferenceAdapter>
    create(const QString& method,
           const QJsonObject& settings,
           StatusCallback statusCallback,
           QString* error);

    ProjectMaskInferenceResult generate(const cv::Mat& image);
    ProjectMaskInferenceResult metadata() const;

private:
    ProjectMaskInferenceAdapter() = default;

    QString _method;
    std::unique_ptr<xjw::mask::U2NetMaskGenerator> _u2net;
    std::unique_ptr<xjw::mask::BiRefNetMaskGenerator> _biRefNet;
    ProjectMaskInferenceResult _metadata;
};

} // namespace xjw::gui::project
