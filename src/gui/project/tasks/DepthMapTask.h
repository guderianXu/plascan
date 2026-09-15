#pragma once
// GUI-owned asynchronous lifecycle for the synchronous MVS service.

#include "MvsPipelineService.h"

#include <QObject>
#include <QFuture>
#include <QMetaType>

namespace xjw::gui::tasks
{
    using mvs::CameraView;
    using mvs::DensePoint;
    using mvs::DepthFrameResult;
    using mvs::DepthGenConfig;
    using mvs::MvsPipelineService;
    using mvs::SparseCloud;

    class DepthMapTask : public QObject
    {
        Q_OBJECT

    public:
        explicit DepthMapTask(QObject* parent = nullptr);

        ~DepthMapTask() override;

        /// 设置输入数据
        void setViews(const std::vector<CameraView>& views);
        void setSparseCloud(const SparseCloud& sparse);
        void setConfig(const DepthGenConfig& config);
        void setSkippedFrameIndices(const std::vector<int>& indices);

        /// 异步启动
        Q_INVOKABLE void start();

        /// 设置输出目录（深度图 PNG 保存位置）
        void setOutputDir(const std::string& dir)
        {
            if (_backgroundFuture.isRunning())
            {
                emit errorOccurred(QStringLiteral("深度图生成任务正在运行，不能修改输出目录"));
                return;
            }
            _service->setOutputDir(dir);
        }

        /// 请求取消
        void requestCancel()
        {
            _service->requestCancel();
        }

    signals:
        /// 每估计完一帧就发出
        void depthMapReady(DepthFrameResult result);
        /// 每帧深度图保存为 PNG 后发出（path, width, height, refImagePath）
        void depthMapSaved(QString pngPath, int width, int height, QString refImagePath);
        /// 每帧深度图全部产物保存后发出结构化元数据，供项目树增量刷新
        void depthMapArtifactSaved(QJsonObject artifact);
        /// 点云生成完毕
        void pointCloudReady(std::vector<DensePoint> cloud);
        /// 进度更新
        void progressChanged(QString stage, float ratio);
        /// 出错
        void errorOccurred(QString msg);
        /// 整个流程完成
        void finished(bool success);

    private:
        std::unique_ptr<MvsPipelineService> _service;
        QFuture<void> _backgroundFuture;
    };
} // namespace xjw::gui::tasks

Q_DECLARE_METATYPE(xjw::mvs::DepthFrameResult)
Q_DECLARE_METATYPE(QSharedPointer<cv::Mat>)
Q_DECLARE_METATYPE(std::vector<xjw::mvs::DensePoint>)
