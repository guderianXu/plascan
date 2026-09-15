#include "MvsPipelineInternals.h"

namespace xjw::mvs
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    void MvsPipelineService::initializeWorkspaceManifest()
    {
        std::vector<QJsonObject> reusableArtifacts;
        int manifestSkipCount = 0;
        QString manifestPathForLog;
        QString depthConfigHashForLog;

        {
            std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
            _workspaceManifestPath = manifestPathForOutput(_config, _outputDir);
            _depthConfigHash = makeMvsDepthInputHash(_config, _views, _sparse);
            {
                std::lock_guard<std::mutex> prepared_lock(_preparedRasterArtifactsMutex);
                _preparedRasterArtifacts.assign(_views.size(), MvsPreparedRasterArtifact{});
            }
            manifestPathForLog = _workspaceManifestPath;
            depthConfigHashForLog = _depthConfigHash;
            _workspaceManifest.clear();
            _workspaceManifest.setConfigHash(_depthConfigHash);

            if (_workspaceManifestPath.isEmpty())
            {
                return;
            }

            QString error;
            if (QFile::exists(_workspaceManifestPath))
            {
                MvsWorkspaceManifest loaded;
                if (loaded.load(_workspaceManifestPath, &error))
                {
                    if (loaded.configHash() == _depthConfigHash)
                    {
                        _workspaceManifest = loaded;
                    }
                    else
                    {
                        LOG_INFO(QStringLiteral("[MVS] 深度图 manifest 参数已变化，旧记录不复用: %1")
                                     .arg(QDir::toNativeSeparators(_workspaceManifestPath)));
                    }
                }
                else
                {
                    LOG_WARN(QStringLiteral("[MVS] 读取深度图 manifest 失败，将重建: %1 error=%2")
                                 .arg(QDir::toNativeSeparators(_workspaceManifestPath), error));
                }
            }

            _workspaceManifest.setConfigHash(_depthConfigHash);
            if (_skipFrameMask.size() != _views.size())
            {
                _skipFrameMask.assign(_views.size(), 0);
            }
            for (int i = 0; i < static_cast<int>(_views.size()); ++i)
            {
                if (_workspaceManifest.hasReusableCompletedFrame(i, _depthConfigHash))
                {
                    _skipFrameMask[static_cast<size_t>(i)] = 1;
                    ++manifestSkipCount;

                    for (const MvsDepthFrameRecord& record : _workspaceManifest.frames())
                    {
                        if (record.refIndex == i)
                        {
                            QJsonObject artifact = record.toJson();
                            artifact[QStringLiteral("result_type")] = QStringLiteral("mvs_depth");
                            artifact[QStringLiteral("manifest_path")] = _workspaceManifestPath;
                            reusableArtifacts.push_back(artifact);
                            break;
                        }
                    }
                }
            }

            LOG_INFO(QStringLiteral("[MVS] 深度图 manifest: path=%1 config=%2 reusable=%3")
                         .arg(QDir::toNativeSeparators(_workspaceManifestPath))
                         .arg(_depthConfigHash.left(12))
                         .arg(manifestSkipCount));

            if (manifestSkipCount > 0 && _config.runFusion)
            {
                LOG_WARN(QStringLiteral("[MVS] 检测到 %1 个可复用深度帧，本次自动切换为深度图续跑模式；"
                                        "完成后请使用已保存深度图运行融合，避免只融合本次新计算的子集。")
                             .arg(manifestSkipCount));
                _config.runFusion = false;
            }
        }

        for (const QJsonObject& artifact : reusableArtifacts)
        {
            depthMapArtifactSaved(artifact);
        }
        if (!reusableArtifacts.empty())
        {
            LOG_INFO(QStringLiteral("[MVS] 已从 manifest 回灌 %1 个深度图记录到项目元数据: %2 config=%3")
                         .arg(static_cast<int>(reusableArtifacts.size()))
                         .arg(QDir::toNativeSeparators(manifestPathForLog))
                         .arg(depthConfigHashForLog.left(12)));
        }
    }

    bool MvsPipelineService::persistWorkspaceManifest(QString* errorMsg)
    {
        if (_workspaceManifestPath.isEmpty())
        {
            return true;
        }
        return _workspaceManifest.saveAtomic(_workspaceManifestPath, errorMsg);
    }

    void MvsPipelineService::markManifestFrameRunning(int frameIndex)
    {
        if (frameIndex < 0 || frameIndex >= static_cast<int>(_views.size()))
        {
            return;
        }

        std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
        if (_workspaceManifestPath.isEmpty())
        {
            return;
        }

        _workspaceManifest.markRunning(
            frameIndex, QString::fromStdString(_views[frameIndex].imagePath), _depthConfigHash);
        QString error;
        if (!persistWorkspaceManifest(&error))
        {
            LOG_WARN(QStringLiteral("[MVS] 写入运行中 manifest 失败: %1").arg(error));
        }
    }

    void MvsPipelineService::markManifestFrameFailed(int frameIndex, const QString& error)
    {
        if (frameIndex < 0 || frameIndex >= static_cast<int>(_views.size()))
        {
            return;
        }

        std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
        if (_workspaceManifestPath.isEmpty())
        {
            return;
        }

        _workspaceManifest.markFailed(frameIndex, error);
        QString saveError;
        if (!persistWorkspaceManifest(&saveError))
        {
            LOG_WARN(QStringLiteral("[MVS] 写入失败 manifest 失败: %1").arg(saveError));
        }
    }

    bool MvsPipelineService::publishTerminalDepthCheckpoints()
    {
        struct TerminalPublication
        {
            MvsDepthFrameRecord frame;
            QJsonObject artifact;
        };
        QVector<TerminalPublication> publications;
        QString publication_error;
        {
            std::lock_guard<std::mutex> lock(_workspaceManifestMutex);
            if (_workspaceManifestPath.isEmpty())
            {
                return true;
            }

            const QVector<MvsDepthFrameRecord> records = _workspaceManifest.frames();
            for (const MvsDepthFrameRecord& checkpoint : records)
            {
                if (checkpoint.status != QStringLiteral("running") || checkpoint.refIndex < 0 ||
                    checkpoint.refIndex >= static_cast<int>(_depthFrames.size()) ||
                    !_depthFrames[static_cast<std::size_t>(checkpoint.refIndex)].success)
                {
                    continue;
                }

                const DepthFrameResult& frame = _depthFrames[static_cast<std::size_t>(checkpoint.refIndex)];
                MvsDepthFrameRecord terminal = checkpoint;
                terminal.consistencyPublicationExpected =
                    detail::expectsConsistencyPublication(frame, static_cast<int>(_views.size()));
                terminal.geometricGuidancePassExpected = frame.geometricGuidancePassExpected;
                terminal.geometricGuidancePassApplied = frame.geometricGuidancePassApplied;
                if (terminal.geometricGuidancePassApplied && !terminal.geometricGuidancePassExpected)
                {
                    publication_error =
                        QStringLiteral("帧 %1 的几何引导状态无效：已执行但未标记为预期").arg(checkpoint.refIndex);
                    _workspaceManifest.markFailed(checkpoint.refIndex, publication_error);
                    break;
                }
                _workspaceManifest.markCompleted(terminal);

                // A memory-skipped optional guidance pass makes the checkpoint
                // intentionally non-reusable, but the diagnostic frame is still
                // structurally publishable. Validate every other durable contract
                // through an isolated manifest copy without weakening cache reuse.
                MvsWorkspaceManifest publication_validation = _workspaceManifest;
                if (terminal.geometricGuidancePassExpected && !terminal.geometricGuidancePassApplied)
                {
                    MvsDepthFrameRecord validation_record = terminal;
                    validation_record.status = QStringLiteral("completed");
                    validation_record.geometricGuidancePassExpected = false;
                    validation_record.geometricGuidancePassApplied = false;
                    publication_validation.upsertFrame(validation_record);
                }
                if (!publication_validation.hasReusableCompletedFrame(checkpoint.refIndex, _depthConfigHash))
                {
                    publication_error = QStringLiteral("帧 %1 的初始 MVS checkpoint 缺少可验证的终态工件，已拒绝发布")
                                            .arg(checkpoint.refIndex);
                    _workspaceManifest.markFailed(checkpoint.refIndex, publication_error);
                    break;
                }

                for (const MvsDepthFrameRecord& completed : _workspaceManifest.frames())
                {
                    if (completed.refIndex != checkpoint.refIndex)
                    {
                        continue;
                    }
                    QJsonObject artifact = completed.toJson();
                    artifact.insert(QStringLiteral("result_type"), QStringLiteral("mvs_depth"));
                    artifact.insert(QStringLiteral("manifest_path"), _workspaceManifestPath);
                    artifact.insert(QStringLiteral("stage"), QStringLiteral("无多视一致性终态"));
                    artifact.insert(
                        QStringLiteral("cache_reusable"),
                        _workspaceManifest.hasReusableCompletedFrame(checkpoint.refIndex, _depthConfigHash));
                    publications.push_back({completed, std::move(artifact)});
                    break;
                }
            }

            QString manifest_error;
            if (!persistWorkspaceManifest(&manifest_error))
            {
                publication_error = QStringLiteral("写入终态 MVS checkpoint manifest 失败: %1").arg(manifest_error);
            }
        }

        if (!publication_error.isEmpty())
        {
            LOG_ERROR(QStringLiteral("[MVS] %1").arg(publication_error));
            errorOccurred(publication_error);
            return false;
        }
        for (const TerminalPublication& publication : publications)
        {
            depthMapSaved(publication.frame.depthPng,
                          publication.frame.gridWidth,
                          publication.frame.gridHeight,
                          publication.frame.refImage);
            depthMapArtifactSaved(publication.artifact);
        }
        return true;
    }
} // namespace xjw::mvs
