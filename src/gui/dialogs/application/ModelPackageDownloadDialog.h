#pragma once

#include "model/ModelAssetCatalog.h"

#include <QCryptographicHash>
#include <QDialog>
#include <QFile>

#include <memory>

class QLabel;
class QNetworkAccessManager;
class QNetworkReply;
class QProgressBar;
class QPushButton;

/**
 * @brief 从固定 GitHub Release 下载一组经过校验的模型文件。
 *
 * 下载始终写入同目录 `.part` 临时文件，大小和 SHA-256 均通过后才替换目标文件。
 * 对话框使用 Qt Network 异步传输，因此大型模型下载期间不会阻塞 GUI 事件循环。
 */
class ModelPackageDownloadDialog final : public QDialog
{
public:
    ModelPackageDownloadDialog(
        xjw::common::model::ModelAssetPackage package,
        QString targetDirectory,
        QWidget *parent = nullptr);

    static bool downloadPackage(
        const xjw::common::model::ModelAssetPackage &package,
        const QString &targetDirectory,
        QWidget *parent,
        QString *errorMessage = nullptr);

    QString errorMessage() const;

protected:
    void reject() override;

private:
    void start();
    void startNextFile();
    void beginDownload(const xjw::common::model::ModelAssetFile &asset);
    void consumeReplyData();
    void finishCurrentFile();
    void fail(const QString &message);
    bool existingFileMatches(const xjw::common::model::ModelAssetFile &asset) const;
    qint64 pendingDownloadBytes() const;
    void updateProgress(qint64 currentFileBytes = 0);

    xjw::common::model::ModelAssetPackage _package;
    QString _targetDirectory;
    int _fileIndex = 0;
    qint64 _completedBytes = 0;
    qint64 _currentWrittenBytes = 0;
    QString _partPath;
    QString _errorMessage;
    bool _cancelled = false;
    bool _writeFailed = false;
    QFile _outputFile;
    std::unique_ptr<QCryptographicHash> _hash;
    QNetworkAccessManager *_network = nullptr;
    QNetworkReply *_reply = nullptr;
    QLabel *_statusLabel = nullptr;
    QLabel *_detailLabel = nullptr;
    QProgressBar *_progressBar = nullptr;
    QPushButton *_cancelButton = nullptr;
};
