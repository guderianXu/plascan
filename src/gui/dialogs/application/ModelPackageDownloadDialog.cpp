#include "application/ModelPackageDownloadDialog.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QLabel>
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProgressBar>
#include <QPushButton>
#include <QStorageInfo>
#include <QSslSocket>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

namespace
{

QString humanReadableBytes(qint64 bytes)
{
    constexpr qint64 kKiB = 1024;
    constexpr qint64 kMiB = 1024 * kKiB;
    constexpr qint64 kGiB = 1024 * kMiB;
    if (bytes >= kGiB)
    {
        return QStringLiteral("%1 GiB").arg(static_cast<double>(bytes) / kGiB, 0, 'f', 2);
    }
    if (bytes >= kMiB)
    {
        return QStringLiteral("%1 MiB").arg(static_cast<double>(bytes) / kMiB, 0, 'f', 1);
    }
    return QStringLiteral("%1 KiB").arg(static_cast<double>(bytes) / kKiB, 0, 'f', 1);
}

QByteArray fileSha256(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return {};
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
    {
        hash.addData(file.read(4 * 1024 * 1024));
    }
    return hash.result().toHex();
}

} // namespace

ModelPackageDownloadDialog::ModelPackageDownloadDialog(
    xjw::common::model::ModelAssetPackage package,
    QString targetDirectory,
    QWidget *parent)
    : QDialog(parent)
    , _package(std::move(package))
    , _targetDirectory(QDir::cleanPath(std::move(targetDirectory)))
{
    setWindowTitle(QStringLiteral("下载模型"));
    setModal(true);
    setMinimumWidth(560);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(20, 18, 20, 18);
    layout->setSpacing(12);

    auto *title = new QLabel(_package.displayName, this);
    QFont titleFont = title->font();
    titleFont.setBold(true);
    title->setFont(titleFont);
    layout->addWidget(title);

    auto *compatibility = new QLabel(_package.compatibilitySummary, this);
    compatibility->setWordWrap(true);
    layout->addWidget(compatibility);

    _detailLabel = new QLabel(
        QStringLiteral("保存到：%1\n总大小：%2")
            .arg(QDir::toNativeSeparators(_targetDirectory),
                 humanReadableBytes(_package.totalBytes())),
        this);
    _detailLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _detailLabel->setWordWrap(true);
    layout->addWidget(_detailLabel);

    _statusLabel = new QLabel(QStringLiteral("准备下载..."), this);
    _statusLabel->setWordWrap(true);
    layout->addWidget(_statusLabel);

    _progressBar = new QProgressBar(this);
    _progressBar->setRange(0, 1000);
    _progressBar->setValue(0);
    _progressBar->setFormat(QStringLiteral("0.0%"));
    layout->addWidget(_progressBar);

    _cancelButton = new QPushButton(QStringLiteral("取消"), this);
    connect(_cancelButton, &QPushButton::clicked, this, &ModelPackageDownloadDialog::reject);
    layout->addWidget(_cancelButton, 0, Qt::AlignRight);

    _network = new QNetworkAccessManager(this);
    QTimer::singleShot(0, this, [this]() { start(); });
}

bool ModelPackageDownloadDialog::downloadPackage(
    const xjw::common::model::ModelAssetPackage &package,
    const QString &targetDirectory,
    QWidget *parent,
    QString *errorMessage)
{
    ModelPackageDownloadDialog dialog(package, targetDirectory, parent);
    const bool accepted = dialog.exec() == QDialog::Accepted;
    if (errorMessage)
    {
        *errorMessage = dialog.errorMessage();
    }
    return accepted;
}

QString ModelPackageDownloadDialog::errorMessage() const
{
    return _errorMessage;
}

void ModelPackageDownloadDialog::reject()
{
    _cancelled = true;
    if (_reply)
    {
        _reply->abort();
    }
    _outputFile.close();
    if (!_partPath.isEmpty())
    {
        QFile::remove(_partPath);
    }
    if (_errorMessage.isEmpty())
    {
        _errorMessage = QStringLiteral("用户取消了模型下载");
    }
    QDialog::reject();
}

void ModelPackageDownloadDialog::start()
{
    if (!_package.isValid())
    {
        fail(QStringLiteral("模型资产目录无效"));
        return;
    }
    if (!QSslSocket::supportsSsl())
    {
        const QString backends = QSslSocket::availableBackends().join(QStringLiteral(", "));
        fail(QStringLiteral(
                 "当前程序没有可用的 HTTPS/TLS 后端，无法下载模型。\n"
                 "请确认 Qt TLS 插件已部署：安装包应包含 plugins/tls，开发构建应包含 bin/tls。\n"
                 "程序目录：%1\n检测到的后端：%2")
                 .arg(QDir::toNativeSeparators(QCoreApplication::applicationDirPath()),
                      backends.isEmpty() ? QStringLiteral("无") : backends));
        return;
    }
    if (!QDir().mkpath(_targetDirectory))
    {
        fail(QStringLiteral("无法创建模型目录：%1").arg(_targetDirectory));
        return;
    }
    const QStorageInfo storage(_targetDirectory);
    constexpr qint64 kDownloadReserveBytes = 64LL * 1024LL * 1024LL;
    if (storage.isValid() && storage.isReady() &&
        storage.bytesAvailable() < _package.totalBytes() + kDownloadReserveBytes)
    {
        fail(QStringLiteral("模型目录可用空间不足，需要约 %1（另预留 64 MiB）：%2")
                 .arg(humanReadableBytes(_package.totalBytes()), _targetDirectory));
        return;
    }
    startNextFile();
}

void ModelPackageDownloadDialog::startNextFile()
{
    if (_cancelled)
    {
        return;
    }
    if (_fileIndex >= _package.files.size())
    {
        _progressBar->setValue(1000);
        _progressBar->setFormat(QStringLiteral("100.0%"));
        _statusLabel->setText(QStringLiteral("模型下载并校验完成"));
        _cancelButton->setText(QStringLiteral("完成"));
        QTimer::singleShot(0, this, &QDialog::accept);
        return;
    }

    const auto &asset = _package.files.at(_fileIndex);
    _statusLabel->setText(
        QStringLiteral("正在校验 %1（%2/%3）")
            .arg(asset.fileName)
            .arg(_fileIndex + 1)
            .arg(_package.files.size()));
    if (existingFileMatches(asset))
    {
        _completedBytes += asset.bytes;
        ++_fileIndex;
        updateProgress();
        QTimer::singleShot(0, this, [this]() { startNextFile(); });
        return;
    }
    beginDownload(asset);
}

void ModelPackageDownloadDialog::beginDownload(
    const xjw::common::model::ModelAssetFile &asset)
{
    const QString finalPath = QDir(_targetDirectory).filePath(asset.fileName);
    _partPath = finalPath + QStringLiteral(".part");
    QFile::remove(_partPath);
    _outputFile.setFileName(_partPath);
    if (!_outputFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        fail(QStringLiteral("无法写入临时模型文件：%1").arg(_partPath));
        return;
    }

    _hash = std::make_unique<QCryptographicHash>(QCryptographicHash::Sha256);
    _currentWrittenBytes = 0;
    _writeFailed = false;
    _statusLabel->setText(
        QStringLiteral("正在下载 %1（%2/%3）")
            .arg(asset.fileName)
            .arg(_fileIndex + 1)
            .arg(_package.files.size()));

    QNetworkRequest request(QUrl(asset.downloadUrl));
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::NoLessSafeRedirectPolicy);
    request.setHeader(QNetworkRequest::UserAgentHeader,
                      QStringLiteral("PlaScan-Model-Downloader/1.0"));
    _reply = _network->get(request);
    connect(_reply, &QNetworkReply::readyRead, this, [this]() { consumeReplyData(); });
    connect(_reply, &QNetworkReply::downloadProgress, this,
            [this](qint64 received, qint64) { updateProgress(received); });
    connect(_reply, &QNetworkReply::finished, this, [this]() { finishCurrentFile(); });
}

void ModelPackageDownloadDialog::consumeReplyData()
{
    if (!_reply || _writeFailed)
    {
        return;
    }
    const QByteArray data = _reply->readAll();
    if (data.isEmpty())
    {
        return;
    }
    if (_outputFile.write(data) != data.size())
    {
        _writeFailed = true;
        _reply->abort();
        return;
    }
    _hash->addData(data);
    _currentWrittenBytes += data.size();
}

void ModelPackageDownloadDialog::finishCurrentFile()
{
    if (!_reply)
    {
        return;
    }
    consumeReplyData();
    const auto networkError = _reply->error();
    const QString networkMessage = _reply->errorString();
    _reply->deleteLater();
    _reply = nullptr;
    _outputFile.close();

    if (_cancelled)
    {
        QFile::remove(_partPath);
        return;
    }
    if (_writeFailed)
    {
        QFile::remove(_partPath);
        fail(QStringLiteral("写入模型文件失败，磁盘空间可能不足：%1").arg(_partPath));
        return;
    }
    if (networkError != QNetworkReply::NoError)
    {
        QFile::remove(_partPath);
        fail(QStringLiteral("下载模型失败：%1").arg(networkMessage));
        return;
    }

    const auto &asset = _package.files.at(_fileIndex);
    const QByteArray actualHash = _hash->result().toHex();
    if (_currentWrittenBytes != asset.bytes ||
        actualHash.compare(asset.sha256.toLatin1(), Qt::CaseInsensitive) != 0)
    {
        QFile::remove(_partPath);
        fail(QStringLiteral("模型校验失败：%1").arg(asset.fileName));
        return;
    }

    const QString finalPath = QDir(_targetDirectory).filePath(asset.fileName);
    if (QFileInfo::exists(finalPath) && !QFile::remove(finalPath))
    {
        QFile::remove(_partPath);
        fail(QStringLiteral("无法替换旧模型文件：%1").arg(finalPath));
        return;
    }
    if (!QFile::rename(_partPath, finalPath))
    {
        QFile::remove(_partPath);
        fail(QStringLiteral("无法提交已校验模型文件：%1").arg(finalPath));
        return;
    }

    _completedBytes += asset.bytes;
    ++_fileIndex;
    _partPath.clear();
    _hash.reset();
    updateProgress();
    startNextFile();
}

void ModelPackageDownloadDialog::fail(const QString &message)
{
    _errorMessage = message;
    _statusLabel->setText(message);
    _cancelButton->setText(QStringLiteral("关闭"));
    QMessageBox::critical(this, QStringLiteral("下载模型"), message);
    QDialog::reject();
}

bool ModelPackageDownloadDialog::existingFileMatches(
    const xjw::common::model::ModelAssetFile &asset) const
{
    const QString path = QDir(_targetDirectory).filePath(asset.fileName);
    const QFileInfo info(path);
    if (!info.isFile() || info.size() != asset.bytes)
    {
        return false;
    }
    return fileSha256(path).compare(asset.sha256.toLatin1(), Qt::CaseInsensitive) == 0;
}

void ModelPackageDownloadDialog::updateProgress(qint64 currentFileBytes)
{
    const qint64 total = _package.totalBytes();
    const qint64 completed = std::min(total, _completedBytes + std::max<qint64>(0, currentFileBytes));
    const int value = total > 0
        ? static_cast<int>((completed * 1000) / total)
        : 0;
    _progressBar->setValue(value);
    _progressBar->setFormat(QStringLiteral("%1%（%2 / %3）")
                                .arg(static_cast<double>(value) / 10.0, 0, 'f', 1)
                                .arg(humanReadableBytes(completed))
                                .arg(humanReadableBytes(total)));
}
