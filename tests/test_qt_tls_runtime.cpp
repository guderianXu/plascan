#include <QCoreApplication>
#include <QEventLoop>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSslSocket>
#include <QTimer>
#include <QUrl>

#include <gtest/gtest.h>

namespace
{

TEST(QtTlsRuntime, HasFunctionalBackend)
{
    // Windows 发布版应至少加载 Schannel；允许 OpenSSL 是为了兼容自定义 Qt 构建。
    EXPECT_TRUE(QSslSocket::supportsSsl())
        << "Qt Network was loaded without a functional TLS backend";
    EXPECT_FALSE(QSslSocket::availableBackends().isEmpty());
}

TEST(QtTlsRuntime, DownloadsConfiguredHttpsResource)
{
    const QByteArray configured_url = qgetenv("PLASCAN_TEST_HTTPS_URL");
    if (configured_url.isEmpty())
    {
        GTEST_SKIP() << "Set PLASCAN_TEST_HTTPS_URL to enable the network integration test";
    }

    QNetworkAccessManager manager;
    QNetworkRequest request(QUrl(QString::fromUtf8(configured_url)));
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::NoLessSafeRedirectPolicy);
    request.setHeader(QNetworkRequest::UserAgentHeader,
                      QStringLiteral("PlaScan-TLS-Runtime-Test/1.0"));

    QNetworkReply *reply = manager.get(request);
    QEventLoop loop;
    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, reply, &QNetworkReply::abort);
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    timeout.start(30000);
    loop.exec();

    const QByteArray body = reply->readAll();
    const QNetworkReply::NetworkError error = reply->error();
    const QString error_text = reply->errorString();
    reply->deleteLater();

    EXPECT_EQ(error, QNetworkReply::NoError) << error_text.toStdString();
    EXPECT_FALSE(body.isEmpty());
}

} // namespace

int main(int argc, char **argv)
{
    QCoreApplication application(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
