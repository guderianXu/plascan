// =============================================================================
// 文件: main.cpp
// 模块: PlaScan GUI 入口
// 说明:
//   应用程序主入口，负责：
//   1. 创建 QApplication 实例，完成 Qt 环境初始化
//   2. 初始化全局日志系统（Logger 单例）
//   3. 构造并展示主窗口 MainWindow
//   4. 进入 Qt 事件循环，直到应用退出
// =============================================================================

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFontDatabase>
#include <QIcon>
#include <QIODevice>
#include <QMessageBox>
#include <QMetaObject>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QTimer>
#include "MainWindow.h"
#include "GpuDeviceLease.h"
#include "Logger.h"
#include "PatchMatchCUDA.h"
#include "application/PythonRuntimeDialog.h"
#include "platform/ProjectFileIntegration.h"
#include "runtime/PythonRuntimeManager.h"
#include "runtime/PythonRuntimeLocator.h"
// 抑制 libtiff 读取 GDAL 写入的 GeoTIFF 时产生的 tag 42113 (GDAL_NODATA) 警告
#include <tiffio.h>

#include <algorithm>
#include <clocale>
#include <exception>
#include <limits>
#include <vector>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

class SafeApplication : public QApplication
{
public:
    using QApplication::QApplication;

    bool notify(QObject *receiver, QEvent *event) override
    {
        try
        {
            return QApplication::notify(receiver, event);
        }
        catch (const std::exception &exception)
        {
            const char *receiverClass = receiver ? receiver->metaObject()->className() : "<null>";
            LOG_ERROR("Unhandled std::exception in Qt event loop. receiver=%s, what=%s",
                      receiverClass,
                      exception.what());
            QMessageBox::critical(nullptr,
                                  QStringLiteral("PlaScan 运行异常"),
                                  QStringLiteral("发生未处理的标准异常：\n%1\n\n请查看控制台获取详细信息。")
                                      .arg(QString::fromLocal8Bit(exception.what())));
            return false;
        }
        catch (...)
        {
            const char *receiverClass = receiver ? receiver->metaObject()->className() : "<null>";
            LOG_ERROR("Unhandled unknown exception in Qt event loop. receiver=%s", receiverClass);
            QMessageBox::critical(nullptr,
                                  QStringLiteral("PlaScan 运行异常"),
                                  QStringLiteral("发生未处理的未知异常。\n\n请查看控制台获取详细信息。"));
            return false;
        }
    }
};

namespace
{
void configureConsoleEncoding()
{
#ifdef Q_OS_WIN
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    std::setlocale(LC_ALL, ".UTF-8");
#else
    std::setlocale(LC_ALL, "");
#endif
}

void applyApplicationStyle(QApplication &app)
{
    QApplication::setStyle(QStringLiteral("Fusion"));

    QFile styleFile(QStringLiteral(":/styles/app.qss"));
    if (!styleFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        LOG_WARN("Failed to load application stylesheet: :/styles/app.qss");
        return;
    }

    app.setStyleSheet(QString::fromUtf8(styleFile.readAll()));
}

bool bindPythonRuntime()
{
    const QString python_path = xjw::common::runtime::resolvePythonExecutable(
        QProcessEnvironment::systemEnvironment(), QStringLiteral(PLASCAN_SOURCE_DIR));
    if (python_path.isEmpty())
    {
        LOG_WARN("PlaScan Python runtime was not found. Use Help > Update Python Environment to install it.");
        return false;
    }

    const QByteArray encoded_path = python_path.toUtf8();
    if (QFileInfo(python_path).absoluteFilePath()
        == QFileInfo(PythonRuntimeManager::managedPythonExecutable()).absoluteFilePath())
    {
        PythonRuntimeManager::bindManagedRuntime();
    }
    else
    {
        qputenv("PLASCAN_PYTHON_EXECUTABLE", encoded_path);
        qputenv("PLASCAN_PYTHON", encoded_path);
    }
    LOG_INFO("PlaScan Python runtime bound: %s", encoded_path.constData());
    return true;
}

void configureOpenClDevicePolicy()
{
    const std::vector<xjw::mvs::OpenClDeviceInfo> devices =
        xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    const auto selected = std::find_if(
        devices.cbegin(), devices.cend(), [](const xjw::mvs::OpenClDeviceInfo &device)
        {
            return !xjw::mvs::isNvidiaOpenClVendor(device.vendor);
        });
    if (selected == devices.cend())
    {
        qputenv("PLAMATRIX_OPENCL_DEVICE_INDEX",
                QByteArray::number(std::numeric_limits<int>::max()));
        LOG_WARN("PlaScan OpenCL policy found no non-NVIDIA GPU; NVIDIA OpenCL devices are ignored");
        return;
    }

    qputenv("PLAMATRIX_OPENCL_DEVICE_INDEX", QByteArray::number(selected->index));
    LOG_INFO("PlaScan OpenCL policy selected non-NVIDIA device: index=%d vendor=%s device=%s",
             selected->index,
             selected->vendor.c_str(),
             selected->name.c_str());
}
} // namespace

// main - 程序入口
// 参数:
//   argc - 命令行参数数量
//   argv - 命令行参数字符串数组
// 返回值: int - Qt 事件循环退出码（0 表示正常退出）
int main(int argc, char *argv[])
{
    configureConsoleEncoding();

    std::set_terminate([]()
    {
        LOG_ERROR("std::terminate triggered. Possible uncaught exception across Qt boundary.");
        std::abort();
    });

    // 屏蔽 libtiff 未知 tag 警告（tag 42113 = GDAL_NODATA，libtiff 不识别但无害）
    TIFFSetWarningHandler(nullptr);

    // Qt6 高 DPI 支持（必须在 QApplication 创建前设置）
    QApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    // 创建 Qt 应用程序对象，必须在任何 Qt 对象之前创建
    SafeApplication app(argc, argv);
    // GNOME/桌面集成 — 必须与 .desktop 文件名一致
    app.setApplicationName(QStringLiteral("PlaScan"));
    app.setApplicationVersion(QStringLiteral(PLASCAN_VERSION));
    app.setDesktopFileName(QStringLiteral("plascan"));
    const bool python_runtime_available = bindPythonRuntime();
    applyApplicationStyle(app);

    // 应用图标 — 内嵌 PNG 资源, 不依赖 SVG 插件
    QIcon appIcon(QStringLiteral(":/plascan.png"));
    if (appIcon.isNull())
        appIcon = QIcon::fromTheme(QStringLiteral("plascan"));
    app.setWindowIcon(appIcon);

    // 设置应用程序全局字体（优先使用思源黑体，回退到系统默认）
    QFont appFont;
    QStringList fontFamilies = {"Noto Sans CJK SC", "Source Han Sans CN", "Microsoft YaHei", "SimHei", "sans-serif"};
    for (const QString &family : fontFamilies) 
    {
        if (QFontDatabase::families().contains(family)) 
        {
            appFont.setFamily(family);
            break;
        }
    }
    appFont.setPointSize(10);  // 基准字体大小 10pt
    app.setFont(appFont);
    
    const QString baseDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir(baseDir).mkpath(QStringLiteral("logs"));
    Logger::instance()->setLogDirectory(QDir(baseDir).filePath(QStringLiteral("logs")));
    LOG_INFO("PlaScan GUI started");
    configureOpenClDevicePolicy();

    const auto association_result =
        xjw::gui::platform::ensureProjectFileAssociation(QCoreApplication::applicationFilePath());
    if (!association_result.success)
    {
        LOG_WARN("PlaScan project file association could not be registered: %s",
                 association_result.errorMessage.toUtf8().constData());
    }
    else if (association_result.changed)
    {
        LOG_INFO("PlaScan project file association registered for the current user");
    }

    if (QCoreApplication::arguments().contains(
            QStringLiteral("--register-project-file-association")))
    {
        return association_result.success ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const QString startup_project =
        xjw::gui::platform::startupProjectPath(QCoreApplication::arguments());
    
    try
    {
        // 创建并展示主窗口
        MainWindow mainWindow;
        mainWindow.show();
        QTimer::singleShot(0, &mainWindow, [&mainWindow, startup_project, python_runtime_available]()
        {
            if (!python_runtime_available && !PythonRuntimeManager::startupPromptSuppressed())
            {
                PythonRuntimeDialog dialog(PythonRuntimeDialog::Mode::StartupPrompt, &mainWindow);
                dialog.exec();
            }
            if (!startup_project.isEmpty())
            {
                mainWindow.openProjectFromPath(startup_project);
            }
        });

        // 进入 Qt 事件循环，阻塞直到应用退出
        return app.exec();
    }
    catch (const std::exception &exception)
    {
        LOG_ERROR("Fatal startup std::exception: %s", exception.what());
        QMessageBox::critical(nullptr,
                              QStringLiteral("PlaScan 启动失败"),
                              QStringLiteral("启动时捕获到异常：\n%1\n\n请查看控制台并反馈该信息。")
                                  .arg(QString::fromLocal8Bit(exception.what())));
        return EXIT_FAILURE;
    }
    catch (...)
    {
        LOG_ERROR("Fatal startup unknown exception.");
        QMessageBox::critical(nullptr,
                              QStringLiteral("PlaScan 启动失败"),
                              QStringLiteral("启动时捕获到未知异常。\n\n请查看控制台并反馈该信息。"));
        return EXIT_FAILURE;
    }
}
