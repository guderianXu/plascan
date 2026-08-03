#include "ImageMatchingRegistry.h"

#include <QHash>
#include <QMutex>
#include <QMutexLocker>

#include <algorithm>
#include <mutex>

namespace xjw::image_matching
{

#if defined(PLASCAN_HAS_TENSORRT)
void registerSiftLightGlueAlgorithm();
void registerLoMaRAlgorithm();
#endif

namespace
{

struct RegistryEntry
{
    ImageMatchingAlgorithmDescriptor descriptor;
    ImageMatchingAlgorithmFactory factory;
};

QHash<QString, RegistryEntry> &entries()
{
    static QHash<QString, RegistryEntry> value;
    return value;
}

QMutex &registryMutex()
{
    static QMutex mutex;
    return mutex;
}

QString normalizedId(const QString &algorithmId)
{
    return algorithmId.trimmed().toLower();
}

void ensureBuiltInAlgorithms()
{
    static std::once_flag once;
    std::call_once(once, []()
    {
#if defined(PLASCAN_HAS_TENSORRT)
        registerSiftLightGlueAlgorithm();
        registerLoMaRAlgorithm();
#endif
    });
}

} // namespace

bool ImageMatchingRegistry::registerAlgorithm(
    const ImageMatchingAlgorithmDescriptor &descriptor,
    ImageMatchingAlgorithmFactory factory,
    QString *errorMessage)
{
    const QString id = normalizedId(descriptor.id);
    if (id.isEmpty() || descriptor.version == 0 || !factory)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配算法注册信息不完整");
        }
        return false;
    }

    QMutexLocker lock(&registryMutex());
    if (entries().contains(id))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("匹配算法已经注册: %1").arg(id);
        }
        return false;
    }
    RegistryEntry entry;
    entry.descriptor = descriptor;
    entry.descriptor.id = id;
    entry.factory = std::move(factory);
    entries().insert(id, std::move(entry));
    return true;
}

std::vector<ImageMatchingAlgorithmDescriptor> ImageMatchingRegistry::descriptors()
{
    ensureBuiltInAlgorithms();
    QMutexLocker lock(&registryMutex());
    std::vector<ImageMatchingAlgorithmDescriptor> result;
    result.reserve(static_cast<std::size_t>(entries().size()));
    for (const RegistryEntry &entry : entries())
    {
        result.push_back(entry.descriptor);
    }
    std::sort(result.begin(), result.end(),
              [](const auto &left, const auto &right)
              {
                  return left.id < right.id;
              });
    return result;
}

bool ImageMatchingRegistry::contains(const QString &algorithmId)
{
    ensureBuiltInAlgorithms();
    QMutexLocker lock(&registryMutex());
    return entries().contains(normalizedId(algorithmId));
}

std::unique_ptr<IImageMatchingAlgorithm> ImageMatchingRegistry::create(
    const QString &algorithmId,
    const ImageMatchingRuntimeConfig &config,
    QString *errorMessage)
{
    ensureBuiltInAlgorithms();
    ImageMatchingAlgorithmFactory factory;
    {
        QMutexLocker lock(&registryMutex());
        const auto it = entries().constFind(normalizedId(algorithmId));
        if (it == entries().constEnd())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("未注册的影像匹配算法: %1").arg(algorithmId);
            }
            return nullptr;
        }
        factory = it->factory;
    }

    try
    {
        return factory(config);
    }
    catch (const std::exception &error)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("创建影像匹配算法失败: %1")
                .arg(QString::fromUtf8(error.what()));
        }
        return nullptr;
    }
}

} // namespace xjw::image_matching
