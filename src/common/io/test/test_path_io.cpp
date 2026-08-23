#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QStringList>
#include <QTemporaryDir>

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace
{

std::filesystem::path temporaryRoot(QTemporaryDir *temporary)
{
    EXPECT_TRUE(temporary != nullptr);
    EXPECT_TRUE(temporary->isValid());
    return xjw::common::io::toFilesystemPath(temporary->path());
}

} // namespace

TEST(PathIOTest, ComparesExistingAndNonexistentPaths)
{
    QTemporaryDir temporary;
    const std::filesystem::path root = temporaryRoot(&temporary);
    const std::filesystem::path parent = root / "existing";
    std::filesystem::create_directories(parent / "child");

    const auto same = xjw::common::io::comparePathsSafely(parent / "future",
                                                           parent / "child" / ".." / "future");
    ASSERT_TRUE(same.valid) << same.error.message();
    EXPECT_TRUE(same.equivalent);
    EXPECT_FALSE(same.firstIsAncestorOfSecond);
    EXPECT_FALSE(same.secondIsAncestorOfFirst);

    const auto nested = xjw::common::io::comparePathsSafely(parent, parent / "missing" / "leaf");
    ASSERT_TRUE(nested.valid) << nested.error.message();
    EXPECT_FALSE(nested.equivalent);
    EXPECT_TRUE(nested.firstIsAncestorOfSecond);
    EXPECT_FALSE(nested.secondIsAncestorOfFirst);
}

TEST(PathIOTest, ResolvesAliasesBeforeMissingLeaf)
{
    QTemporaryDir temporary;
    const std::filesystem::path root = temporaryRoot(&temporary);
    const std::filesystem::path real = root / "real";
    const std::filesystem::path alias = root / "alias";
    std::filesystem::create_directories(real);

    std::error_code error;
    std::filesystem::create_directory_symlink(real, alias, error);
    std::filesystem::path aliased_leaf = alias / "future.dat";
    if (error)
    {
        // 普通 Windows 环境可能没有创建符号链接的权限；仍验证不存在末级
        // 与词法别名，具备权限的平台则覆盖真实目录符号链接。
        std::filesystem::create_directories(real / "child");
        aliased_leaf = real / "child" / ".." / "future.dat";
    }

    const auto comparison = xjw::common::io::comparePathsSafely(aliased_leaf,
                                                                 real / "future.dat");
    ASSERT_TRUE(comparison.valid) << comparison.error.message();
    EXPECT_TRUE(comparison.equivalent);
}

TEST(PathIOTest, DetectsHardLinkEquivalence)
{
    QTemporaryDir temporary;
    const std::filesystem::path root = temporaryRoot(&temporary);
    const std::filesystem::path first = root / "first.dat";
    const std::filesystem::path second = root / "second.dat";
    {
        std::ofstream out(first, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.good());
        out << "payload";
    }

    std::error_code error;
    std::filesystem::create_hard_link(first, second, error);
    if (error)
    {
        GTEST_SKIP() << "当前文件系统不允许创建硬链接: " << error.message();
    }

    const auto comparison = xjw::common::io::comparePathsSafely(first, second);
    ASSERT_TRUE(comparison.valid) << comparison.error.message();
    EXPECT_TRUE(comparison.equivalent);
}

TEST(PathIOTest, IdentifiesFilesystemRoot)
{
    QTemporaryDir temporary;
    const std::filesystem::path root = temporaryRoot(&temporary);
    const std::filesystem::path filesystem_root = root.root_path();

    const auto comparison = xjw::common::io::comparePathsSafely(filesystem_root, root);
    ASSERT_TRUE(comparison.valid) << comparison.error.message();
    EXPECT_TRUE(comparison.firstIsRoot);
    EXPECT_FALSE(comparison.secondIsRoot);
    EXPECT_TRUE(comparison.firstIsAncestorOfSecond);
}

TEST(PathIOTest, InvalidPathFailsClosed)
{
    QTemporaryDir temporary;
    const std::filesystem::path root = temporaryRoot(&temporary);

    const auto comparison = xjw::common::io::comparePathsSafely({}, root);
    EXPECT_FALSE(comparison.valid);
    EXPECT_TRUE(static_cast<bool>(comparison.error));
}

TEST(PathIOTest, SortsFileNamesNaturallyWithoutLocaleOrIntegerLimits)
{
    QStringList names{
        QStringLiteral("frame_10.PNG"),
        QStringLiteral("FRAME_2.png"),
        QStringLiteral("frame_000000000000000000000000000000003.png"),
        QStringLiteral("frame_1.png")};

    std::stable_sort(names.begin(), names.end(), xjw::common::io::naturalFileNameLessThan);

    EXPECT_EQ(names,
              (QStringList{QStringLiteral("frame_1.png"),
                           QStringLiteral("FRAME_2.png"),
                           QStringLiteral("frame_000000000000000000000000000000003.png"),
                           QStringLiteral("frame_10.PNG")}));
    EXPECT_FALSE(xjw::common::io::naturalFileNameLessThan(
        QStringLiteral("frame_02.png"), QStringLiteral("FRAME_2.PNG")));
    EXPECT_FALSE(xjw::common::io::naturalFileNameLessThan(
        QStringLiteral("FRAME_2.PNG"), QStringLiteral("frame_02.png")));
}
