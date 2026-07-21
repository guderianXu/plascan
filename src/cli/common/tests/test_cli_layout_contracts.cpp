#include "CliTestSupport.h"

TEST(CliModuleLayoutTest, GroupsTargetsByDomainAndBuildsSharedPhotogrammetrySupportOnce)
{
    const QString rootCmake = readSourceFile(QStringLiteral("src/cli/CMakeLists.txt"));
    const QString commonCmake = readSourceFile(QStringLiteral("src/cli/common/CMakeLists.txt"));

    expectContainsAll(rootCmake, {
        "add_subdirectory(common)",
        "add_subdirectory(camera)",
        "add_subdirectory(control_points)",
        "add_subdirectory(features)",
        "add_subdirectory(dense)",
        "add_subdirectory(reconstruction)",
        "add_subdirectory(workflows)",
        "add_subdirectory(quality)",
    });
    expectNotContainsAll(rootCmake, {
        "cli_feature_extract.cpp",
        "cli_dense_match.cpp",
        "cli_reconstruct_pipeline.cpp",
    });
    expectContainsAll(commonCmake, {
        "add_library(plascan_cli_photogrammetry_common STATIC",
        "cli_photogrammetry_common.cpp",
    });

    EXPECT_FALSE(QFileInfo::exists(
        QDir(repoRoot()).filePath(QStringLiteral("tests/test_cli_contracts.cpp"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(repoRoot()).filePath(
        QStringLiteral("src/cli/camera/tests/test_camera_cli.cpp"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(repoRoot()).filePath(
        QStringLiteral("src/cli/features/tests/test_features_cli.cpp"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(repoRoot()).filePath(
        QStringLiteral("src/cli/dense/tests/test_dense_cli.cpp"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(repoRoot()).filePath(
        QStringLiteral("src/cli/reconstruction/tests/test_reconstruction_cli.cpp"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(repoRoot()).filePath(
        QStringLiteral("src/cli/workflows/tests/test_workflow_cli.cpp"))));
    EXPECT_TRUE(QFileInfo::exists(QDir(repoRoot()).filePath(
        QStringLiteral("src/cli/quality/tests/test_quality_cli.cpp"))));
}

TEST(WindowsCudaBuildScriptContractTest, UsesShortVcpkgWorkRootsForManifestInstall)
{
    const QString script = readSourceFile(QStringLiteral("scripts/build_win/build_windows_cuda.ps1"));

    expectContainsAll(script,
                      {"[string] $VcpkgBuildtreesRoot = \"\"",
                       "[string] $VcpkgPackagesRoot = \"\"",
                       "[string] $VcpkgDownloadsRoot = \"\"",
                       "$vcpkgWorkDrive = Split-Path -Qualifier $BuildDir",
                       "Join-Path $vcpkgWorkDrive \"vbt\"",
                       "Join-Path $vcpkgWorkDrive \"vpk\"",
                       "Join-Path $vcpkgWorkDrive \"vdl\"",
                       "--x-buildtrees-root=",
                       "--x-packages-root=",
                       "--downloads-root=",
                       "-DVCPKG_INSTALL_OPTIONS="});
}

TEST(WindowsCudaBuildScriptContractTest, PersistsDependencyPrefixForAutomaticReconfigure)
{
    const QString script = readSourceFile(QStringLiteral("scripts/build_win/build_windows_cuda.ps1"));

    expectContainsAll(script,
                      {"$env:CMAKE_PREFIX_PATH = ($tripletRootCMake, $torchPathCMake) -join ';'",
                       "-DCMAKE_PREFIX_PATH=$env:CMAKE_PREFIX_PATH",
                       "-UQt6*_DIR"});
}

TEST(WindowsCudaBuildScriptContractTest, NativeWarningsDoNotBypassExitCodeChecks)
{
    const QString script = readSourceFile(QStringLiteral("scripts/build_win/build_windows_cuda.ps1"));

    expectContainsAll(script,
                      {"function Invoke-NativeCommand",
                       "$ErrorActionPreference = \"Continue\"",
                       "-ExitCode ([ref]$configureExitCode)",
                       "-ExitCode ([ref]$buildExitCode)",
                       "-ExitCode ([ref]$ctestExitCode)",
                       "CMake configure failed with exit code $configureExitCode",
                       "CMake build failed with exit code $buildExitCode",
                       "CTest failed with exit code $ctestExitCode"});
    expectNotContainsAll(script, {"& $CMakeExe @configureArgs", "& $CMakeExe @buildArgs", "& $ctestExe @ctestArgs"});
}

TEST(WindowsCMakeBuildContractTest, DefersGtestDiscoveryUntilCtest)
{
    const QString cmake = readSourceFile(QStringLiteral("CMakeLists.txt"));

    expectContainsAll(cmake,
                      {"if(WIN32)",
                       "set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)"});
}
