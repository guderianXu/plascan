#include "CameraFormatConverter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        static std::atomic<unsigned long long> sequence{0};
        const auto tick = static_cast<unsigned long long>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        _path = std::filesystem::temp_directory_path()
            / ("plascan_camera_converter_safety_" + std::to_string(tick) + "_"
               + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(_path);
    }

    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

    ~TemporaryDirectory()
    {
        std::error_code ignored_error;
        std::filesystem::remove_all(_path, ignored_error);
    }

    const std::filesystem::path &path() const
    {
        return _path;
    }

private:
    std::filesystem::path _path;
};

void writeText(const std::filesystem::path &path, const std::string &text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(out.good()) << path;
    out << text;
    ASSERT_TRUE(out.good()) << path;
}

std::string readText(const std::filesystem::path &path)
{
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void writeMiddleburyDataset(const std::filesystem::path &source,
                            const std::vector<std::string> &imageNames,
                            bool createImages = true)
{
    std::ostringstream parameters;
    parameters << imageNames.size() << "\n";
    for (const std::string &image_name : imageNames)
    {
        if (createImages)
        {
            writeText(source / image_name, "fake image");
        }
        parameters << image_name
                   << " 100 0 40 0 100 40 0 0 1"
                      " 1 0 0 0 1 0 0 0 1"
                      " 0 0 0\n";
    }
    writeText(source / "dataset_par.txt", parameters.str());
}

xjw::camera::CameraConversionOptions conversionOptions(const std::filesystem::path &source,
                                                        const std::filesystem::path &output)
{
    xjw::camera::CameraConversionOptions options;
    options.format = xjw::camera::CameraFormat::MiddleburyPar;
    options.inputPath = source;
    options.outputDir = output;
    options.datasetId = "safety-test";
    options.overwrite = true;
    return options;
}

size_t transactionResidueCount(const std::filesystem::path &parent)
{
    size_t count = 0;
    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(parent))
    {
        const std::string name = entry.path().filename().string();
        if (name.rfind(".plascan-camera-", 0) == 0)
        {
            ++count;
        }
    }
    return count;
}

} // namespace

TEST(CameraFormatConverterSafetyTest, RejectsOutputThatContainsInputDirectory)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path source = root / "dataset";
    writeMiddleburyDataset(source, {"image.png"});
    writeText(root / "unrelated.txt", "keep");

    const auto result = xjw::camera::convertCameraDataset(conversionOptions(source, root));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.errorMessage.find("等于或包含输入路径"), std::string::npos);
    EXPECT_TRUE(std::filesystem::exists(source / "dataset_par.txt"));
    EXPECT_EQ(readText(root / "unrelated.txt"), "keep");
}

TEST(CameraFormatConverterSafetyTest, RejectsAliasThatContainsInput)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path real_output = root / "real-output";
    std::filesystem::path alias_output = root / "alias-output";
    const std::filesystem::path source = real_output / "dataset";
    writeMiddleburyDataset(source, {"image.png"});

    std::error_code error;
    std::filesystem::create_directory_symlink(real_output, alias_output, error);
    const bool alias_is_link = !error;
    if (error)
    {
        // 无符号链接权限时改用等价的词法别名，保证 Windows 默认环境也执行
        // 覆盖保护；具备权限的平台仍会验证真实目录符号链接。
        alias_output = real_output / "missing" / "..";
    }

    const auto result = xjw::camera::convertCameraDataset(conversionOptions(source, alias_output));

    EXPECT_FALSE(result.success);
    if (alias_is_link)
    {
        EXPECT_NE(result.errorMessage.find("符号链接或目录联接"), std::string::npos);
    }
    else
    {
        EXPECT_NE(result.errorMessage.find("等于或包含输入路径"), std::string::npos);
    }
    EXPECT_TRUE(std::filesystem::exists(source / "image.png"));
}

TEST(CameraFormatConverterSafetyTest, RejectsLinkedOutputWithoutTouchingExternalTarget)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path source = root / "dataset";
    const std::filesystem::path external_output = root / "external-output";
    const std::filesystem::path linked_output = root / "linked-output";
    writeMiddleburyDataset(source, {"image.png"});
    writeText(external_output / "sentinel.txt", "preserve me");

    std::error_code error;
    std::filesystem::create_directory_symlink(
        external_output, linked_output, error);
    if (error)
    {
        // Windows 未开启开发者模式时创建目录链接可能被系统拒绝；Linux CI
        // 与允许链接的 Windows 环境会执行完整的外部目标保护断言。
        return;
    }

    const auto result = xjw::camera::convertCameraDataset(
        conversionOptions(source, linked_output));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.errorMessage.find("符号链接或目录联接"), std::string::npos);
    EXPECT_EQ(readText(external_output / "sentinel.txt"), "preserve me");
    EXPECT_FALSE(std::filesystem::exists(external_output / "summary.json"));
}

TEST(CameraFormatConverterSafetyTest, RejectsOutputThatContainsReferencedImage)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path source = root / "camera-input";
    const std::filesystem::path output = root / "old-output";
    writeText(output / "referenced.png", "source image");
    writeText(output / "old-result.txt", "old result");
    writeMiddleburyDataset(source, {"../old-output/referenced.png"}, false);

    const auto result = xjw::camera::convertCameraDataset(conversionOptions(source, output));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.errorMessage.find("等于或包含输入路径"), std::string::npos);
    EXPECT_EQ(readText(output / "referenced.png"), "source image");
    EXPECT_EQ(readText(output / "old-result.txt"), "old result");
}

TEST(CameraFormatConverterSafetyTest, RejectsFilesystemRootWithoutInspectingOverwriteContents)
{
    TemporaryDirectory temporary;
    const std::filesystem::path source = temporary.path() / "dataset";
    writeMiddleburyDataset(source, {"image.png"});
    auto options = conversionOptions(source, temporary.path().root_path());
    options.overwrite = false;

    const auto result = xjw::camera::convertCameraDataset(options);

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.errorMessage.find("文件系统根目录"), std::string::npos);
    EXPECT_TRUE(std::filesystem::exists(source / "image.png"));
}

TEST(CameraFormatConverterSafetyTest, MissingImageKeepsExistingOutputAndLeavesNoResidue)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path source = root / "dataset";
    const std::filesystem::path output = root / "output";
    writeMiddleburyDataset(source, {"present.png", "missing.png"}, false);
    writeText(source / "present.png", "source image");
    writeText(output / "cameras" / "old.tsai", "old camera");
    writeText(output / "old-result.txt", "old result");

    const auto result = xjw::camera::convertCameraDataset(conversionOptions(source, output));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.errorMessage.find("影像不存在"), std::string::npos);
    EXPECT_EQ(readText(output / "cameras" / "old.tsai"), "old camera");
    EXPECT_EQ(readText(output / "old-result.txt"), "old result");
    EXPECT_EQ(transactionResidueCount(root), 0U);
}

TEST(CameraFormatConverterSafetyTest,
     SourceImageDeletedBeforeCommitKeepsExistingOutputAndLeavesNoResidue)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path source = root / "dataset";
    const std::filesystem::path output = root / "output";
    const std::filesystem::path image = source / "image.png";
    writeMiddleburyDataset(source, {"image.png"});
    writeText(output / "cameras" / "old.tsai", "old camera");
    writeText(output / "old-result.txt", "old result");

    auto options = conversionOptions(source, output);
    options.beforeCommitHook = [&]()
    {
        std::error_code error;
        if (!std::filesystem::remove(image, error) || error)
        {
            throw std::runtime_error(
                "test hook could not delete source image");
        }
    };
    const auto result = xjw::camera::convertCameraDataset(options);

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.errorMessage.find("提交前复检源影像失败"),
              std::string::npos) << result.errorMessage;
    EXPECT_EQ(readText(output / "cameras" / "old.tsai"), "old camera");
    EXPECT_EQ(readText(output / "old-result.txt"), "old result");
    EXPECT_FALSE(std::filesystem::exists(output / "summary.json"));
    EXPECT_EQ(transactionResidueCount(root), 0U);
}

TEST(CameraFormatConverterSafetyTest,
     SourceImageReplacedBeforeCommitKeepsExistingOutputAndLeavesNoResidue)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path source = root / "dataset";
    const std::filesystem::path output = root / "output";
    const std::filesystem::path image = source / "image.png";
    const std::filesystem::path replacement = source / "replacement.tmp";
    writeMiddleburyDataset(source, {"image.png"});
    const auto original_write_time = std::filesystem::last_write_time(image);
    writeText(replacement, "new! image");
    std::filesystem::last_write_time(replacement, original_write_time);
    ASSERT_EQ(std::filesystem::file_size(image),
              std::filesystem::file_size(replacement));
    writeText(output / "cameras" / "old.tsai", "old camera");
    writeText(output / "old-result.txt", "old result");

    auto options = conversionOptions(source, output);
    options.beforeCommitHook = [&]()
    {
        std::error_code error;
        if (!std::filesystem::remove(image, error) || error)
        {
            throw std::runtime_error(
                "test hook could not remove original source image");
        }
        std::filesystem::rename(replacement, image, error);
        if (error)
        {
            throw std::runtime_error(
                "test hook could not install replacement source image");
        }
    };
    const auto result = xjw::camera::convertCameraDataset(options);

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.errorMessage.find("源影像已删除、替换或修改"),
              std::string::npos) << result.errorMessage;
    EXPECT_EQ(readText(image), "new! image");
    EXPECT_EQ(readText(output / "cameras" / "old.tsai"), "old camera");
    EXPECT_EQ(readText(output / "old-result.txt"), "old result");
    EXPECT_FALSE(std::filesystem::exists(output / "summary.json"));
    EXPECT_EQ(transactionResidueCount(root), 0U);
}

TEST(CameraFormatConverterSafetyTest, DuplicateCameraStemKeepsExistingOutputAndLeavesNoResidue)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path source = root / "dataset";
    const std::filesystem::path output = root / "output";
    writeMiddleburyDataset(source, {"same-name.jpg", "same-name.png"});
    writeText(output / "cameras" / "old.tsai", "old camera");
    writeText(output / "old-result.txt", "old result");

    const auto result = xjw::camera::convertCameraDataset(conversionOptions(source, output));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.errorMessage.find("同一输出文件"), std::string::npos) << result.errorMessage;
    EXPECT_EQ(readText(output / "cameras" / "old.tsai"), "old camera");
    EXPECT_EQ(readText(output / "old-result.txt"), "old result");
    EXPECT_FALSE(std::filesystem::exists(output / "cameras" / "same-name.tsai"));
    EXPECT_EQ(transactionResidueCount(root), 0U);
}

TEST(CameraFormatConverterSafetyTest, SuccessfulOverwriteReplacesOutputAndLeavesNoResidue)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path source = root / "dataset";
    const std::filesystem::path output = root / "output";
    writeMiddleburyDataset(source, {"image.png"});
    writeText(output / "cameras" / "old.tsai", "old camera");
    writeText(output / "old-result.txt", "old result");

    const auto result = xjw::camera::convertCameraDataset(conversionOptions(source, output));

    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_FALSE(std::filesystem::exists(output / "old-result.txt"));
    EXPECT_FALSE(std::filesystem::exists(output / "cameras" / "old.tsai"));
    EXPECT_TRUE(std::filesystem::exists(output / "cameras" / "image.tsai"));
    EXPECT_TRUE(std::filesystem::exists(output / "image_camera.lis"));
    EXPECT_TRUE(std::filesystem::exists(output / "summary.json"));
    EXPECT_TRUE(result.retainedBackupPath.empty());
    EXPECT_EQ(transactionResidueCount(root), 0U);
}

TEST(CameraFormatConverterSafetyTest, NormalizesMissingIntermediateComponentsInReturnedPaths)
{
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path();
    const std::filesystem::path source = root / "dataset";
    const std::filesystem::path normalized_output = root / "output";
    const std::filesystem::path aliased_output = root / "missing" / ".." / "output";
    writeMiddleburyDataset(source, {"image.png"});

    const auto result = xjw::camera::convertCameraDataset(
        conversionOptions(source, aliased_output));

    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(result.outputDir, normalized_output);
    EXPECT_EQ(result.imageCameraList, normalized_output / "image_camera.lis");
    EXPECT_EQ(result.summaryPath, normalized_output / "summary.json");
    ASSERT_EQ(result.writtenCameraFiles.size(), 1U);
    EXPECT_EQ(result.writtenCameraFiles.front(),
              normalized_output / "cameras" / "image.tsai");
    EXPECT_TRUE(std::filesystem::exists(result.imageCameraList));
    EXPECT_TRUE(std::filesystem::exists(result.summaryPath));
    EXPECT_TRUE(std::filesystem::exists(result.writtenCameraFiles.front()));
}
