#include "plafs/PlaChunkLayout.h"
#include "plafs/PlaDir.h"
#include "plafs/PlaFile.h"
#include "plafs/PlaProjectLayout.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>

namespace
{

    class PlaFsTest : public testing::Test
    {
    protected:
        void SetUp() override
        {
            static std::atomic<unsigned> sequence{0};
            _root = std::filesystem::path(PLASCAN_PLAFS_TMP_DIR) /
                    (std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                     std::to_string(sequence.fetch_add(1)));
            ASSERT_TRUE(xjw::common::plafs::PlaDir(_root).ensure());
        }

        void TearDown() override
        {
            std::error_code error;
            std::filesystem::remove_all(_root, error);
            EXPECT_FALSE(error) << error.message();
        }

        std::filesystem::path _root;
    };

    TEST_F(PlaFsTest, ProjectLayoutCentralizesPhysicalPaths)
    {
        const auto project = _root / "月球测量.plascan";
        std::string error;
        const auto layout = xjw::common::plafs::PlaProjectLayout::open(project, &error);
        ASSERT_TRUE(layout.has_value()) << error;

        EXPECT_EQ(layout->projectFile(), std::filesystem::absolute(project).lexically_normal());
        EXPECT_EQ(layout->filesDirectory(), _root / "月球测量.files");
        EXPECT_EQ(layout->metadataArchive(), _root / "月球测量.files/project.zip");
        EXPECT_EQ(layout->sharedImagesDirectory(), _root / "月球测量.files/shared/images");
        EXPECT_EQ(layout->chunkDirectory(1), _root / "月球测量.files/1");
        EXPECT_EQ(layout->imageMatchesDirectory(1), _root / "月球测量.files/1/assets/image_matches");
        EXPECT_EQ(layout->tiePointsDirectory(1), _root / "月球测量.files/1/assets/tie_points");
        EXPECT_EQ(layout->reconstructionDirectory(1), _root / "月球测量.files/1/reconstruction");
        EXPECT_TRUE(layout->chunkDirectory(0).empty());
        EXPECT_TRUE(layout->chunkArchive(-1).empty());
    }

    TEST_F(PlaFsTest, PlaFileAndPlaDirKeepBasicOperationsSmall)
    {
        const xjw::common::plafs::PlaDir directory(_root / "assets");
        std::string error;
        ASSERT_TRUE(directory.ensure(&error)) << error;
        EXPECT_TRUE(directory.isDirectory());
        EXPECT_TRUE(directory.isEmpty(&error)) << error;
        EXPECT_FALSE(directory.contains("result.json"));

        const xjw::common::plafs::PlaFile file(directory.path() / "result.json");
        ASSERT_TRUE(file.writeAtomic("{\"ok\":true}", &error)) << error;
        EXPECT_TRUE(file.exists());
        EXPECT_TRUE(file.isRegularFile());
        EXPECT_TRUE(directory.contains("result.json"));

        std::string bytes;
        ASSERT_TRUE(file.read(&bytes, &error)) << error;
        EXPECT_EQ(bytes, "{\"ok\":true}");
        EXPECT_FALSE(directory.isEmpty(&error));
    }

    TEST_F(PlaFsTest, ChunkLayoutCentralizesActiveChunkPaths)
    {
        const auto chunk = xjw::common::plafs::PlaChunkLayout::open(_root / "当前 Chunk");
        ASSERT_TRUE(chunk.has_value());

        EXPECT_EQ(chunk->root(), _root / "当前 Chunk");
        EXPECT_EQ(chunk->assetsDirectory(), _root / "当前 Chunk/assets");
        EXPECT_EQ(chunk->imageMatchesDirectory(), _root / "当前 Chunk/assets/image_matches");
        EXPECT_EQ(chunk->tiePointsDirectory(), _root / "当前 Chunk/assets/tie_points");
        EXPECT_EQ(chunk->controlPointsDirectory(), _root / "当前 Chunk/assets/control_points");
        EXPECT_EQ(chunk->cameraReferencesDirectory(), _root / "当前 Chunk/assets/camera_references");
        EXPECT_EQ(chunk->importedDirectory(), _root / "当前 Chunk/assets/imported");
        EXPECT_EQ(chunk->importedCategoryDirectory("models"), _root / "当前 Chunk/assets/imported/models");
        EXPECT_EQ(chunk->importedCategoryDirectory("point_clouds"), _root / "当前 Chunk/assets/imported/point_clouds");
        EXPECT_EQ(chunk->packedDirectory(), _root / "当前 Chunk/assets/packed");
        EXPECT_EQ(chunk->masksDirectory(), _root / "当前 Chunk/assets/masks");
        EXPECT_EQ(chunk->bundleAdjustDirectory(), _root / "当前 Chunk/bundle_adjust");
        EXPECT_EQ(chunk->reportsDirectory(), _root / "当前 Chunk/reports");
        EXPECT_EQ(chunk->reconstructionDirectory(), _root / "当前 Chunk/reconstruction");
        EXPECT_EQ(chunk->resourcesDirectory(), _root / "当前 Chunk/resources");
        EXPECT_EQ(chunk->temporaryFilesPath(), _root / "当前 Chunk/.plascan_tmp/project_files.json");
        EXPECT_EQ(chunk->temporaryResultsPath(), _root / "当前 Chunk/.plascan_tmp/project_results.json");
        EXPECT_EQ(chunk->temporaryConfigPath(), _root / "当前 Chunk/.plascan_tmp/project_config.json");
        EXPECT_EQ(chunk->temporaryUiStatePath(), _root / "当前 Chunk/.plascan_tmp/project_ui_state.json");
        EXPECT_EQ(chunk->markerSetPath(), _root / "当前 Chunk/assets/control_points/marker_set.json");
        EXPECT_EQ(chunk->markerDetectionReviewPath(), _root / "当前 Chunk/assets/control_points/detection_review.json");
        EXPECT_EQ(chunk->cameraReferenceSetPath(),
                  _root / "当前 Chunk/assets/camera_references/camera_reference_set.json");
    }

    TEST_F(PlaFsTest, RejectsEmptyProjectAndAbsoluteDirectoryMembers)
    {
        std::string error;
        EXPECT_FALSE(xjw::common::plafs::PlaProjectLayout::open({}, &error).has_value());
        EXPECT_FALSE(error.empty());
        EXPECT_FALSE(xjw::common::plafs::PlaChunkLayout::open({}, &error).has_value());
        EXPECT_FALSE(error.empty());

        const auto chunk = xjw::common::plafs::PlaChunkLayout::open(_root / "chunk");
        ASSERT_TRUE(chunk.has_value());
        EXPECT_TRUE(chunk->importedCategoryDirectory("").empty());
        EXPECT_TRUE(chunk->importedCategoryDirectory("../escape").empty());
        EXPECT_TRUE(chunk->importedCategoryDirectory("nested/escape").empty());
        EXPECT_TRUE(chunk->importedCategoryDirectory("/absolute").empty());

        const xjw::common::plafs::PlaDir directory(_root);
        EXPECT_FALSE(directory.contains(std::filesystem::path("/tmp/escape")));
        EXPECT_FALSE(directory.contains(std::filesystem::path("nested/../escape")));
        EXPECT_FALSE(directory.contains({}));
    }

} // namespace
