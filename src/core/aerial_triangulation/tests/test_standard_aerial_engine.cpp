#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>

#include "engine/PinholeEngine.h"
#include "engine/RpcEngine.h"
#include "engine/TiePointGraphReader.h"
#include "file/FileIO.h"
#include "file/JsonFile.h"
#include "reporting/SparsePlyWriter.h"

namespace
{
    namespace engine = xjw::aerial_triangulation::engine;
    using namespace xjw::common::file;
    using Json = nlohmann::json;

    std::shared_ptr<const placamera::FramePinholeModel> makePinholeCamera(const char* imageId)
    {
        const placamera::FrameId frame("world");
        const auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId(std::string("definition-") + imageId),
            placamera::FrameIntrinsics{500.0, 500.0, 320.0, 240.0, 1.0, 1, 1},
            {},
            placamera::PixelConvention::PixelCenter,
            frame);
        return std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId(std::string("instance-") + imageId),
            placamera::ImageId(imageId),
            definition,
            placamera::ImageSize{640, 480},
            placamera::Pose::create(frame, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
    }

    class StandardAerialEngineTest : public testing::Test
    {
    protected:
        void SetUp() override
        {
            static std::atomic<unsigned> sequence{0};
            _directory = pathFromUtf8(PLASCAN_STANDARD_AERIAL_TMP_DIR) /
                         (std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                          std::to_string(sequence.fetch_add(1)));
            ASSERT_TRUE(ensureDirectory(_directory));
            _images = {_directory / pathFromUtf8("影像甲.tif"), _directory / pathFromUtf8("影像乙.tif")};
        }
        void TearDown() override
        {
            std::error_code error;
            std::filesystem::remove_all(_directory, error);
            EXPECT_FALSE(error) << error.message();
        }
        Json document() const
        {
            return {{"format", "plascan_tie_points"},
                    {"format_version", 3},
                    {"images",
                     {{{"image_id", 10}, {"path", pathToUtf8(_images[0])}},
                      {{"image_id", 20}, {"path", pathToUtf8(_images[1])}}}},
                    {"tracks", Json::array()}};
        }
        std::filesystem::path _directory;
        std::vector<std::filesystem::path> _images;
    };

    TEST_F(StandardAerialEngineTest, ReadsCompactTracksWithoutLosingOriginalFeaturePrecision)
    {
        auto json = document();
        constexpr std::uint64_t original = 9007199254740993ULL;
        for (std::uint64_t index = 0; index < 2; ++index)
        {
            json["tracks"].push_back(
                {{"confidence", 0.8},
                 {"observations",
                  {{10, original + index, 12.0 + index, 24.0}, {20, original + index, 15.0 + index, 24.0}}},
                 {"direct_edges", {{0, 1}}}});
        }
        const auto path = _directory / "tracks.json";
        ASSERT_TRUE(writeJsonAtomic(path, json));
        engine::TiePointGraph graph;
        std::string error;
        ASSERT_TRUE(engine::readTiePointGraph(path, _images, &graph, &error)) << error;
        ASSERT_EQ(graph.tracks.size(), 2U);
        ASSERT_EQ(graph.keypointsByImage.at(0).size(), 2U);
        EXPECT_FLOAT_EQ(graph.keypointsByImage.at(0)[1].x, 13.0F);
        ASSERT_EQ(graph.matchPairs.size(), 1U);
        EXPECT_EQ(graph.matchPairs[0].matches.size(), 2U);
        EXPECT_EQ(graph.directEdgeCount, 2U);
        EXPECT_EQ(graph.synthesizedClosureEdgeCount, 0U);
    }

    TEST_F(StandardAerialEngineTest, SkipsMalformedObservationsAndEdgesWithoutThrowing)
    {
        auto json = document();
        json["tracks"] = {
            {{"observations",
              {nullptr, "invalid", {10, 3, 5.0, 6.0}, {20, 8, 9.0, 10.0}, {10, -1, 0, 0}, {20, 2, "x", 0}}},
             {"direct_edges", {nullptr, false, {0, 1}, {2.5, 3}, {2, 99}, {2, 3}}}}};
        const auto path = _directory / "tracks.json";
        ASSERT_TRUE(writeJsonAtomic(path, json));
        engine::TiePointGraph graph;
        std::string error;
        ASSERT_TRUE(engine::readTiePointGraph(path, _images, &graph, &error)) << error;
        EXPECT_EQ(graph.directEdgeCount, 1U);
        EXPECT_EQ(graph.tracks.size(), 1U);
    }

    TEST_F(StandardAerialEngineTest, ReportsMalformedAndMissingJsonWithFileContext)
    {
        const auto path = _directory / "tracks.json";
        engine::TiePointGraph graph;
        std::string error;
        EXPECT_FALSE(engine::readTiePointGraph(path, _images, &graph, &error));
        EXPECT_NE(error.find("连接点"), std::string::npos);
        EXPECT_NE(error.find(pathToUtf8(path)), std::string::npos);
        ASSERT_TRUE(writeFileAtomic(path, "{broken"));
        EXPECT_FALSE(engine::readTiePointGraph(path, _images, &graph, &error));
        EXPECT_NE(error.find("JSON"), std::string::npos);
        EXPECT_NE(error.find(pathToUtf8(path)), std::string::npos);
    }

    TEST_F(StandardAerialEngineTest, WritesExplicitLittleEndianPlyBytes)
    {
        const auto path = _directory / "cloud.ply";
        ASSERT_TRUE(xjw::aerial_triangulation::writeSparsePly(
            path,
            1,
            [](std::size_t) { return xjw::aerial_triangulation::SparsePlyVertex{{1.0F, -2.0F, 0.5F}, {255, 128, 3}}; },
            "test"));
        std::string bytes;
        ASSERT_TRUE(readFile(path, &bytes));
        const auto offset = bytes.find("end_header\n") + std::string("end_header\n").size();
        ASSERT_EQ(bytes.size() - offset, 15U);
        const unsigned char expected[15]{0, 0, 128, 63, 0, 0, 0, 192, 0, 0, 0, 63, 255, 128, 3};
        for (std::size_t index = 0; index < 15; ++index)
        {
            EXPECT_EQ(static_cast<unsigned char>(bytes[offset + index]), expected[index]);
        }
    }

    TEST_F(StandardAerialEngineTest, RejectsInvalidPinholeImagesAndUnpreparedPairs)
    {
        engine::PinholeInput input;
        input.graph = std::make_shared<engine::TiePointGraph>();
        EXPECT_FALSE(engine::runPinhole(input).success);
        input.images = {{0, _images[0], makePinholeCamera("first"), "sensor"},
                        {0, _images[1], makePinholeCamera("second"), "sensor"}};
        EXPECT_NE(engine::runPinhole(input).summary.find("ID"), std::string::npos);
        input.images[1].id = 1;
        auto graph = std::make_shared<engine::TiePointGraph>();
        graph->matchPairs.push_back({0, 99, {}});
        input.graph = graph;
        EXPECT_NE(engine::runPinhole(input).summary.find("未准备"), std::string::npos);
    }

    TEST_F(StandardAerialEngineTest, RejectsMissingPlaCameraInstance)
    {
        engine::PinholeInput input;
        input.graph = std::make_shared<engine::TiePointGraph>();
        input.images = {{0, _images[0], makePinholeCamera("first"), "sensor"}, {1, _images[1], nullptr, "sensor"}};

        const auto result = engine::runPinhole(input);
        EXPECT_FALSE(result.success);
        EXPECT_NE(result.summary.find("PlaCamera"), std::string::npos);
    }

    TEST_F(StandardAerialEngineTest, RejectsDuplicateCanonicalCameraIdentity)
    {
        engine::PinholeInput input;
        input.graph = std::make_shared<engine::TiePointGraph>();
        const auto camera = makePinholeCamera("same");
        input.images = {{0, _images[0], camera, "sensor"}, {1, _images[1], camera, "sensor"}};

        const auto result = engine::runPinhole(input);
        EXPECT_FALSE(result.success);
        EXPECT_NE(result.summary.find("身份重复"), std::string::npos);
    }

    TEST_F(StandardAerialEngineTest, CancelsBothNumericalEntrypointsWithoutQtCallbacks)
    {
        engine::PinholeInput pinhole;
        pinhole.graph = std::make_shared<engine::TiePointGraph>();
        pinhole.images = {{0, _images[0], makePinholeCamera("first"), "sensor"},
                          {1, _images[1], makePinholeCamera("second"), "sensor"}};
        pinhole.cancelFlag = std::make_shared<std::atomic<bool>>(true);
        EXPECT_EQ(engine::runPinhole(pinhole).summary, "用户取消");
        engine::RpcInput rpc;
        rpc.cancelFlag = pinhole.cancelFlag;
        EXPECT_EQ(engine::runRpc(rpc).error, "用户取消");
    }
} // namespace
