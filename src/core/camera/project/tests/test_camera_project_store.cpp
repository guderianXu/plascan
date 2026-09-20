#include "camera/project/CameraProjectStore.h"
#include "camera/project/CameraProjectRecords.h"
#include "camera/project/CameraProjectRuntime.h"
#include "camera/models/CameraModelFactories.h"
#include "camera/core/capabilities/CapabilityRequirements.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/models/frame_pinhole/FramePinholeInstance.h"
#include "camera/models/linescan/LineScanInstance.h"
#include "camera/models/rpc/RpcInstance.h"

#include <gtest/gtest.h>

namespace
{

    using namespace xjw::camera_project;
    namespace camera_core = xjw::camera_core;
    namespace camera_models = xjw::camera_models;

    QJsonObject image(const QString& id)
    {
        return QJsonObject{{QStringLiteral("image_uuid"), id},
                           {QStringLiteral("path"), QStringLiteral("images/") + id + QStringLiteral(".tif")}};
    }

    QJsonObject definition(const QString& id = QStringLiteral("def-1"))
    {
        return QJsonObject{{QStringLiteral("id"), id},
                           {QStringLiteral("model_type"), QStringLiteral("frame_pinhole")},
                           {QStringLiteral("schema_version"), 1},
                           {QStringLiteral("frame"), QStringLiteral("local")},
                           {QStringLiteral("parameters"),
                            QJsonObject{{QStringLiteral("intrinsics"),
                                         QJsonObject{{QStringLiteral("fx_px"), 100.0},
                                                     {QStringLiteral("fy_px"), 100.0},
                                                     {QStringLiteral("cx_px"), 50.0},
                                                     {QStringLiteral("cy_px"), 40.0},
                                                     {QStringLiteral("pixel_pitch_mm"), 0.01},
                                                     {QStringLiteral("u_axis_sign"), 1},
                                                     {QStringLiteral("v_axis_sign"), 1}}},
                                        {QStringLiteral("distortion"),
                                         QJsonObject{{QStringLiteral("k1"), 0.0},
                                                     {QStringLiteral("k2"), 0.0},
                                                     {QStringLiteral("k3"), 0.0},
                                                     {QStringLiteral("p1"), 0.0},
                                                     {QStringLiteral("p2"), 0.0}}},
                                        {QStringLiteral("pixel_convention"), QStringLiteral("center")},
                                        {QStringLiteral("depth_axis_flipped"), false}}}};
    }

    QJsonObject instance(const QString& imageId = QStringLiteral("image-1"),
                         const QString& definitionId = QStringLiteral("def-1"))
    {
        return QJsonObject{{QStringLiteral("id"), QStringLiteral("instance-") + imageId},
                           {QStringLiteral("image_uuid"), imageId},
                           {QStringLiteral("definition_id"), definitionId},
                           {QStringLiteral("schema_version"), 1},
                           {QStringLiteral("image_size"),
                            QJsonObject{{QStringLiteral("samples"), 1000}, {QStringLiteral("lines"), 800}}},
                           {QStringLiteral("pose"),
                            QJsonObject{{QStringLiteral("frame"), QStringLiteral("local")},
                                        {QStringLiteral("center_m"), QJsonArray{0.0, 0.0, 0.0}},
                                        {QStringLiteral("camera_to_world_rotation"),
                                         QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}}}};
    }

    QJsonObject validProjectFiles()
    {
        return QJsonObject{{QStringLiteral("images"), QJsonArray{image(QStringLiteral("image-1"))}},
                           {QStringLiteral("camera_definitions"), QJsonArray{definition()}},
                           {QStringLiteral("camera_instances"), QJsonArray{instance()}}};
    }

    QJsonObject frameMetadata(const QString& worldFrame = QStringLiteral("local"))
    {
        return QJsonObject{
            {QStringLiteral("model"), QStringLiteral("frame_pinhole")},
            {QStringLiteral("world_frame"), worldFrame},
            {QStringLiteral("intrinsics_unit"), QStringLiteral("mm")},
            {QStringLiteral("camera_center_unit"), QStringLiteral("m")},
            {QStringLiteral("pixel_convention"), QStringLiteral("center")},
            {QStringLiteral("pitch"), 0.01},
            {QStringLiteral("fu"), 20.0},
            {QStringLiteral("fv"), 20.0},
            {QStringLiteral("cu"), 5.0},
            {QStringLiteral("cv"), 4.0},
            {QStringLiteral("u_direction"), 1},
            {QStringLiteral("v_direction"), 1},
            {QStringLiteral("k1"), 0.0},
            {QStringLiteral("k2"), 0.0},
            {QStringLiteral("k3"), 0.0},
            {QStringLiteral("p1"), 0.0},
            {QStringLiteral("p2"), 0.0},
            {QStringLiteral("depth_axis_flipped"), false},
            {QStringLiteral("image_width"), 1000},
            {QStringLiteral("image_height"), 800},
            {QStringLiteral("C"), QJsonArray{0.0, 0.0, 1.0}},
            {QStringLiteral("R"), QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}};
    }

    TEST(CameraProjectStoreTest, LoadsAndSavesDefinitionAndInstanceCollections)
    {
        QJsonObject files = validProjectFiles();
        CameraProjectData data;
        QStringList errors;
        ASSERT_TRUE(CameraProjectStore::load(files, &data, &errors)) << errors.join('\n').toStdString();
        EXPECT_EQ(data.definitions.size(), 1);
        EXPECT_EQ(data.instances.size(), 1);

        QJsonObject saved;
        saved.insert(QStringLiteral("images"), files.value(QStringLiteral("images")));
        ASSERT_TRUE(CameraProjectStore::save(&saved, data, &errors));
        EXPECT_TRUE(saved.value(QStringLiteral("camera_definitions")).isArray());
        EXPECT_TRUE(saved.value(QStringLiteral("camera_instances")).isArray());
    }

    TEST(CameraProjectStoreTest, RejectsDuplicateDefinitionsDanglingInstancesAndDuplicateBindings)
    {
        QJsonObject files = validProjectFiles();
        files.insert(QStringLiteral("camera_definitions"), QJsonArray{definition(), definition()});
        files.insert(
            QStringLiteral("camera_instances"),
            QJsonArray{instance(), instance(), instance(QStringLiteral("image-2"), QStringLiteral("missing"))});
        CameraProjectData data;
        QStringList errors;
        EXPECT_FALSE(CameraProjectStore::load(files, &data, &errors));
        EXPECT_TRUE(errors.join('\n').contains(QStringLiteral("duplicate camera definition id")));
        EXPECT_TRUE(errors.join('\n').contains(QStringLiteral("more than one camera instance")));
        EXPECT_TRUE(errors.join('\n').contains(QStringLiteral("unknown image")));
    }

    TEST(CameraProjectStoreTest, RejectsLegacyImageCameraField)
    {
        QJsonObject files = validProjectFiles();
        QJsonObject imageWithLegacyCamera = image(QStringLiteral("image-1"));
        imageWithLegacyCamera.insert(QStringLiteral("camera"), QJsonObject{});
        files.insert(QStringLiteral("images"), QJsonArray{imageWithLegacyCamera});

        CameraProjectData data;
        QStringList errors;
        EXPECT_FALSE(CameraProjectStore::load(files, &data, &errors));
        EXPECT_TRUE(errors.join('\n').contains(QStringLiteral("images[0].camera is not supported")));
    }

    TEST(CameraProjectStoreTest, RejectsLegacyImageCameraFileField)
    {
        QJsonObject files = validProjectFiles();
        QJsonObject imageWithLegacyCameraFile = image(QStringLiteral("image-1"));
        imageWithLegacyCameraFile.insert(QStringLiteral("camera_file"), QStringLiteral("old.tsai"));
        files.insert(QStringLiteral("images"), QJsonArray{imageWithLegacyCameraFile});

        CameraProjectData data;
        QStringList errors;
        EXPECT_FALSE(CameraProjectStore::load(files, &data, &errors));
        EXPECT_TRUE(errors.join('\n').contains(QStringLiteral("images[0].camera_file is not supported")));
    }

    TEST(CameraProjectStoreTest, RejectsLegacyLineScanDefinitionSchema)
    {
        const QJsonObject legacyDefinition{{QStringLiteral("id"), QStringLiteral("legacy-line")},
                                           {QStringLiteral("model_type"), QStringLiteral("planetary_linescan")},
                                           {QStringLiteral("schema_version"), 1},
                                           {QStringLiteral("frame"), QStringLiteral("MOON_ME")},
                                           {QStringLiteral("parameters"), QJsonObject{}}};
        const QJsonObject files{{QStringLiteral("images"), QJsonArray{}},
                                {QStringLiteral("camera_definitions"), QJsonArray{legacyDefinition}},
                                {QStringLiteral("camera_instances"), QJsonArray{}}};
        CameraProjectData data;
        QStringList errors;
        EXPECT_FALSE(CameraProjectStore::load(files, &data, &errors));
        EXPECT_TRUE(errors.join('\n').contains(QStringLiteral("must be 2 for model planetary_linescan")));
    }

    TEST(CameraProjectStoreTest, SaveIsAtomicWhenValidationFails)
    {
        QJsonObject files = validProjectFiles();
        const QJsonObject before = files;
        CameraProjectData invalid;
        invalid.definitions = QJsonArray{definition()};
        invalid.instances = QJsonArray{instance(QStringLiteral("image-1"), QStringLiteral("missing"))};
        QStringList errors;
        EXPECT_FALSE(CameraProjectStore::save(&files, invalid, &errors));
        EXPECT_EQ(files, before);
    }

    TEST(CameraProjectStoreTest, UpsertsModelMetadataAsIndependentDefinitionAndInstance)
    {
        QJsonObject files{{QStringLiteral("images"),
                           QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("image-1")},
                                                  {QStringLiteral("path"), QStringLiteral("/tmp/image-1.tif")}}}},
                          {QStringLiteral("camera_definitions"), QJsonArray{}},
                          {QStringLiteral("camera_instances"), QJsonArray{}}};
        QJsonObject metadata = frameMetadata(QStringLiteral("project-world"));
        metadata.insert(QStringLiteral("image_width"), 10);
        metadata.insert(QStringLiteral("image_height"), 8);
        const auto update = CameraProjectRecords::upsertByImagePath(
            &files, QMap<QString, QJsonObject>{{QStringLiteral("/tmp/image-1.tif"), metadata}});
        ASSERT_TRUE(update.ok()) << update.errors.join(';').toStdString();
        ASSERT_EQ(files.value(QStringLiteral("camera_definitions")).toArray().size(), 1);
        ASSERT_EQ(files.value(QStringLiteral("camera_instances")).toArray().size(), 1);
        const QJsonObject stored = files.value(QStringLiteral("camera_instances")).toArray().at(0).toObject();
        EXPECT_EQ(stored.value(QStringLiteral("image_uuid")).toString(), QStringLiteral("image-1"));
        EXPECT_TRUE(stored.value(QStringLiteral("pose")).isObject());
        EXPECT_TRUE(stored.value(QStringLiteral("image_size")).isObject());
    }

    TEST(CameraProjectStoreTest, SolverUpdateUsesImageIdAndPreservesCanonicalInstanceIdentity)
    {
        QJsonObject files = validProjectFiles();
        QJsonObject metadata = frameMetadata();
        metadata.insert(QStringLiteral("fu"), 22.0);
        metadata.insert(QStringLiteral("fv"), 22.0);
        metadata.insert(QStringLiteral("C"), QJsonArray{1.0, 2.0, 3.0});
        const auto update = CameraProjectRecords::upsertByImageId(
            &files,
            CameraInstanceUpdates{{camera_core::ImageId("image-1"),
                                   camera_core::CameraInstanceId("instance-image-1"),
                                   xjw::coordinate_system::CoordinateFrameId("local"),
                                   metadata}});
        ASSERT_TRUE(update.ok()) << update.errors.join(';').toStdString();
        ASSERT_EQ(update.updatedCount, 1);

        const QJsonObject stored = files.value(QStringLiteral("camera_instances")).toArray().at(0).toObject();
        EXPECT_EQ(stored.value(QStringLiteral("id")).toString(), QStringLiteral("instance-image-1"));
        EXPECT_EQ(stored.value(QStringLiteral("image_uuid")).toString(), QStringLiteral("image-1"));
        EXPECT_EQ(stored.value(QStringLiteral("definition_id")).toString().isEmpty(), false);
    }

    TEST(CameraProjectStoreTest, SolverUpdateRejectsUnknownImageIdWithoutMutation)
    {
        QJsonObject files = validProjectFiles();
        const QJsonObject before = files;
        const auto update = CameraProjectRecords::upsertByImageId(
            &files,
            CameraInstanceUpdates{{camera_core::ImageId("missing-image"),
                                   camera_core::CameraInstanceId("missing-instance"),
                                   xjw::coordinate_system::CoordinateFrameId("local"),
                                   QJsonObject{{QStringLiteral("model"), QStringLiteral("frame_pinhole")},
                                               {QStringLiteral("world_frame"), QStringLiteral("local")}}}});
        EXPECT_FALSE(update.ok());
        EXPECT_TRUE(update.errors.join(';').contains(QStringLiteral("unknown ImageId")));
        EXPECT_EQ(files, before);
    }

    TEST(CameraProjectStoreTest, SolverUpdateRejectsStaleInstanceOrFrameBindingWithoutMutation)
    {
        QJsonObject metadata = frameMetadata();
        metadata.insert(QStringLiteral("fu"), 22.0);
        metadata.insert(QStringLiteral("fv"), 22.0);
        metadata.insert(QStringLiteral("C"), QJsonArray{1.0, 2.0, 3.0});
        for (const camera_core::CameraInstanceId& instanceId :
             {camera_core::CameraInstanceId("stale-instance"), camera_core::CameraInstanceId("instance-image-1")})
        {
            QJsonObject files = validProjectFiles();
            const QJsonObject before = files;
            const xjw::coordinate_system::CoordinateFrameId frame = instanceId.value() == "stale-instance"
                                                             ? xjw::coordinate_system::CoordinateFrameId("local")
                                                             : xjw::coordinate_system::CoordinateFrameId("wrong-frame");
            const auto update = CameraProjectRecords::upsertByImageId(
                &files, CameraInstanceUpdates{{camera_core::ImageId("image-1"), instanceId, frame, metadata}});
            EXPECT_FALSE(update.ok());
            EXPECT_EQ(files, before);
            EXPECT_FALSE(update.errors.isEmpty());
        }
    }

    TEST(CameraProjectStoreTest, SolverUpdateCannotChangeCanonicalModelFamily)
    {
        QJsonObject files = validProjectFiles();
        const QJsonObject before = files;
        const QJsonObject rpcMetadata{{QStringLiteral("model"), QStringLiteral("rpc00b")},
                                      {QStringLiteral("world_frame"), QStringLiteral("local")},
                                      {QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")}};
        const auto update = CameraProjectRecords::upsertByImageId(
            &files,
            CameraInstanceUpdates{{camera_core::ImageId("image-1"),
                                   camera_core::CameraInstanceId("instance-image-1"),
                                   xjw::coordinate_system::CoordinateFrameId("local"),
                                   rpcMetadata}});
        EXPECT_FALSE(update.ok());
        EXPECT_TRUE(update.errors.join(';').contains(QStringLiteral("changes model type")));
        EXPECT_EQ(files, before);
    }

    TEST(CameraProjectStoreTest, EncodesPinholeParametersAndRetainsInstanceMetadata)
    {
        QJsonObject files{{QStringLiteral("images"), QJsonArray{image(QStringLiteral("image-1"))}},
                          {QStringLiteral("camera_definitions"), QJsonArray{}},
                          {QStringLiteral("camera_instances"), QJsonArray{}}};
        QJsonObject metadata = frameMetadata();
        metadata.insert(QStringLiteral("fv"), 21.0);
        metadata.insert(QStringLiteral("image_width"), 10);
        metadata.insert(QStringLiteral("image_height"), 8);
        metadata.insert(QStringLiteral("aligned"), true);
        metadata.insert(QStringLiteral("solution"), QStringLiteral("current"));
        const CameraProjectUpdateResult update = CameraProjectRecords::upsertByImagePath(
            &files, QMap<QString, QJsonObject>{{QStringLiteral("images/image-1.tif"), metadata}});
        ASSERT_TRUE(update.ok()) << update.errors.join(';').toStdString();

        const QJsonObject storedDefinition =
            files.value(QStringLiteral("camera_definitions")).toArray().at(0).toObject();
        const QJsonObject parameters = storedDefinition.value(QStringLiteral("parameters")).toObject();
        EXPECT_TRUE(parameters.value(QStringLiteral("intrinsics")).isObject());
        EXPECT_TRUE(parameters.value(QStringLiteral("distortion")).isObject());
        EXPECT_FALSE(parameters.contains(QStringLiteral("fu")));
        EXPECT_FALSE(parameters.contains(QStringLiteral("aligned")));

        const QJsonObject storedInstance = files.value(QStringLiteral("camera_instances")).toArray().at(0).toObject();
        const QJsonObject state = storedInstance.value(QStringLiteral("state")).toObject();
        EXPECT_TRUE(state.value(QStringLiteral("metadata")).toObject().value(QStringLiteral("aligned")).toBool());
        EXPECT_EQ(state.value(QStringLiteral("metadata")).toObject().value(QStringLiteral("solution")).toString(),
                  QStringLiteral("current"));

        const QJsonObject restored = CameraProjectRecords::modelParametersForImage(
            files, files.value(QStringLiteral("images")).toArray().at(0).toObject());
        EXPECT_EQ(restored.value(QStringLiteral("model")).toString(), QStringLiteral("frame_pinhole"));
        EXPECT_EQ(restored.value(QStringLiteral("intrinsics_unit")).toString(), QStringLiteral("mm"));
        EXPECT_DOUBLE_EQ(restored.value(QStringLiteral("pitch")).toDouble(), 0.01);
        EXPECT_DOUBLE_EQ(restored.value(QStringLiteral("fu")).toDouble(), 20.0);
        EXPECT_FALSE(restored.contains(QStringLiteral("fx")));
        EXPECT_FALSE(restored.contains(QStringLiteral("pixel_pitch_mm")));
        EXPECT_TRUE(restored.value(QStringLiteral("aligned")).toBool());
        EXPECT_EQ(restored.value(QStringLiteral("solution")).toString(), QStringLiteral("current"));
    }

    TEST(CameraProjectStoreTest, RejectsUnknownModelInsteadOfSilentlyTreatingItAsPinhole)
    {
        QJsonObject files{{QStringLiteral("images"), QJsonArray{image(QStringLiteral("image-1"))}},
                          {QStringLiteral("camera_definitions"), QJsonArray{}},
                          {QStringLiteral("camera_instances"), QJsonArray{}}};
        const QJsonObject metadata{{QStringLiteral("model"), QStringLiteral("future_sensor_v2")},
                                   {QStringLiteral("image_width"), 10},
                                   {QStringLiteral("image_height"), 8}};
        const CameraProjectUpdateResult update = CameraProjectRecords::upsertByImagePath(
            &files, QMap<QString, QJsonObject>{{QStringLiteral("images/image-1.tif"), metadata}});
        EXPECT_FALSE(update.ok());
        EXPECT_TRUE(update.errors.join('\n').contains(QStringLiteral("unsupported camera model")));
        EXPECT_TRUE(files.value(QStringLiteral("camera_definitions")).toArray().isEmpty());
        EXPECT_TRUE(files.value(QStringLiteral("camera_instances")).toArray().isEmpty());
    }

    TEST(CameraProjectStoreTest, RejectsIncompleteModelMetadataBeforeMutation)
    {
        const auto expectRejected = [](const QJsonObject& metadata, const QString& expectedError)
        {
            QJsonObject files{{QStringLiteral("images"), QJsonArray{image(QStringLiteral("image-1"))}},
                              {QStringLiteral("camera_definitions"), QJsonArray{}},
                              {QStringLiteral("camera_instances"), QJsonArray{}}};
            const QJsonObject before = files;
            const CameraProjectUpdateResult update = CameraProjectRecords::upsertByImagePath(
                &files, QMap<QString, QJsonObject>{{QStringLiteral("images/image-1.tif"), metadata}});
            EXPECT_FALSE(update.ok());
            EXPECT_TRUE(update.errors.join('\n').contains(expectedError)) << update.errors.join(';').toStdString();
            EXPECT_EQ(files, before);
        };

        expectRejected(QJsonObject{{QStringLiteral("model"), QStringLiteral("tsai")}},
                       QStringLiteral("unsupported camera model"));

        QJsonObject incompleteFrame = frameMetadata();
        incompleteFrame.remove(QStringLiteral("fu"));
        expectRejected(incompleteFrame, QStringLiteral("camera model factory returned a null definition"));

        QJsonObject wrongUnitFrame = frameMetadata();
        wrongUnitFrame.insert(QStringLiteral("intrinsics_unit"), QStringLiteral("pixels"));
        expectRejected(wrongUnitFrame, QStringLiteral("intrinsics_unit must use the canonical value"));

        QJsonObject rpc{{QStringLiteral("model"), QStringLiteral("rpc00b")},
                        {QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                        {QStringLiteral("ground_crs"), QStringLiteral("EPSG:4979")},
                        {QStringLiteral("world_frame"), QStringLiteral("EPSG:4978")},
                        {QStringLiteral("height_datum"), QStringLiteral("WGS84_ellipsoidal")},
                        {QStringLiteral("pixel_convention"), QStringLiteral("opencv_zero_based_center")},
                        {QStringLiteral("line_offset"), 0.0},
                        {QStringLiteral("sample_offset"), 0.0},
                        {QStringLiteral("latitude_offset"), 0.0},
                        {QStringLiteral("longitude_offset"), 0.0},
                        {QStringLiteral("height_offset"), 0.0},
                        {QStringLiteral("line_scale"), 1.0},
                        {QStringLiteral("sample_scale"), 1.0},
                        {QStringLiteral("latitude_scale"), 1.0},
                        {QStringLiteral("longitude_scale"), 1.0},
                        {QStringLiteral("height_scale"), 1.0},
                        {QStringLiteral("image_samples"), 100},
                        {QStringLiteral("image_lines"), 80}};
        expectRejected(rpc, QStringLiteral("legacy or alternate camera field is not supported: line_offset"));

        expectRejected(QJsonObject{{QStringLiteral("model"), QStringLiteral("planetary_linescan")},
                                   {QStringLiteral("world_frame"), QStringLiteral("MOON_ME")},
                                   {QStringLiteral("focal_length_mm"), 10.0},
                                   {QStringLiteral("image_samples"), 100},
                                   {QStringLiteral("image_lines"), 80}},
                       QStringLiteral("legacy or alternate camera field is not supported: focal_length_mm"));
    }

    TEST(CameraProjectStoreTest, MissingCameraInstanceDoesNotProduceModelParameters)
    {
        const QJsonObject files = validProjectFiles();
        QJsonObject imageEntry = files.value(QStringLiteral("images")).toArray().at(0).toObject();
        imageEntry.insert(QStringLiteral("image_uuid"), QStringLiteral("unbound-image"));
        EXPECT_TRUE(CameraProjectRecords::modelParametersForImage(files, imageEntry).isEmpty());
    }

    TEST(CameraProjectRuntimeTest, LoadsCanonicalInstancesAndReportsModelCapabilities)
    {
        QJsonArray rpcNumerator;
        QJsonArray rpcDenominator;
        for (int index = 0; index < 20; ++index)
        {
            rpcNumerator.append(index == 0 ? 1.0 : 0.0);
            rpcDenominator.append(index == 0 ? 1.0 : 0.0);
        }
        QJsonObject rpcDefinition{{QStringLiteral("id"), QStringLiteral("rpc-definition")},
                                  {QStringLiteral("model_type"), QStringLiteral("rpc00b")},
                                  {QStringLiteral("schema_version"), 1},
                                  {QStringLiteral("frame"), QStringLiteral("wgs84-geodetic")},
                                  {QStringLiteral("parameters"),
                                   QJsonObject{{QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                                               {QStringLiteral("line_offset"), 0.0},
                                               {QStringLiteral("sample_offset"), 0.0},
                                               {QStringLiteral("latitude_offset"), 0.0},
                                               {QStringLiteral("longitude_offset"), 0.0},
                                               {QStringLiteral("height_offset"), 0.0},
                                               {QStringLiteral("line_scale"), 1.0},
                                               {QStringLiteral("sample_scale"), 1.0},
                                               {QStringLiteral("latitude_scale"), 1.0},
                                               {QStringLiteral("longitude_scale"), 1.0},
                                               {QStringLiteral("height_scale"), 1.0},
                                               {QStringLiteral("line_numerator"), rpcNumerator},
                                               {QStringLiteral("line_denominator"), rpcDenominator},
                                               {QStringLiteral("sample_numerator"), rpcNumerator},
                                               {QStringLiteral("sample_denominator"), rpcDenominator}}}};
        QJsonObject rpcImage = image(QStringLiteral("rpc-image"));
        rpcImage.insert(QStringLiteral("path"), QStringLiteral("images/rpc-image.tif"));
        QJsonObject rpcInstance{{QStringLiteral("id"), QStringLiteral("rpc-instance")},
                                {QStringLiteral("image_uuid"), QStringLiteral("rpc-image")},
                                {QStringLiteral("definition_id"), QStringLiteral("rpc-definition")},
                                {QStringLiteral("schema_version"), 1},
                                {QStringLiteral("image_size"),
                                 QJsonObject{{QStringLiteral("samples"), 100}, {QStringLiteral("lines"), 80}}}};

        QJsonObject files{{QStringLiteral("images"), QJsonArray{image(QStringLiteral("image-1")), rpcImage}},
                          {QStringLiteral("camera_definitions"), QJsonArray{definition(), rpcDefinition}},
                          {QStringLiteral("camera_instances"), QJsonArray{instance(), rpcInstance}}};

        const camera_core::CameraModelRegistry registry = camera_models::makeBuiltinCameraModelRegistry();
        const CameraProjectRuntimeResult loaded = CameraProjectRuntime::load(files, registry);
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        ASSERT_EQ(loaded.instances.size(), 2U);

        const auto pinhole = loaded.instances.forImage(camera_core::ImageId("image-1"));
        ASSERT_TRUE(pinhole.ok()) << pinhole.error;
        EXPECT_NE(dynamic_cast<const camera_models::frame_pinhole::FramePinholeInstance*>(pinhole.instance.get()),
                  nullptr);
        EXPECT_TRUE(
            camera_core::requireCapabilities(*pinhole.instance,
                                             camera_core::CapabilitySet{camera_core::CapabilityKind::Projection,
                                                                        camera_core::CapabilityKind::StaticPose})
                .ok());

        const auto rpc = loaded.instances.forImage(camera_core::ImageId("rpc-image"));
        ASSERT_TRUE(rpc.ok()) << rpc.error;
        EXPECT_NE(dynamic_cast<const camera_models::rpc::RpcInstance*>(rpc.instance.get()), nullptr);
        const auto capabilities =
            camera_core::requireCapabilities(*rpc.instance,
                                             camera_core::CapabilitySet{camera_core::CapabilityKind::Projection,
                                                                        camera_core::CapabilityKind::StaticPose});
        ASSERT_FALSE(capabilities.ok());
        EXPECT_EQ(capabilities.missing().size(), 1U);
        EXPECT_EQ(capabilities.missing().front(), camera_core::CapabilityKind::StaticPose);

        const auto staticPlan =
            loaded.planOperationForImages({camera_core::ImageId("image-1")}, camera_core::CameraOperation::DenseMvs);
        EXPECT_TRUE(staticPlan.ok()) << staticPlan.failureMessage();
        const auto rpcPlan =
            loaded.planOperationForImages({camera_core::ImageId("rpc-image")}, camera_core::CameraOperation::DenseMvs);
        EXPECT_FALSE(rpcPlan.ok());
        EXPECT_NE(rpcPlan.failureMessage().find("static_pose"), std::string::npos);
    }

    TEST(CameraProjectRuntimeTest, RejectsMalformedCanonicalStateWithoutPartialInstances)
    {
        QJsonObject files = validProjectFiles();
        QJsonObject brokenInstance = instance();
        brokenInstance[QStringLiteral("pose")] = QJsonObject{
            {QStringLiteral("frame"), QStringLiteral("local")},
            {QStringLiteral("center_m"), QJsonArray{0.0, 0.0}},
            {QStringLiteral("camera_to_world_rotation"), QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}};
        files[QStringLiteral("camera_instances")] = QJsonArray{brokenInstance};

        const camera_core::CameraModelRegistry registry = camera_models::makeBuiltinCameraModelRegistry();
        const CameraProjectRuntimeResult loaded = CameraProjectRuntime::load(files, registry);
        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());
        EXPECT_TRUE(loaded.errors.join('\n').contains(QStringLiteral("camera_instances[0]")));
    }

    TEST(CameraProjectRuntimeTest, ResolvesBoundPinholeNumericStateByImageId)
    {
        const CameraProjectRuntimeResult loaded =
            CameraProjectRuntime::load(validProjectFiles(), camera_models::makeBuiltinCameraModelRegistry());
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();

        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
        std::string error;
        ASSERT_TRUE(loaded.framePinholeStateForImage(camera_core::ImageId("image-1"), &state, &error)) << error;
        EXPECT_TRUE(error.empty());
        EXPECT_TRUE(state.hasBoundIdentity());
        EXPECT_EQ(state.instanceId().value(), "instance-image-1");
        EXPECT_EQ(state.imageId().value(), "image-1");
        EXPECT_EQ(state.worldFrame().value(), "local");
        EXPECT_TRUE(state.validateNumericalState(&error)) << error;

        std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> states;
        ASSERT_TRUE(loaded.framePinholeStatesForImages({camera_core::ImageId("image-1")}, &states, &error)) << error;
        ASSERT_EQ(states.size(), 1U);
        EXPECT_EQ(states.front().imageId().value(), "image-1");
        EXPECT_FALSE(loaded.framePinholeStatesForImages(
            {camera_core::ImageId("image-1"), camera_core::ImageId("image-1")}, &states, &error));
        EXPECT_TRUE(states.empty());
        EXPECT_NE(error.find("duplicate"), std::string::npos);

        state.setIntrinsics(900.0, 900.0, 512.0, 384.0);
        EXPECT_FALSE(loaded.framePinholeStateForImage(camera_core::ImageId("missing-image"), &state, &error));
        EXPECT_FALSE(state.isValid());
        EXPECT_FALSE(state.hasBoundIdentity());
    }

    TEST(CameraProjectRuntimeTest, RejectsRpcWhenResolvingPinholeNumericState)
    {
        QJsonArray rpcNumerator;
        QJsonArray rpcDenominator;
        for (int index = 0; index < 20; ++index)
        {
            rpcNumerator.append(index == 0 ? 1.0 : 0.0);
            rpcDenominator.append(index == 0 ? 1.0 : 0.0);
        }
        const QJsonObject rpcDefinition{{QStringLiteral("id"), QStringLiteral("rpc-definition")},
                                        {QStringLiteral("model_type"), QStringLiteral("rpc00b")},
                                        {QStringLiteral("schema_version"), 1},
                                        {QStringLiteral("frame"), QStringLiteral("wgs84-geodetic")},
                                        {QStringLiteral("parameters"),
                                         QJsonObject{{QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                                                     {QStringLiteral("line_offset"), 0.0},
                                                     {QStringLiteral("sample_offset"), 0.0},
                                                     {QStringLiteral("latitude_offset"), 0.0},
                                                     {QStringLiteral("longitude_offset"), 0.0},
                                                     {QStringLiteral("height_offset"), 0.0},
                                                     {QStringLiteral("line_scale"), 1.0},
                                                     {QStringLiteral("sample_scale"), 1.0},
                                                     {QStringLiteral("latitude_scale"), 1.0},
                                                     {QStringLiteral("longitude_scale"), 1.0},
                                                     {QStringLiteral("height_scale"), 1.0},
                                                     {QStringLiteral("line_numerator"), rpcNumerator},
                                                     {QStringLiteral("line_denominator"), rpcDenominator},
                                                     {QStringLiteral("sample_numerator"), rpcNumerator},
                                                     {QStringLiteral("sample_denominator"), rpcDenominator}}}};
        const QJsonObject rpcImage = QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("rpc-image")},
                                                 {QStringLiteral("path"), QStringLiteral("images/rpc-image.tif")}};
        const QJsonObject rpcInstance{{QStringLiteral("id"), QStringLiteral("rpc-instance")},
                                      {QStringLiteral("image_uuid"), QStringLiteral("rpc-image")},
                                      {QStringLiteral("definition_id"), QStringLiteral("rpc-definition")},
                                      {QStringLiteral("schema_version"), 1},
                                      {QStringLiteral("image_size"),
                                       QJsonObject{{QStringLiteral("samples"), 100}, {QStringLiteral("lines"), 80}}}};
        const QJsonObject files{{QStringLiteral("images"), QJsonArray{rpcImage}},
                                {QStringLiteral("camera_definitions"), QJsonArray{rpcDefinition}},
                                {QStringLiteral("camera_instances"), QJsonArray{rpcInstance}}};

        const CameraProjectRuntimeResult loaded =
            CameraProjectRuntime::load(files, camera_models::makeBuiltinCameraModelRegistry());
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();

        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
        std::string error;
        EXPECT_FALSE(loaded.framePinholeStateForImage(camera_core::ImageId("rpc-image"), &state, &error));
        EXPECT_TRUE(QString::fromStdString(error).contains(QStringLiteral("frame-pinhole"))) << error;
        EXPECT_TRUE(QString::fromStdString(error).contains(QStringLiteral("rpc"))) << error;
    }

    TEST(CameraProjectRuntimeTest, PreservesExactFrameComposedLineScanState)
    {
        const QString imagePath = QStringLiteral("images/line-image.tif");
        QJsonObject files{{QStringLiteral("images"),
                           QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("line-image")},
                                                  {QStringLiteral("path"), imagePath},
                                                  {QStringLiteral("samples"), 1024},
                                                  {QStringLiteral("lines"), 256}}}},
                          {QStringLiteral("camera_definitions"), QJsonArray{}},
                          {QStringLiteral("camera_instances"), QJsonArray{}}};
        const QJsonArray identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        const QJsonArray quaternionSamples{
            QJsonObject{{QStringLiteral("time_seconds"), 10.0},
                        {QStringLiteral("quaternion_scalar_first"), QJsonArray{1.0, 0.0, 0.0, 0.0}}},
            QJsonObject{{QStringLiteral("time_seconds"), 20.0},
                        {QStringLiteral("quaternion_scalar_first"), QJsonArray{1.0, 0.0, 0.0, 0.0}}}};
        const QJsonObject rotationTrajectory{{QStringLiteral("constant_rotation"), identity},
                                             {QStringLiteral("samples"), quaternionSamples}};
        const QJsonObject trajectory{
            {QStringLiteral("representation"), QStringLiteral("frame_composed")},
            {QStringLiteral("time_scale"), QStringLiteral("tdb")},
            {QStringLiteral("inertial_states"),
             QJsonArray{QJsonObject{{QStringLiteral("time_seconds"), 10.0},
                                    {QStringLiteral("position_m"), QJsonArray{0.0, 0.0, 1.0}},
                                    {QStringLiteral("velocity_m_per_s"), QJsonArray{0.0, 0.0, 0.1}}},
                        QJsonObject{{QStringLiteral("time_seconds"), 20.0},
                                    {QStringLiteral("position_m"), QJsonArray{0.0, 0.0, 2.0}},
                                    {QStringLiteral("velocity_m_per_s"), QJsonArray{0.0, 0.0, 0.1}}}}},
            {QStringLiteral("inertial_to_world"), rotationTrajectory},
            {QStringLiteral("inertial_to_sensor"), rotationTrajectory}};
        const QJsonObject metadata{
            {QStringLiteral("model"), QStringLiteral("planetary_linescan")},
            {QStringLiteral("world_frame"), QStringLiteral("MOON_ME")},
            {QStringLiteral("image_samples"), 1024},
            {QStringLiteral("image_lines"), 256},
            {QStringLiteral("optics"),
             QJsonObject{{QStringLiteral("focal_length_mm"), 10.0},
                         {QStringLiteral("distortion_model"), QStringLiteral("lro_nac_focal_plane")},
                         {QStringLiteral("distortion_k1"), 0.001},
                         {QStringLiteral("sample_geometry"),
                          QJsonObject{{QStringLiteral("type"), QStringLiteral("detector_affine")},
                                      {QStringLiteral("detector_sample_summing"), 1.0},
                                      {QStringLiteral("detector_line_summing"), 1.0},
                                      {QStringLiteral("detector_sample_origin"), 512.0},
                                      {QStringLiteral("detector_line_origin"), 0.0},
                                      {QStringLiteral("starting_detector_sample"), 0.0},
                                      {QStringLiteral("starting_detector_line"), 0.0},
                                      {QStringLiteral("focal_to_pixel_samples"), QJsonArray{0.0, 0.0, 1.0}},
                                      {QStringLiteral("focal_to_pixel_lines"), QJsonArray{0.0, 1.0, 0.0}}}}}},
            {QStringLiteral("pixel_convention"), QStringLiteral("pixel_center")},
            {QStringLiteral("trajectory"), trajectory},
            {QStringLiteral("line_timing"),
             QJsonObject{{QStringLiteral("time_scale"), QStringLiteral("tdb")},
                         {QStringLiteral("segments"),
                          QJsonArray{QJsonObject{{QStringLiteral("start_line"), 0.5},
                                                 {QStringLiteral("start_time_seconds"), 10.0},
                                                 {QStringLiteral("seconds_per_line"), 0.01}},
                                     QJsonObject{{QStringLiteral("start_line"), 128.5},
                                                 {QStringLiteral("start_time_seconds"), 11.28},
                                                 {QStringLiteral("seconds_per_line"), 0.02}}}}}}};

        const CameraProjectUpdateResult update =
            CameraProjectRecords::upsertByImagePath(&files, QMap<QString, QJsonObject>{{imagePath, metadata}});
        ASSERT_TRUE(update.ok()) << update.errors.join('\n').toStdString();
        const QJsonObject storedDefinition =
            files.value(QStringLiteral("camera_definitions")).toArray().at(0).toObject();
        EXPECT_EQ(storedDefinition.value(QStringLiteral("schema_version")).toInt(), 2);
        const QJsonObject stored = files.value(QStringLiteral("camera_instances")).toArray().at(0).toObject();
        EXPECT_TRUE(stored.value(QStringLiteral("state")).toObject().value(QStringLiteral("trajectory")).isObject());
        EXPECT_TRUE(stored.value(QStringLiteral("state")).toObject().value(QStringLiteral("line_timing")).isObject());

        const CameraProjectRuntimeResult loaded =
            CameraProjectRuntime::load(files, camera_models::makeBuiltinCameraModelRegistry());
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        const auto lookup = loaded.instances.forImage(camera_core::ImageId("line-image"));
        ASSERT_TRUE(lookup.ok()) << lookup.error;
        const auto* lineInstance =
            dynamic_cast<const camera_models::linescan::LineScanInstance*>(lookup.instance.get());
        ASSERT_NE(lineInstance, nullptr);
        ASSERT_NE(lineInstance->trajectory().frameComposed(), nullptr);
        EXPECT_EQ(lineInstance->trajectory().frameComposed()->inertialStates.size(), 2U);
        EXPECT_EQ(lineInstance->lineTiming().segments.size(), 2U);
        EXPECT_TRUE(lineInstance->lineScanDefinition().optics().detectorGeometry.has_value());
    }

} // namespace
