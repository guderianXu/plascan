#include "cli_common.h"
#include "cli_photogrammetry_common.h"
#include "CliJsonIO.h"
#include "CliTokenUtils.h"

#include "ModelImageQualityEvaluator.h"

#include <QCoreApplication>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QHash>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace
{

using xjw::cli::PhotogrammetryInputItem;

std::vector<int> allIndices(int count)
{
    std::vector<int> indices(static_cast<std::size_t>(count));
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

std::array<double, 3> centerOf(const PhotogrammetryInputItem &item)
{
    return item.camera.cameraCenter();
}

std::vector<int> angularValidationIndices(
    const std::vector<PhotogrammetryInputItem> &items)
{
    if (items.size() <= 5)
    {
        return allIndices(static_cast<int>(items.size()));
    }

    std::array<double, 3> mean{{0.0, 0.0, 0.0}};
    for (const PhotogrammetryInputItem &item : items)
    {
        const auto center = centerOf(item);
        for (int axis = 0; axis < 3; ++axis)
        {
            mean[static_cast<std::size_t>(axis)] += center[static_cast<std::size_t>(axis)];
        }
    }
    for (double &value : mean)
    {
        value /= static_cast<double>(items.size());
    }
    std::array<double, 3> variance{{0.0, 0.0, 0.0}};
    for (const PhotogrammetryInputItem &item : items)
    {
        const auto center = centerOf(item);
        for (int axis = 0; axis < 3; ++axis)
        {
            const double difference = center[static_cast<std::size_t>(axis)] -
                                      mean[static_cast<std::size_t>(axis)];
            variance[static_cast<std::size_t>(axis)] += difference * difference;
        }
    }
    std::array<int, 3> axes{{0, 1, 2}};
    std::sort(axes.begin(), axes.end(), [&](int left, int right)
    {
        return variance[static_cast<std::size_t>(left)] >
               variance[static_cast<std::size_t>(right)];
    });

    std::vector<std::pair<double, int>> angles;
    angles.reserve(items.size());
    for (std::size_t index = 0; index < items.size(); ++index)
    {
        const auto center = centerOf(items[index]);
        const double first = center[static_cast<std::size_t>(axes[0])] -
                             mean[static_cast<std::size_t>(axes[0])];
        const double second = center[static_cast<std::size_t>(axes[1])] -
                              mean[static_cast<std::size_t>(axes[1])];
        angles.emplace_back(std::atan2(second, first), static_cast<int>(index));
    }
    std::sort(angles.begin(), angles.end());
    std::vector<int> selected;
    for (std::size_t index = 0; index < angles.size(); index += 5)
    {
        selected.push_back(angles[index].second);
    }
    return selected;
}

std::vector<int> aerialGridValidationIndices(
    const std::vector<PhotogrammetryInputItem> &items)
{
    if (items.size() <= 2)
    {
        return allIndices(static_cast<int>(items.size()));
    }
    std::array<double, 3> mean{{0.0, 0.0, 0.0}};
    for (const PhotogrammetryInputItem &item : items)
    {
        const auto center = centerOf(item);
        for (int axis = 0; axis < 3; ++axis)
        {
            mean[static_cast<std::size_t>(axis)] += center[static_cast<std::size_t>(axis)];
        }
    }
    for (double &value : mean)
    {
        value /= static_cast<double>(items.size());
    }

    int nearest_index = 0;
    int farthest_index = 0;
    double nearest_distance = std::numeric_limits<double>::infinity();
    double farthest_distance = -1.0;
    for (std::size_t index = 0; index < items.size(); ++index)
    {
        const auto center = centerOf(items[index]);
        double squared_distance = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double difference = center[static_cast<std::size_t>(axis)] -
                                      mean[static_cast<std::size_t>(axis)];
            squared_distance += difference * difference;
        }
        if (squared_distance < nearest_distance)
        {
            nearest_distance = squared_distance;
            nearest_index = static_cast<int>(index);
        }
        if (squared_distance > farthest_distance)
        {
            farthest_distance = squared_distance;
            farthest_index = static_cast<int>(index);
        }
    }
    if (nearest_index == farthest_index)
    {
        return {nearest_index};
    }
    return {nearest_index, farthest_index};
}

std::vector<int> manifestValidationIndices(
    const QVector<xjw::qc::ModelValidationView> &views,
    xjw::qc::ModelSceneType scene_type,
    const QString &split)
{
    if (split == QStringLiteral("all") || views.size() <= 2)
    {
        return allIndices(views.size());
    }
    if (scene_type == xjw::qc::ModelSceneType::Dino ||
        split == QStringLiteral("angular-20"))
    {
        if (views.size() <= 5)
        {
            return allIndices(views.size());
        }
        std::array<double, 3> mean{{0.0, 0.0, 0.0}};
        for (const auto &view : views)
        {
            const std::array<double, 3> center = view.camera.cameraCenter();
            for (int axis = 0; axis < 3; ++axis)
            {
                mean[static_cast<std::size_t>(axis)] += center[static_cast<std::size_t>(axis)];
            }
        }
        for (double &value : mean)
        {
            value /= views.size();
        }
        std::array<double, 3> variance{{0.0, 0.0, 0.0}};
        for (const auto &view : views)
        {
            const std::array<double, 3> center = view.camera.cameraCenter();
            for (int axis = 0; axis < 3; ++axis)
            {
                const double difference = center[static_cast<std::size_t>(axis)] -
                                          mean[static_cast<std::size_t>(axis)];
                variance[static_cast<std::size_t>(axis)] += difference * difference;
            }
        }
        std::array<int, 3> axes{{0, 1, 2}};
        std::sort(axes.begin(), axes.end(), [&](int left, int right)
        {
            return variance[static_cast<std::size_t>(left)] >
                   variance[static_cast<std::size_t>(right)];
        });
        std::vector<std::pair<double, int>> angles;
        for (int index = 0; index < views.size(); ++index)
        {
            const std::array<double, 3> center = views[index].camera.cameraCenter();
            angles.emplace_back(
                std::atan2(center[static_cast<std::size_t>(axes[1])] -
                               mean[static_cast<std::size_t>(axes[1])],
                           center[static_cast<std::size_t>(axes[0])] -
                               mean[static_cast<std::size_t>(axes[0])]),
                index);
        }
        std::sort(angles.begin(), angles.end());
        std::vector<int> selected;
        for (std::size_t index = 0; index < angles.size(); index += 5)
        {
            selected.push_back(angles[index].second);
        }
        return selected;
    }

    std::array<double, 3> mean{{0.0, 0.0, 0.0}};
    for (const auto &view : views)
    {
        const std::array<double, 3> center = view.camera.cameraCenter();
        for (int axis = 0; axis < 3; ++axis)
        {
            mean[static_cast<std::size_t>(axis)] += center[static_cast<std::size_t>(axis)];
        }
    }
    for (double &value : mean)
    {
        value /= views.size();
    }
    int nearest_index = 0;
    int farthest_index = 0;
    double nearest_distance = std::numeric_limits<double>::infinity();
    double farthest_distance = -1.0;
    for (int index = 0; index < views.size(); ++index)
    {
        const std::array<double, 3> center = views[index].camera.cameraCenter();
        double distance = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double difference = center[static_cast<std::size_t>(axis)] -
                                      mean[static_cast<std::size_t>(axis)];
            distance += difference * difference;
        }
        if (distance < nearest_distance)
        {
            nearest_distance = distance;
            nearest_index = index;
        }
        if (distance > farthest_distance)
        {
            farthest_distance = distance;
            farthest_index = index;
        }
    }
    return nearest_index == farthest_index
        ? std::vector<int>{nearest_index}
        : std::vector<int>{nearest_index, farthest_index};
}

std::vector<int> validationIndices(const QString &split,
                                   xjw::qc::ModelSceneType scene_type,
                                   const std::vector<PhotogrammetryInputItem> &items)
{
    if (split == QStringLiteral("all"))
    {
        return allIndices(static_cast<int>(items.size()));
    }
    if (split == QStringLiteral("angular-20"))
    {
        return angularValidationIndices(items);
    }
    if (split == QStringLiteral("grid-center-edge"))
    {
        return aerialGridValidationIndices(items);
    }
    return scene_type == xjw::qc::ModelSceneType::Dino
        ? angularValidationIndices(items) : aerialGridValidationIndices(items);
}

QJsonObject resultJson(const xjw::qc::ModelImageQualityResult &result)
{
    QJsonObject object = result.summary;
    object[QStringLiteral("ok")] = result.ok;
    object[QStringLiteral("error")] = result.error;
    QJsonArray failures;
    for (const QString &failure : result.failureReasons)
    {
        failures.append(failure);
    }
    object[QStringLiteral("failure_reasons")] = failures;
    return object;
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication application(argc, argv);
    CLI::App app{"PlaScan 模型影像回归验收工具"};

    std::string mesh_path;
    std::string list_path;
    std::string mvs_workspace;
    std::string scene_type = "dino";
    std::string validation_split = "auto";
    std::string reference_cloud;
    std::string reference_camera_list;
    std::string output_directory;
    int maximum_render_dimension = 1600;
    bool align_reference_cloud = false;

    app.add_option("--mesh", mesh_path, "待验收 PLY 三角网格")->required();
    app.add_option("--image-camera-list", list_path, "影像与 TSAI 相机列表");
    app.add_option("--mvs-workspace", mvs_workspace,
                   "优先使用 MVS 清单中与模型同坐标系的相机模型");
    app.add_option("--scene-type", scene_type, "场景类型: dino 或 aerial");
    app.add_option("--validation-split", validation_split,
                   "留出策略: auto, angular-20, grid-center-edge 或 all");
    app.add_option("--reference-cloud", reference_cloud, "可选 Metashape 参考点云 PLY");
    app.add_option("--reference-camera-list", reference_camera_list,
                   "与参考点云同坐标系的影像/TSAI 列表，用相机中心估计 Sim3");
    app.add_option("--output-dir", output_directory, "验收报告输出目录")->required();
    app.add_option("--max-render-dim", maximum_render_dimension, "渲染最大边长")
        ->check(CLI::Range(128, 8192));
    app.add_flag("--align-reference-cloud", align_reference_cloud,
                 "使用最近邻平移 ICP 对齐参考点云");

    CLI11_PARSE(app, argc, argv);

    const QString scene = xjw::cli::normalizedToken(scene_type);
    if (scene != QStringLiteral("dino") && scene != QStringLiteral("aerial"))
    {
        cli::fatal("--scene-type 仅支持 dino 或 aerial", cli::EXIT_ARG_ERR);
    }
    const xjw::qc::ModelSceneType model_scene = scene == QStringLiteral("dino")
        ? xjw::qc::ModelSceneType::Dino : xjw::qc::ModelSceneType::Aerial;
    const QString split = xjw::cli::normalizedToken(validation_split);
    if (split != QStringLiteral("auto") && split != QStringLiteral("all") &&
        split != QStringLiteral("angular-20") &&
        split != QStringLiteral("grid-center-edge"))
    {
        cli::fatal("--validation-split 参数无效", cli::EXIT_ARG_ERR);
    }

    if (list_path.empty() == mvs_workspace.empty())
    {
        cli::fatal("--image-camera-list 与 --mvs-workspace 必须且只能指定一个",
                   cli::EXIT_ARG_ERR);
    }
    if (!reference_camera_list.empty() &&
        (reference_cloud.empty() || mvs_workspace.empty()))
    {
        cli::fatal("--reference-camera-list 需要同时指定 --reference-cloud 和 --mvs-workspace",
                   cli::EXIT_ARG_ERR);
    }

    xjw::qc::ModelImageQualityOptions options;
    options.meshPath = QString::fromUtf8(mesh_path);
    options.outputDirectory = QString::fromUtf8(output_directory);
    options.referenceCloudPath = QString::fromUtf8(reference_cloud);
    options.maximumRenderDimension = maximum_render_dimension;
    options.alignReferenceCloud = align_reference_cloud;
    options.sceneType = model_scene;
    QVector<xjw::qc::ModelValidationView> manifest_views;
    if (!mvs_workspace.empty())
    {
        QString manifest_error;
        manifest_views =
            xjw::qc::ModelImageQualityEvaluator::validationViewsFromMvsWorkspace(
                QString::fromUtf8(mvs_workspace), &manifest_error);
        if (!manifest_error.isEmpty())
        {
            cli::fatal(manifest_error.toStdString(), cli::EXIT_IO_ERR);
        }
        for (const int index : manifestValidationIndices(manifest_views, model_scene, split))
        {
            options.validationViews.append(manifest_views[index]);
        }
    }
    else
    {
        xjw::cli::PhotogrammetryListOptions list_options;
        list_options.allowImageOnlyRows = false;
        list_options.loadCameras = true;
        list_options.requireExistingImages = true;
        list_options.requireExistingCameras = true;
        std::vector<PhotogrammetryInputItem> items;
        QString input_error;
        if (!xjw::cli::readPhotogrammetryImageList(
                QString::fromUtf8(list_path), list_options, &items, &input_error))
        {
            cli::fatal(input_error.toStdString(), cli::EXIT_IO_ERR);
        }
        for (const int index : validationIndices(split, model_scene, items))
        {
            const PhotogrammetryInputItem &item = items[static_cast<std::size_t>(index)];
            xjw::qc::ModelValidationView view;
            view.id = QFileInfo(item.imagePath).completeBaseName();
            view.imagePath = item.imagePath;
            view.camera = item.camera;
            options.validationViews.append(view);
        }
    }
    if (options.validationViews.isEmpty())
    {
        cli::fatal("留出策略没有选择任何验收视角", cli::EXIT_IO_ERR);
    }

    if (!reference_camera_list.empty())
    {
        xjw::cli::PhotogrammetryListOptions reference_options;
        reference_options.allowImageOnlyRows = false;
        reference_options.loadCameras = true;
        reference_options.requireExistingImages = true;
        reference_options.requireExistingCameras = true;
        std::vector<PhotogrammetryInputItem> reference_items;
        QString reference_error;
        if (!xjw::cli::readPhotogrammetryImageList(
                QString::fromUtf8(reference_camera_list), reference_options,
                &reference_items, &reference_error))
        {
            cli::fatal(reference_error.toStdString(), cli::EXIT_IO_ERR);
        }
        QHash<QString, std::array<double, 3>> centers_by_name;
        for (const PhotogrammetryInputItem &item : reference_items)
        {
            centers_by_name.insert(QFileInfo(item.imagePath).fileName().toCaseFolded(),
                                   item.camera.cameraCenter());
        }
        std::vector<xjw::qc::Point3D> current_centers;
        std::vector<xjw::qc::Point3D> reference_centers;
        for (const xjw::qc::ModelValidationView &view : manifest_views)
        {
            const auto it = centers_by_name.constFind(
                QFileInfo(view.imagePath).fileName().toCaseFolded());
            if (it == centers_by_name.cend())
            {
                continue;
            }
            const std::array<double, 3> center = view.camera.cameraCenter();
            current_centers.push_back({center[0], center[1], center[2]});
            reference_centers.push_back({(*it)[0], (*it)[1], (*it)[2]});
        }
        if (current_centers.size() < 3)
        {
            cli::fatal("当前 MVS 清单与参考相机列表的同名相机少于 3 个，无法估计 Sim3",
                       cli::EXIT_IO_ERR);
        }
        const xjw::qc::PointCloudAlignmentResult alignment =
            xjw::qc::PointCloudAlignment::alignPairedSimilarity(
                current_centers, reference_centers);
        if (!alignment.success)
        {
            cli::fatal(alignment.error.toStdString(), cli::EXIT_IO_ERR);
        }
        options.hasReferenceTransform = true;
        options.referenceTransform = alignment.transform;
        options.cropReferenceToModelBounds = true;
    }

    const xjw::qc::ModelImageQualityResult result =
        xjw::qc::ModelImageQualityEvaluator().evaluate(options);
    xjw::cli::writeJson(result.ok ? stdout : stderr, resultJson(result));
    if (!result.error.isEmpty())
    {
        return cli::EXIT_IO_ERR;
    }
    return result.ok ? cli::EXIT_OK : cli::EXIT_ALGO_ERR;
}
