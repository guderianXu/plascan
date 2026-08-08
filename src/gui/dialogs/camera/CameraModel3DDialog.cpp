#include "camera/CameraModel3DDialog.h"

#include "ProjectCameraIO.h"
#include "ProjectManager.h"
#include "project/ProjectMetadata.h"
#include "ui_CameraModel3DDialog.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>

#include <array>

CameraModel3DDialog::CameraModel3DDialog(ProjectManager *projectManager,
                                         QWidget *parent)
    : QDialog(parent)
    , _projectManager(projectManager)
{
    Ui::CameraModel3DDialog form;
    form.setupUi(this);

    _scene = form.m_scene;
    _summaryLabel = form.m_summaryLabel;

    connect(form.reloadButton,
            &QPushButton::clicked,
            this,
            &CameraModel3DDialog::reloadFromProject);
    connect(form.closeButton, &QPushButton::clicked, this, &QDialog::accept);
    reloadFromProject();
}

QVector<CameraSceneWidget::CameraPose> CameraModel3DDialog::readCamerasFromMeta() const
{
    QVector<CameraSceneWidget::CameraPose> poses;
    if (!_projectManager)
    {
        return poses;
    }

    const QJsonArray images =
        xjw::common::project::projectImageEntries(_projectManager->currentMeta());
    for (const QJsonValue &imageValue : images)
    {
        const QJsonObject imageObject = imageValue.toObject();
        xjw::Camera camera;
        if (!xjw::common::project::imageCameraFromEntry(imageObject, &camera))
        {
            continue;
        }

        const std::array<double, 3> cameraCenter = camera.cameraCenter();
        const std::array<double, 9> cameraToWorldRotation = camera.cameraToWorldRotation();
        const xjw::Camera::Intrinsics intrinsics = camera.intrinsics();
        const QJsonObject cameraObject =
            imageObject.value(QStringLiteral("camera")).toObject();

        CameraSceneWidget::CameraPose pose;
        pose.name = imageObject.value(QStringLiteral("name")).toString();
        pose.imagePath = imageObject.value(QStringLiteral("path")).toString();
        if (pose.imagePath.isEmpty())
        {
            pose.imagePath = imageObject.value(QStringLiteral("image_path")).toString();
        }
        if (pose.name.isEmpty())
        {
            pose.name = pose.imagePath;
        }
        pose.center = QVector3D(float(cameraCenter[0]),
                                float(cameraCenter[1]),
                                float(cameraCenter[2]));
        pose.focalX = static_cast<float>(intrinsics.focalX);
        pose.focalY = static_cast<float>(intrinsics.focalY);
        pose.principalX = static_cast<float>(intrinsics.principalX);
        pose.principalY = static_cast<float>(intrinsics.principalY);
        pose.imageWidth = cameraObject.value(QStringLiteral("image_width")).toInt();
        pose.imageHeight = cameraObject.value(QStringLiteral("image_height")).toInt();
        pose.uAxisSign = intrinsics.uAxisSign;
        pose.vAxisSign = intrinsics.vAxisSign;
        pose.depthAxisFlipped = camera.depthAxisFlipped();

        QMatrix3x3 rotation;
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                rotation(row, column) =
                    float(cameraToWorldRotation[row * 3 + column]);
            }
        }
        pose.rotation = rotation;
        poses.push_back(pose);
    }
    return poses;
}

void CameraModel3DDialog::reloadFromProject()
{
    const QVector<CameraSceneWidget::CameraPose> poses = readCamerasFromMeta();
    _scene->setCameraPoses(poses);
    const QString labelHint = poses.size() > 40
        ? tr("，相机名称已抽样显示")
        : QString();
    _summaryLabel->setText(
        tr("相机数量: %1%2（左键旋转，滚轮缩放）")
            .arg(poses.size())
            .arg(labelHint));
}
