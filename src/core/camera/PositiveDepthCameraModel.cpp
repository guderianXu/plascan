#include "PositiveDepthCameraModel.h"

#include "Camera.h"

#include <cmath>

namespace xjw
{
namespace
{

void applyPixelTransform(const std::array<float, 9> &H,
                         float u,
                         float v,
                         float &outU,
                         float &outV)
{
    const float w = H[6] * u + H[7] * v + H[8];
    if (std::fabs(w) < 1e-12f)
    {
        outU = u;
        outV = v;
        return;
    }

    outU = (H[0] * u + H[1] * v + H[2]) / w;
    outV = (H[3] * u + H[4] * v + H[5]) / w;
}

} // namespace

PositiveDepthCameraModel::PositiveDepthCameraModel()
    : fx(0.0f)
    , cx(0.0f)
    , fy(0.0f)
    , cy(0.0f)
    , R_cw{1.0f, 0.0f, 0.0f,
           0.0f, 1.0f, 0.0f,
           0.0f, 0.0f, 1.0f}
    , T{0.0f, 0.0f, 0.0f}
    , C{0.0f, 0.0f, 0.0f}
{
}

PositiveDepthCameraModel::PositiveDepthCameraModel(const Camera &camera)
    : PositiveDepthCameraModel()
{
    focalX = static_cast<float>(std::fabs(camera.focalX()));
    focalY = static_cast<float>(std::fabs(camera.focalY()));
    principalX = static_cast<float>(camera.principalX());
    principalY = static_cast<float>(camera.principalY());

    const auto center = camera.cameraCenter();
    for (int index = 0; index < 3; ++index)
    {
        cameraCenter[index] = static_cast<float>(center[index]);
    }

    const float zSign = camera.depthAxisFlipped() ? -1.0f : 1.0f;
    const float xSign = zSign * static_cast<float>(camera.uAxisSign() > 0 ? 1 : -1);
    const float ySign = zSign * static_cast<float>(camera.vAxisSign() > 0 ? 1 : -1);
    const float axisSign[3] = {xSign, ySign, zSign};

    const auto rotationCameraToWorld = camera.cameraToWorldRotation();
    float rotationWorldToCameraAsp[9];
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            rotationWorldToCameraAsp[row * 3 + col] = static_cast<float>(rotationCameraToWorld[col * 3 + row]);
        }
    }

    float translationWorldToCameraAsp[3] = {0.0f, 0.0f, 0.0f};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            translationWorldToCameraAsp[row] -= rotationWorldToCameraAsp[row * 3 + col]
                                              * static_cast<float>(center[col]);
        }
    }

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            rotationWorldToCamera[row * 3 + col] = axisSign[row] * rotationWorldToCameraAsp[row * 3 + col];
        }
        translationWorldToCamera[row] = axisSign[row] * translationWorldToCameraAsp[row];
    }
}

bool PositiveDepthCameraModel::valid() const
{
    return focalX > 0.0f && focalY > 0.0f;
}

bool PositiveDepthCameraModel::project(float worldX, float worldY, float worldZ,
                                       float &pixelX, float &pixelY) const
{
    float depth = 0.0f;
    return projectWithDepth(worldX, worldY, worldZ, pixelX, pixelY, depth);
}

bool PositiveDepthCameraModel::projectWithDepth(float worldX, float worldY, float worldZ,
                                                float &pixelX, float &pixelY, float &depth) const
{
    const float cameraX = rotationWorldToCamera[0] * worldX
                        + rotationWorldToCamera[1] * worldY
                        + rotationWorldToCamera[2] * worldZ
                        + translationWorldToCamera[0];
    const float cameraY = rotationWorldToCamera[3] * worldX
                        + rotationWorldToCamera[4] * worldY
                        + rotationWorldToCamera[5] * worldZ
                        + translationWorldToCamera[1];
    const float cameraZ = rotationWorldToCamera[6] * worldX
                        + rotationWorldToCamera[7] * worldY
                        + rotationWorldToCamera[8] * worldZ
                        + translationWorldToCamera[2];
    depth = cameraZ;
    if (cameraZ < 1e-6f)
    {
        return false;
    }

    pixelX = focalX * cameraX / cameraZ + principalX;
    pixelY = focalY * cameraY / cameraZ + principalY;

    if (m_hasPixelTransform)
    {
        applyPixelTransform(m_pixelTransform, pixelX, pixelY, pixelX, pixelY);
    }

    return true;
}

void PositiveDepthCameraModel::unproject(float pixelX, float pixelY, float depth,
                                         float &worldX, float &worldY, float &worldZ) const
{
    if (m_hasPixelTransform)
    {
        applyPixelTransform(m_pixelTransformInv, pixelX, pixelY, pixelX, pixelY);
    }

    const float cameraX = (pixelX - principalX) / focalX * depth;
    const float cameraY = (pixelY - principalY) / focalY * depth;
    const float cameraZ = depth;

    worldX = rotationWorldToCamera[0] * cameraX
           + rotationWorldToCamera[3] * cameraY
           + rotationWorldToCamera[6] * cameraZ
           + cameraCenter[0];
    worldY = rotationWorldToCamera[1] * cameraX
           + rotationWorldToCamera[4] * cameraY
           + rotationWorldToCamera[7] * cameraZ
           + cameraCenter[1];
    worldZ = rotationWorldToCamera[2] * cameraX
           + rotationWorldToCamera[5] * cameraY
           + rotationWorldToCamera[8] * cameraZ
           + cameraCenter[2];
}

void PositiveDepthCameraModel::setPixelTransform(const std::array<double, 9> &transform,
                                                 const std::array<double, 9> &inverseTransform)
{
    m_hasPixelTransform = true;
    for (int i = 0; i < 9; ++i)
    {
        m_pixelTransform[i] = static_cast<float>(transform[i]);
        m_pixelTransformInv[i] = static_cast<float>(inverseTransform[i]);
    }
}

} // namespace xjw
