#pragma once

#include <array>

namespace xjw
{

class Camera;

class PositiveDepthCameraModel
{
public:
    union
    {
        struct
        {
            float focalX;
            float principalX;
            float focalY;
            float principalY;
        };
        struct
        {
            float fx;
            float cx;
            float fy;
            float cy;
        };
    };

    union
    {
        float rotationWorldToCamera[9];
        float R_cw[9];
    };

    union
    {
        float translationWorldToCamera[3];
        float T[3];
    };

    union
    {
        float cameraCenter[3];
        float C[3];
    };

    PositiveDepthCameraModel();
    explicit PositiveDepthCameraModel(const Camera &camera);

    bool valid() const;
    bool project(float worldX, float worldY, float worldZ, float &pixelX, float &pixelY) const;
    void unproject(float pixelX, float pixelY, float depth,
                   float &worldX, float &worldY, float &worldZ) const;
    void setPixelTransform(const std::array<double, 9> &transform,
                           const std::array<double, 9> &inverseTransform);

private:
    bool m_hasPixelTransform = false;
    std::array<float, 9> m_pixelTransform = {1.0f, 0.0f, 0.0f,
                                             0.0f, 1.0f, 0.0f,
                                             0.0f, 0.0f, 1.0f};
    std::array<float, 9> m_pixelTransformInv = {1.0f, 0.0f, 0.0f,
                                                0.0f, 1.0f, 0.0f,
                                                0.0f, 0.0f, 1.0f};
};

} // namespace xjw
