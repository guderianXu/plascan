#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "Camera.h"
#include "Intersection.h"

using namespace xjw;

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 7)
    {
        std::cout << "用法:\n"
                  << "  " << argv[0] << " <cam1.tsai> <cam2.tsai>\n"
                  << "  " << argv[0] << " <cam1.tsai> <cam2.tsai> <u1> <v1> <u2> <v2>\n"
                  << "说明: 仅传入两个相机时, 默认使用两相机主点(cu,cv)作为同名点。\n";
        return 1;
    }

    Camera cam1, cam2;
    if (!cam1.loadFromFile(argv[1]))
    {
        std::cerr << "加载失败: " << argv[1] << "\n";
        return 2;
    }
    if (!cam2.loadFromFile(argv[2]))
    {
        std::cerr << "加载失败: " << argv[2] << "\n";
        return 2;
    }

    double u1 = cam1.principalX();
    double v1 = cam1.principalY();
    double u2 = cam2.principalX();
    double v2 = cam2.principalY();

    if (argc == 7)
    {
        u1 = std::atof(argv[3]);
        v1 = std::atof(argv[4]);
        u2 = std::atof(argv[5]);
        v2 = std::atof(argv[6]);
    }

    auto result = Intersection::intersectPair(cam1, u1, v1, cam2, u2, v2);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "是否有效: " << (result.valid ? "是" : "否") << "\n";
    std::cout << "三维坐标 (X, Y, Z): "
              << result.point[0] << ", "
              << result.point[1] << ", "
              << result.point[2] << "\n";
    std::cout << "交汇角 (度): " << result.angle_deg << "\n";
    std::cout << "射线最短距离残差: " << result.ray_miss_distance << "\n";
    std::cout << "重投影残差 - 相机1 (像素): " << result.reproj_error_cam1 << "\n";
    std::cout << "重投影残差 - 相机2 (像素): " << result.reproj_error_cam2 << "\n";
    std::cout << "重投影残差 RMS (像素): " << result.reproj_error_rms << "\n";

    return 0;
}
