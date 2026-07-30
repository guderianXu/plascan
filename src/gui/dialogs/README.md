# GUI Dialog 目录

`src/gui/dialogs` 只保存参数收集、结果查看和用户确认类对话框，并按业务域分类：

- `application/`：应用级信息与报告。
- `camera/`：相机、前方交汇和测量控制。
- `image/`：单影像与蒙版处理。
- `reconstruction/`：空三、模型、纹理、DEM 和正射工作流程。
- `tie_points/`：连接点创建、清理、查看和重叠分析。
- `shared/`：多个对话框复用的样式与布局辅助。

新增对话框时应放入对应业务目录，并在 include 中写明分类路径，例如：

```cpp
#include "tie_points/MatchViewerDialog.h"
```

源文件、头文件和 `.ui` 统一登记在
`src/gui/cmake/GuiDialogSources.cmake` 的对应业务分组中，不要同时加入
`GuiSources.cmake`。这样可以避免同一对话框被重复注册，也便于按业务域审查。

对话框只负责展示状态、校验输入和收集参数。长任务由 controller、manager、service
或 task 执行；文件 IO、算法、项目持久化和渲染数学不要放入 Dialog 类。非对话框的
视图辅助代码应放在 `src/gui/views`，跨模块能力应下沉到 `src/common` 或 `src/core`。
