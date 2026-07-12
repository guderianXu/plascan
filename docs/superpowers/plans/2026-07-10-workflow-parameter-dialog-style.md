# Workflow Parameter Dialog Style Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可复用的重建参数对话框样式基础，并将“生成模型”窗口迁移为 A 方案的 Metashape 紧凑分组布局。

**Architecture:** 新增无业务状态的 `WorkflowParameterDialogStyle` 辅助模块，集中配置 Qt 对话框、表单、组合框和按钮盒。视觉规则通过 `workflowParameterDialog` 动态属性限定在参数对话框内部；`GenerateModelDialog` 只调用样式辅助并调整容器结构，现有参数与信号流程保持不变。

**Tech Stack:** C++17、Qt6 Widgets/Test、QSS、CMake、GTest、Ninja/MSVC

---

## 文件结构

- Create: `src/gui/dialogs/WorkflowParameterDialogStyle.h` — 通用参数对话框样式接口。
- Create: `src/gui/dialogs/WorkflowParameterDialogStyle.cpp` — 布局、组合框和按钮盒配置实现。
- Modify: `src/gui/dialogs/GenerateModelDialog.h` — 保存区域只读标签以同步启用状态。
- Modify: `src/gui/dialogs/GenerateModelDialog.cpp` — 应用通用样式并迁移为紧凑分组布局。
- Modify: `resources/styles/app.qss` — 增加动态属性作用域下的参数对话框视觉规则。
- Modify: `src/gui/cmake/GuiSources.cmake` — 将通用样式模块加入 GUI 构建。
- Create: `tests/test_workflow_parameter_dialog_style.cpp` — 独立 GTest 验证通用样式与试点行为。
- Modify: `tests/CMakeLists.txt` — 增加独立测试目标。

`GenerateModelDialog.cpp` 当前已有未提交的自动深度图候选逻辑。执行时只补丁式修改构造函数和区域控件状态，不覆盖 `setSourceCandidates()` 等现有改动。除非用户明确要求，不执行 commit 或 push。

### Task 1: 用运行时断言锁定紧凑对话框行为

**Files:**
- Create: `tests/test_workflow_parameter_dialog_style.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: 新增独立 GTest 目标**

在 `tests/CMakeLists.txt` 增加：

```cmake
add_executable(test_workflow_parameter_dialog_style
    test_workflow_parameter_dialog_style.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/GenerateModelDialog.cpp
)
set_target_properties(test_workflow_parameter_dialog_style PROPERTIES
    AUTOMOC ON
)
target_link_libraries(test_workflow_parameter_dialog_style PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test
    Qt6::Widgets
    GTest::gtest
)
target_include_directories(test_workflow_parameter_dialog_style PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs
)
gtest_discover_tests(test_workflow_parameter_dialog_style
    DISCOVERY_TIMEOUT 60
    DISCOVERY_MODE PRE_TEST
    PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen;QT_QPA_PLATFORM_PLUGIN_PATH=${PLASCAN_QT_PLATFORM_PLUGIN_PATH}")
```

- [ ] **Step 2: 编写当前实现会失败的布局测试**

`tests/test_workflow_parameter_dialog_style.cpp` 创建 `QApplication`，实例化 `GenerateModelDialog` 并验证：

```cpp
TEST(WorkflowParameterDialogStyleTest, GenerateModelUsesCompactScopedLayout)
{
    GenerateModelDialog dialog;
    EXPECT_TRUE(dialog.property("workflowParameterDialog").toBool());

    auto *generalGroup = dialog.findChild<QGroupBox *>(QStringLiteral("workflowGeneralGroup"));
    auto *regionGroup = dialog.findChild<QGroupBox *>(QStringLiteral("workflowRegionGroup"));
    auto *advancedGroup = dialog.findChild<QGroupBox *>(QStringLiteral("workflowAdvancedGroup"));
    ASSERT_NE(generalGroup, nullptr);
    ASSERT_NE(regionGroup, nullptr);
    ASSERT_NE(advancedGroup, nullptr);

    auto *generalForm = qobject_cast<QFormLayout *>(generalGroup->layout());
    ASSERT_NE(generalForm, nullptr);
    EXPECT_EQ(generalForm->fieldGrowthPolicy(), QFormLayout::AllNonFixedFieldsGrow);
    EXPECT_EQ(generalForm->horizontalSpacing(), 12);
    EXPECT_EQ(generalForm->verticalSpacing(), 6);

    auto *sourceItems = dialog.findChild<QComboBox *>(QStringLiteral("modelSourceItemCombo"));
    ASSERT_NE(sourceItems, nullptr);
    EXPECT_EQ(sourceItems->sizeAdjustPolicy(), QComboBox::AdjustToMinimumContentsLengthWithIcon);
    EXPECT_EQ(sourceItems->minimumContentsLength(), 24);

    auto *buttonBox = dialog.findChild<QDialogButtonBox *>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(buttonBox, nullptr);
    EXPECT_TRUE(buttonBox->centerButtons());
    EXPECT_EQ(buttonBox->button(QDialogButtonBox::Ok)->text(), QStringLiteral("确定"));
    EXPECT_EQ(buttonBox->button(QDialogButtonBox::Cancel)->text(), QStringLiteral("取消"));
}
```

- [ ] **Step 3: 编写高级折叠与区域状态测试**

```cpp
TEST(WorkflowParameterDialogStyleTest, AdvancedSectionAndRegionControlsKeepBehavior)
{
    GenerateModelDialog dialog;
    QJsonObject pointCloud{
        {QStringLiteral("source_data"), QStringLiteral("point_cloud")},
        {QStringLiteral("source_label"), QStringLiteral("点云")},
        {QStringLiteral("source_path"), QStringLiteral("E:/tmp/cloud.ply")},
        {QStringLiteral("display"), QStringLiteral("cloud.ply")},
        {QStringLiteral("supported"), true}
    };
    dialog.setSourceCandidates(QJsonArray{pointCloud});

    auto *toggle = dialog.findChild<QToolButton *>(QStringLiteral("workflowAdvancedToggle"));
    auto *advanced = dialog.findChild<QGroupBox *>(QStringLiteral("workflowAdvancedGroup"));
    auto *split = dialog.findChild<QCheckBox *>(QStringLiteral("splitRegionCheck"));
    auto *blockSize = dialog.findChild<QDoubleSpinBox *>(QStringLiteral("blockSizeSpin"));
    ASSERT_NE(toggle, nullptr);
    ASSERT_NE(advanced, nullptr);
    ASSERT_NE(split, nullptr);
    ASSERT_NE(blockSize, nullptr);

    EXPECT_FALSE(toggle->isChecked());
    EXPECT_TRUE(advanced->isHidden());
    EXPECT_FALSE(blockSize->isEnabled());

    split->setChecked(true);
    EXPECT_TRUE(blockSize->isEnabled());
    toggle->setChecked(true);
    EXPECT_FALSE(advanced->isHidden());
}
```

同一测试文件增加参数保持测试，避免布局重构改变业务输出：

```cpp
TEST(WorkflowParameterDialogStyleTest, LayoutMigrationPreservesModelSettings)
{
    GenerateModelDialog dialog;
    QJsonObject candidate{
        {QStringLiteral("source_data"), QStringLiteral("point_cloud")},
        {QStringLiteral("source_label"), QStringLiteral("点云")},
        {QStringLiteral("source_path"), QStringLiteral("E:/tmp/cloud.ply")},
        {QStringLiteral("display"), QStringLiteral("cloud.ply")},
        {QStringLiteral("supported"), true}
    };
    dialog.applySettings(QJsonObject{
        {QStringLiteral("source_data"), QStringLiteral("point_cloud")},
        {QStringLiteral("source_path"), QStringLiteral("E:/tmp/cloud.ply")},
        {QStringLiteral("quality"), QStringLiteral("low")},
        {QStringLiteral("targetFaces"), 60000}
    });
    dialog.setSourceCandidates(QJsonArray{candidate});

    QSignalSpy runSpy(&dialog, &GenerateModelDialog::runRequested);
    auto *buttonBox = dialog.findChild<QDialogButtonBox *>(QStringLiteral("workflowButtonBox"));
    ASSERT_NE(buttonBox, nullptr);
    buttonBox->button(QDialogButtonBox::Ok)->click();
    ASSERT_EQ(runSpy.count(), 1);

    const QJsonObject settings = runSpy.at(0).at(0).toJsonObject();
    EXPECT_EQ(settings.value(QStringLiteral("source_data")).toString(), QStringLiteral("point_cloud"));
    EXPECT_EQ(settings.value(QStringLiteral("source_path")).toString(), QStringLiteral("E:/tmp/cloud.ply"));
    EXPECT_EQ(settings.value(QStringLiteral("quality")).toString(), QStringLiteral("low"));
    EXPECT_EQ(settings.value(QStringLiteral("targetFaces")).toInt(), 60000);
}
```

- [ ] **Step 4: 编译并运行，确认 RED**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_workflow_parameter_dialog_style --parallel 8
build\windows-vcpkg-cuda-release\tests\test_workflow_parameter_dialog_style.exe
```

Expected: 测试以运行时断言失败，原因包括缺少作用域属性/对象名、按钮仍为 `OK/Cancel`、组合框仍使用默认尺寸策略。

### Task 2: 实现通用参数对话框样式模块

**Files:**
- Create: `src/gui/dialogs/WorkflowParameterDialogStyle.h`
- Create: `src/gui/dialogs/WorkflowParameterDialogStyle.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: 定义无状态辅助接口**

`WorkflowParameterDialogStyle.h`：

```cpp
#pragma once

class QComboBox;
class QDialog;
class QDialogButtonBox;
class QFormLayout;
class QVBoxLayout;

namespace xjw::gui::dialogs
{

void configureWorkflowParameterDialog(QDialog *dialog);
void configureWorkflowDialogLayout(QVBoxLayout *layout);
void configureWorkflowForm(QFormLayout *form);
void configureWorkflowComboBox(QComboBox *comboBox);
void configureWorkflowButtonBox(QDialogButtonBox *buttonBox);

} // namespace xjw::gui::dialogs
```

- [ ] **Step 2: 实现稳定尺寸和间距**

`WorkflowParameterDialogStyle.cpp` 使用以下固定规则：

```cpp
void configureWorkflowParameterDialog(QDialog *dialog)
{
    if (!dialog) return;
    dialog->setProperty("workflowParameterDialog", true);
    dialog->setMinimumWidth(520);
    dialog->setSizeGripEnabled(false);
}

void configureWorkflowDialogLayout(QVBoxLayout *layout)
{
    if (!layout) return;
    layout->setContentsMargins(14, 12, 14, 12);
    layout->setSpacing(9);
    layout->setSizeConstraint(QLayout::SetMinimumSize);
    layout->addStrut(492);
}

void configureWorkflowForm(QFormLayout *form)
{
    if (!form) return;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setRowWrapPolicy(QFormLayout::DontWrapRows);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setFormAlignment(Qt::AlignTop);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(6);
    form->setContentsMargins(12, 12, 12, 10);
}

void configureWorkflowComboBox(QComboBox *comboBox)
{
    if (!comboBox) return;
    comboBox->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    comboBox->setMinimumContentsLength(24);
    comboBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void configureWorkflowButtonBox(QDialogButtonBox *buttonBox)
{
    if (!buttonBox) return;
    buttonBox->setCenterButtons(true);
    if (QPushButton *ok = buttonBox->button(QDialogButtonBox::Ok)) ok->setText(QObject::tr("确定"));
    if (QPushButton *cancel = buttonBox->button(QDialogButtonBox::Cancel)) cancel->setText(QObject::tr("取消"));
}
```

- [ ] **Step 3: 接入 GUI 和测试构建**

将 `WorkflowParameterDialogStyle.cpp/.h` 加入 `src/gui/cmake/GuiSources.cmake`，并将 `.cpp` 加入 `test_workflow_parameter_dialog_style` 的源文件列表。

### Task 3: 将 GenerateModelDialog 迁移为 A 方案

**Files:**
- Modify: `src/gui/dialogs/GenerateModelDialog.h`
- Modify: `src/gui/dialogs/GenerateModelDialog.cpp`

- [ ] **Step 1: 保存区域只读标签并命名关键控件**

在头文件新增：

```cpp
QLabel *_coordinateLabel = nullptr;
QLabel *_originLabel = nullptr;
```

构造函数为关键控件设置以下对象名：

```cpp
workflowGeneralGroup
workflowRegionGroup
workflowAdvancedToggle
workflowAdvancedGroup
workflowStatusLabel
workflowButtonBox
modelSourceCombo
modelSourceItemCombo
splitRegionCheck
blockSizeSpin
```

- [ ] **Step 2: 应用通用布局规则**

构造函数开头调用 `configureWorkflowParameterDialog(this)`，主布局调用 `configureWorkflowDialogLayout()`，通过 492 px 内容宽度支撑加左右 14 px 边距维持 520 px 最小宽度，同时允许用户按需放大。三个表单全部调用 `configureWorkflowForm()`，五个基本组合框和两个高级组合框全部调用 `configureWorkflowComboBox()`。

“一般”和“区域”继续使用 `QGroupBox`；高级内容由无标题 `QWidget` 改为 `QGroupBox(tr("高级参数"))`，但 `_advancedContent` 成员类型保持 `QWidget *`，避免扩大接口变更。

- [ ] **Step 3: 将高级按钮和状态区整理为稳定层级**

`_advancedToggle` 使用 `Qt::ToolButtonTextBesideIcon`、水平扩展尺寸策略和现有左右/向下箭头。状态标签设置 `workflowStatusLabel` 对象名并保持自动换行。

按钮盒设置 `workflowButtonBox` 后调用 `configureWorkflowButtonBox()`。删除构造函数中的 `OK/Cancel` 手工文案。

`setAdvancedExpanded()` 删除固定最小高度 `620/430`，仅控制可见性、刷新布局并 `adjustSize()`，从而避免高级展开后形成大块空白。

- [ ] **Step 4: 同步区域只读信息的启用状态**

`updateBlockControlsAvailability()` 使用：

```cpp
const bool splitEnabled = blockCapable && _splitRegionCheck->isChecked();
_splitRegionCheck->setEnabled(blockCapable);
_coordinateLabel->setEnabled(splitEnabled);
_blockSizeSpin->setEnabled(splitEnabled);
_originLabel->setEnabled(splitEnabled);
_skipBoundaryBlocksCheck->setEnabled(splitEnabled);
```

该改动只调整显示状态，不改变提交的参数值。

### Task 4: 增加作用域 QSS 并完成验证

**Files:**
- Modify: `resources/styles/app.qss`
- Verify: all files above

- [ ] **Step 1: 添加局部 QSS**

在 `app.qss` 末尾增加全部以 `QDialog[workflowParameterDialog="true"]` 开头的规则：

```css
QDialog[workflowParameterDialog="true"] QLabel,
QDialog[workflowParameterDialog="true"] QCheckBox {
    background: transparent;
}

QDialog[workflowParameterDialog="true"] QGroupBox {
    background: #fafbfc;
    border: 1px solid #d3d9e1;
    border-radius: 2px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-weight: 600;
}

QDialog[workflowParameterDialog="true"] QGroupBox QLabel,
QDialog[workflowParameterDialog="true"] QGroupBox QCheckBox,
QDialog[workflowParameterDialog="true"] QGroupBox QComboBox,
QDialog[workflowParameterDialog="true"] QGroupBox QAbstractSpinBox {
    font-weight: 400;
}

QDialog[workflowParameterDialog="true"] QToolButton#workflowAdvancedToggle {
    background: transparent;
    border: none;
    border-top: 1px solid #d3d9e1;
    border-bottom: 1px solid #d3d9e1;
    border-radius: 0;
    padding: 4px 2px;
    text-align: left;
}

QDialog[workflowParameterDialog="true"] QLabel#workflowStatusLabel {
    color: #5b6674;
    padding: 2px 2px 6px 2px;
}
```

- [ ] **Step 2: 编译并运行独立测试，确认 GREEN**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_workflow_parameter_dialog_style --parallel 8
build\windows-vcpkg-cuda-release\tests\test_workflow_parameter_dialog_style.exe
ctest --test-dir build/windows-vcpkg-cuda-release -C Release --output-on-failure -R "WorkflowParameterDialogStyleTest"
```

Expected: 独立测试全部通过，0 failures。

- [ ] **Step 3: 构建 GUI 并检查补丁**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target plascan.exe --parallel 8
git diff --check -- src/gui/dialogs/WorkflowParameterDialogStyle.h src/gui/dialogs/WorkflowParameterDialogStyle.cpp src/gui/dialogs/GenerateModelDialog.h src/gui/dialogs/GenerateModelDialog.cpp resources/styles/app.qss src/gui/cmake/GuiSources.cmake tests/CMakeLists.txt tests/test_workflow_parameter_dialog_style.cpp
```

Expected: Release GUI 链接成功；无空白错误，Windows 行尾转换警告单独说明。

- [ ] **Step 4: 人工视觉检查**

打开“生成模型”对话框，分别检查高级折叠和展开状态：

- 标签和复选框没有独立灰底。
- 一般、区域、高级参数边框轻量且层级清楚。
- 长数据项不会撑大窗口。
- 展开高级参数时宽度不跳变，内容没有重叠。
- “确定/取消”居中，状态说明不抢占视觉层级。
- 100%、150%、200% DPI 下文字与控件完整显示。

不在本计划中执行 commit 或 push；等待用户明确指令。
