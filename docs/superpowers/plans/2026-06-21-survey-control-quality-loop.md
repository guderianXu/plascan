# Survey Control Quality Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first production-survey layer for PlaScan by making GCP/control points, checkpoints, and scale bars visible in the reconstruction quality report.

**Architecture:** Keep the first batch metadata-first and expose a lightweight GUI management layer. `ReconstructionQualityReport` reads `projectMeta["survey_control"]`, summarizes counts and residual RMSE values, writes them into the existing JSON/CSV report path, while `SurveyControlDialog` lets users import and inspect the same metadata before BA editing lands.

**Tech Stack:** C++17, Qt6 JSON classes, existing `src/core/qc` library, GTest, project metadata JSON.

---

### File Structure

- Create: `src/core/qc/SurveyControlReport.h`
  - Expose `buildSurveyControlSummary(const QJsonObject &projectMeta)`.
- Create: `src/core/qc/SurveyControlReport.cpp`
  - Parse `survey_control.control_points`, `survey_control.check_points`, and `survey_control.scale_bars`.
  - Summarize enabled item counts, residual RMSE, max residual, and quality status.
- Modify: `src/core/qc/ReconstructionQualityReport.cpp`
  - Embed the survey control summary in the existing report and expose CSV-friendly scalar keys.
  - Preserve existing report keys.
- Modify: `src/core/qc/tests/test_reconstruction_quality_report.cpp`
  - Add regression tests for survey control summary and CSV scalar output.
- Create: `src/gui/dialogs/SurveyControlDialog.h/cpp`
  - Read project `survey_control` metadata into three tables: control points, check points, and scale bars.
  - Emit an import request handled by `ProjectManager`, keeping file dialogs and project writes outside the view.
- Modify: `src/gui/menu/MainMenu.h/cpp`, `src/gui/main_window/MainWindow.cpp`, `src/gui/project/manager/ProjectManager.h/cpp`
  - Add the Tools → Survey Control entry and connect it to project metadata import/view logic.
- GUI editing, interactive GCP observation picking, and BA weighting remain follow-up work after the report contract is stable.

### Task 1: Add Survey Control Report Test

**Files:**
- Modify: `src/core/qc/tests/test_reconstruction_quality_report.cpp`

- [ ] **Step 1: Write the failing test**

Add a test that builds metadata with two control points, one checkpoint, and one scale bar:

```cpp
TEST(ReconstructionQualityReport, SummarizesSurveyControlResiduals)
{
    QJsonObject control0;
    control0[QStringLiteral("id")] = QStringLiteral("GCP001");
    control0[QStringLiteral("enabled")] = true;
    control0[QStringLiteral("residual")] = QJsonObject{
        {QStringLiteral("horizontal_m"), 0.03},
        {QStringLiteral("vertical_m"), 0.04}
    };

    QJsonObject control1;
    control1[QStringLiteral("id")] = QStringLiteral("GCP002");
    control1[QStringLiteral("enabled")] = true;
    control1[QStringLiteral("residual")] = QJsonObject{
        {QStringLiteral("total_m"), 0.05}
    };

    QJsonObject check0;
    check0[QStringLiteral("id")] = QStringLiteral("CHK001");
    check0[QStringLiteral("enabled")] = true;
    check0[QStringLiteral("residual")] = QJsonObject{
        {QStringLiteral("total_m"), 0.12}
    };

    QJsonObject scaleBar0;
    scaleBar0[QStringLiteral("id")] = QStringLiteral("SB001");
    scaleBar0[QStringLiteral("enabled")] = true;
    scaleBar0[QStringLiteral("residual_m")] = -0.02;

    QJsonObject thresholds;
    thresholds[QStringLiteral("checkpoint_rmse_warn_m")] = 0.10;
    thresholds[QStringLiteral("scale_bar_rmse_warn_m")] = 0.05;

    QJsonObject survey;
    survey[QStringLiteral("control_points")] = QJsonArray{control0, control1};
    survey[QStringLiteral("check_points")] = QJsonArray{check0};
    survey[QStringLiteral("scale_bars")] = QJsonArray{scaleBar0};
    survey[QStringLiteral("quality_thresholds")] = thresholds;

    QJsonObject meta;
    meta[QStringLiteral("survey_control")] = survey;

    const QJsonObject report = ReconstructionQualityReport::buildFromProjectMeta(meta);
    const QJsonObject surveyReport = report.value(QStringLiteral("survey_control")).toObject();

    EXPECT_EQ(surveyReport.value(QStringLiteral("control_point_count")).toInt(), 2);
    EXPECT_EQ(surveyReport.value(QStringLiteral("check_point_count")).toInt(), 1);
    EXPECT_EQ(surveyReport.value(QStringLiteral("scale_bar_count")).toInt(), 1);
    EXPECT_NEAR(surveyReport.value(QStringLiteral("control_point_rmse_m")).toDouble(), 0.05, 1e-9);
    EXPECT_NEAR(surveyReport.value(QStringLiteral("check_point_rmse_m")).toDouble(), 0.12, 1e-9);
    EXPECT_NEAR(surveyReport.value(QStringLiteral("scale_bar_rmse_m")).toDouble(), 0.02, 1e-9);
    EXPECT_EQ(surveyReport.value(QStringLiteral("status")).toString(), QStringLiteral("warn"));

    EXPECT_EQ(report.value(QStringLiteral("control_point_count")).toInt(), 2);
    EXPECT_EQ(report.value(QStringLiteral("check_point_count")).toInt(), 1);
    EXPECT_EQ(report.value(QStringLiteral("scale_bar_count")).toInt(), 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ReconstructionQualityReport\\.SummarizesSurveyControlResiduals" --output-on-failure
```

Expected: FAIL because `survey_control` is not summarized yet.

### Task 2: Implement Survey Control Summary

**Files:**
- Create: `src/core/qc/SurveyControlReport.h`
- Create: `src/core/qc/SurveyControlReport.cpp`
- Modify: `src/core/qc/ReconstructionQualityReport.cpp`
- Modify: `src/core/qc/CMakeLists.txt`

- [ ] **Step 1: Add residual helpers**

Add helpers that:

- treat disabled records as ignored;
- read residual magnitude from `residual.total_m`;
- fall back to `sqrt(horizontal_m^2 + vertical_m^2)`;
- fall back to absolute `residual_m` for scale bars;
- calculate count, enabled count, RMSE, max residual, and status.

- [ ] **Step 2: Add report keys**

`buildFromProjectMeta()` must add:

```json
"survey_control": {
  "control_point_count": 2,
  "enabled_control_point_count": 2,
  "check_point_count": 1,
  "enabled_check_point_count": 1,
  "scale_bar_count": 1,
  "enabled_scale_bar_count": 1,
  "control_point_rmse_m": 0.05,
  "check_point_rmse_m": 0.12,
  "scale_bar_rmse_m": 0.02,
  "status": "warn"
}
```

It must also expose scalar CSV-friendly top-level keys:

```json
"control_point_count": 2,
"check_point_count": 1,
"scale_bar_count": 1,
"control_point_rmse_m": 0.05,
"check_point_rmse_m": 0.12,
"scale_bar_rmse_m": 0.02
```

- [ ] **Step 3: Run targeted test**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ReconstructionQualityReport" --output-on-failure
```

Expected: PASS.

### Task 3: Verify Build Slice

**Files:**
- No extra source edits expected.

- [ ] **Step 1: Build the QC test target**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_reconstruction_quality_report -Jobs 8
```

Expected: build succeeds.

- [ ] **Step 2: Run related tests**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ReconstructionQualityReport|QualityReport" --output-on-failure
```

Expected: PASS.

### Follow-Up Batches

- GCP/checkpoint CSV import and metadata persistence.
- GUI table for control points, checkpoints, and scale bars.
- Interactive observation picking and marker residual table.
- BA observation weights for control points and holdout checkpoints.
- Scale bar constraints inside `BundleAdjust`.
- CRS/geoid metadata and coordinate consistency checks.
- DOM seamline/color/ghost filtering after control/report contract is stable.

### Progress Notes

- Core quality report summary implemented in `SurveyControlReport`.
- Core CSV parsing implemented in `SurveyControlImport`.
- Project metadata persistence implemented in `ProjectSurveyControl`.
- GUI Tools menu entry implemented as `actionSurveyControl`.
- `SurveyControlDialog` implemented with CSV import trigger plus read-only control point, check point, and scale bar tables.
