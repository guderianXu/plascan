# 2026-06-12 Vocabulary Overlap Pairs Implementation

## Goal

Add a sparse-reconstruction menu step named "获取重叠对..." between feature extraction and feature matching. The step uses extracted feature descriptors to build a project vocabulary, scores image overlap candidates, optionally runs a geometry check, writes overlap pair outputs, and can pass generated pairs into the existing feature-matching dialog settings.

## Constraints

- Do not change existing reconstruction logic or feature matching behavior unless the user explicitly runs the new step.
- Keep UI layout in Qt Designer `.ui` form.
- Preserve existing user changes in the dirty worktree.
- Do not commit unless explicitly requested.

## Tasks

1. Add red tests.
   - Core test for expected overlap pair retrieval from synthetic descriptors.
   - Core test for descriptor dimension mismatch.
   - GUI source tests for menu ordering, required dialog controls, setting key, and generated pair handoff.

2. Implement core overlap retrieval.
   - Add `VocabularyOverlapRetriever` under `src/core/overlap`.
   - Convert descriptor sets into a visual vocabulary, build TF/TF-IDF histograms, score top-K candidate image pairs, and optionally validate with RANSAC.
   - Keep this layer independent from Qt and project files.

3. Implement Qt dialog.
   - Add `VocabularyOverlapDialog.ui/.h/.cpp`.
   - Load project images and feature files from `assets/ip`.
   - Expose vocabulary, candidate, geometry, and output parameters.
   - Write JSON and LIS outputs.
   - Save generated pair tokens compatible with `FeatureMatchingDialog`.

4. Wire workflow.
   - Add `actionVocabularyOverlap` to `MainWindow.ui`.
   - Insert menu action between "特征点提取" and "创建连接点".
   - Add `MenuWorkflowController::openVocabularyOverlapDialog()`.
   - Persist settings via `DialogSettingKeys::VocabularyOverlap`.
   - Push `generated_pairs` into existing feature matching settings when requested.

5. Verify.
   - Build targeted tests.
   - Run core overlap tests and GUI source tests.
   - Run a focused CTest filter for overlap/gui if available.
