# 2026-06-12 Depth Map Tooltip Design

## Goal

Make the depth-map estimation dialog explain cost functions and important parameters when the user hovers over the related controls.

## Scope

- Add specific explanations for AD, SD, NCC, Census, and Ternary Census cost functions.
- Add practical hover text for resolution scale, iterations, patch size, minimum views, depth range, confidence, tile size, and threads.
- Do not change depth-map estimation runtime behavior or persisted setting keys.

## Implementation

- Keep layout in `DepthMapEstimateDialog.ui`.
- Add helper text in the dialog initialization only where dynamic combo-item tooltips are needed.
- Add tests that check the UI/source contains the expected cost-function and parameter explanations.
