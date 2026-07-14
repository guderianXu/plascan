# Adaptive Visual Hull Design

## Goal

Produce a coherent closed model for studio turntable data such as Middlebury Dino, while retaining optional depth-based carving for reliable depth maps.

## Design

- Silhouette occupancy is the base visual hull and is always evaluated first.
- Depth free-space carving is opt-in through `strictVolumetricMasks`; low-confidence depth must not carve the hull by default.
- When strict carving is enabled, require enough valid multi-view depth evidence before carving.
- Measure mesh connectivity after reconstruction. If the largest connected component contains less than 85% of faces, or fragmentation exceeds the configured limit, retry without depth carving.
- If the silhouette-only retry still fails connectivity QA, reject the model with an actionable aerial-triangulation/camera-pose message. Do not silently emit the fragmented mesh or switch to a point-cloud mesh built from the same inconsistent geometry.
- Return the actual algorithm, whether depth carving was used, connectivity metrics, and fallback reason to the workflow payload.
- Do not silently label a visual-hull result as Poisson. The result metadata records the method that actually produced the mesh.

## Verification

- Unit-test that non-strict mode disables depth carving.
- Unit-test connectivity analysis and fallback policy.
- Run the shared model CLI on `E:/code/test/dino` into a separate output directory and compare component count/largest-component ratio with the current model.
