# Adaptive Visual Hull Implementation Plan

1. Add visual-hull options/result diagnostics and failing policy tests.
2. Wire `strictVolumetricMasks` from model settings into the depth-map mesh request.
3. Disable depth carving by default and add connectivity QA.
4. Retry an over-fragmented strict result without depth carving.
5. Record the actual algorithm/fallback in workflow JSON.
6. Build and run mesh tests, then execute the shared CLI on Dino in a new output directory.
