# Depth Batch Prefetch Implementation Plan

1. Add frame timing and memory-budget helpers with failing unit tests.
2. Make the LRU cache thread-safe and deduplicate in-flight frame loads.
3. Add a bounded CPU prefetch scheduler and next-window look-ahead.
4. Replace batch status text with `loaded X/Y` and emit stage timing logs.
5. Build and run targeted MVS/GUI tests plus a small real-data fusion.
