# Depth Batch Prefetch Design

## Goal

Reduce the visible pause before depth-map fusion without moving disk I/O to CUDA or allowing parallel loads to exhaust RAM.

## Design

- Load and post-process missing depth frames on a bounded CPU worker pool.
- Choose 2-4 workers from the requested limit, available physical memory, and the estimated per-frame working set.
- Keep the existing LRU cache as the authoritative owner of decoded frames. Cache lookup and insertion are synchronized; expensive decoding happens outside the cache lock.
- While fusion consumes the current window, prefetch the next window's missing frames. A frame may have only one in-flight load.
- Report `loaded X/Y` and aggregate read, post-process, resize, and total timings.
- CUDA remains responsible for fusion. Depth storage reads, CPU filtering, and resize stay on the CPU to avoid unnecessary host-device transfers.
- Cancellation stops scheduling new work and lets already-running CPU jobs finish safely.

## Memory Policy

Estimate one loading worker from the raw depth, confidence, post-process scratch, and resized output. Reserve at least 25% of currently available RAM and clamp concurrency to 2-4 workers. If memory is tight, allow one worker rather than failing.

## Verification

- Unit-test memory-adaptive worker selection.
- Unit-test timing fields produced by frame construction.
- Run MVS/GUI tests and a real Dino/9-image fusion, checking progressive loading logs and bounded memory.
