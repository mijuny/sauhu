# Sauhu: Roadmap to 10/10

After the second ThePrimeagen x DHH review (8.5/10), these are the concrete items that close the remaining gap.

## Tier 1: High impact, low effort (do this week)

### 1. Name those tuples

The `(PathBuf, usize, usize, ViewportId, u64)` appears in channels, pending lists, and function signatures. One struct fixes readability everywhere:

```rust
pub struct ImageLoadRequest {
    pub path: PathBuf,
    pub index: usize,
    pub total: usize,
    pub viewport_id: ViewportId,
    pub generation: u64,
}
```

Also kills the `#[allow(clippy::type_complexity)]` suppression.

### 2. Commit `interaction.rs`

It's sitting there untracked. Ship it.

### 3. Stop silencing channel errors

43 `let _ = tx.send(...)` across the codebase. Most are fine (shutdown races), but the ones in anonymization and retrieval workflows should log:

```rust
if let Err(e) = result_tx.send(result) {
    tracing::warn!("Channel closed: {e}");
}
```

Users currently get zero feedback when background operations fail silently.

### 4. Kill the blanket `#![allow(dead_code)]`

It's on almost every file. Remove them all, see what the compiler flags, and either delete the dead code or put `#[allow]` on the specific item with a comment why it's kept.

## Tier 2: High impact, medium effort (next sprint)

### 5. Split `database_window.rs` (2,116 lines)

It has four natural pieces already marked by state enums:

```
ui/database_window/
  mod.rs           -- DatabaseWindow struct, orchestration
  pacs_query.rs    -- QueryState, C-FIND/C-MOVE, results table
  anonymize.rs     -- AnonymizeState, progress, PHI stripping
  local_browser.rs -- local study listing, file management
```

### 6. Tests on core algorithms

47 tests total. The modules with zero tests that actually matter:

- `dicom/anonymize.rs` -- correctness is a compliance issue
- `dicom/mpr.rs` -- geometry bugs are silent and dangerous
- `coregistration/pipeline.rs` -- preprocessing and volume resampling logic
- `cache.rs` -- eviction behavior, memory accounting

Don't need UI tests or GPU tests. But the pure computational code, yes.

### 7. Fix `#[allow(clippy::too_many_arguments)]` suppressions

The compiler is telling you a function wants to be a method on a struct. `gpu/coregistration.rs:277` and `viewport_manager.rs:960` both take 7+ args that could be bundled.

## Tier 3: Polish (when it matters)

### 8. Consistent error types

Mix of `anyhow::Result`, `Result<T, String>`, and custom errors. Pick a story: `anyhow` for application code, `thiserror` for library boundaries. The IPC module returning `Result<T, String>` is the worst offender.

### 9. GPU runtime resilience

`gpu_available` is set once at startup. If the GPU runs out of VRAM mid-session (large study, many viewports), there's no graceful degradation. At minimum, catch texture upload failures and fall back to CPU for that viewport.

### 10. Document the unsafe

One `unsafe { libc::getuid() }` in `ipc.rs`. It's fine, but add a safety comment. Reviewers shouldn't have to think about whether the unsafe block is sound.

## What NOT to do

- Don't split the coregistration pyramid into a loop. Three sequential blocks with clear labels reads better than abstracted iteration for exactly three items.
- Don't add doc comments to private internals just for coverage numbers.
- Don't chase 100% test coverage. Test the math, test the data pipelines, skip the UI glue.
- Don't rewrite `app/mod.rs`. At 1,940 lines with 14 fields it's a large orchestrator but it's coherent. Splitting it further just scatters the control flow.

## Summary

The gap between 8.5 and 9.5 is items 1-7. The gap between 9.5 and 10 doesn't exist, it's just maintenance discipline over time.
