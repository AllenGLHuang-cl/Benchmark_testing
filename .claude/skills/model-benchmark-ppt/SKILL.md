---
name: model-benchmark-ppt
description: Use when the user wants a PowerPoint (.pptx) deck comparing generative-model outputs — currently images, either an i2i/editing dataset (feature dirs containing src/ + meta_data/ + results/<model>/) or a flat t2i dataset (one dir per model, matched by filename key). e.g. "生成比較 PPT"、"比較這幾個 model"、"做一份 model comparison 簡報" / "benchmark 簡報". This is the only comparison-deck skill — there is no HTML equivalent. Not yet wired for video (t2v/i2v); see the "Extending to video" note before building that from scratch.
---

# Model Benchmark PPT

## Overview
One config + `make_comparison_ppt.py` → a native, editable .pptx: cover, optional quantitative
perf table, then one slide per comparison group showing every selected model (+ Source,
for the features layout) in a grid, prompt text in the title area, full prompt & file
paths in speaker notes. Supports **two dataset layouts** (`dataset_layout` in config) and
can **auto-fill the perf table from `performance.jsonl`** instead of hand-typing numbers.
All resolution of the dataset's messy filename conventions lives in the script — never
hand-match files.

Media support today is **images only** (t2i/i2i); see "Extending to video" at the bottom
before adding video comparisons.

## Two dataset layouts
- **`"features"` (default, i2i/editing)** — `dataset_root` has one dir per feature, each
  with `src/`, `meta_data/`, `results/<model_dir>/`. Groups by (source image, prompt
  index); shows Source + models — the Source sits in its own left column, vertically
  centered, with the model outputs in an aligned grid to its right (user preference,
  2026-07-02). Missing model outputs render as a grey "missing" slide cell rather than
  dropping the group.
- **`"flat"` (t2i, no source image)** — `models[].dir` is a flat directory; a `key_regex`
  extracts the comparison key from each filename (e.g. `tti_{idx}_seed{seed}.jpg`). Only
  keys present in every model are rendered (intersect, don't guess).

## Workflow
1. **Scan** the dataset root: for features layout, features = dirs containing non-empty
   `meta_data/` and `results/`; list the union of `results/*` model dirs **with
   per-feature coverage counts** (some models cover 9/18 srcs — surface this before the
   user picks). For flat layout, just list the candidate model dirs.
2. **Ask the user** (AskUserQuestion, multiSelect) which features/dataset and which
   models to include — skip the question if the user already named them. Estimate deck
   size (~0.2–0.4 MB per slide at default `thumb_px`) and warn if huge.
3. **Write a config** (copy `example-config.json` for features, `example-config-flat.json`
   for flat; keep it next to the output for reproducibility) and run:
   ```bash
   python /workspace/Benchmark_testing/.claude/skills/model-benchmark-ppt/make_comparison_ppt.py config.json
   ```
   Decks + their configs go in **`/workspace/Benchmark_testing/comparison_ppt/`** (user
   preference, 2026-07-02) — the dir is gitignored so multi-MB `.pptx` files never reach
   the GitHub remote; don't un-ignore it.
4. **Report** the printed summary: layout, group count, missing-cell stats (every model
   listed, including 0-missing ones), file size.

## Config essentials (see example configs)
| field | layout | meaning |
|---|---|---|
| `dataset_layout` | both | `"features"` (default) or `"flat"` |
| `models[]` | both | features: `{label, dir}` (`dir` = `results/` subdir name); flat: `{label, dir}` (`dir` = full path to the flat model dir) |
| `thumb_px` | both | long-side downscale before embedding (default 768) |
| `dataset_root`, `features` | features | dataset root; list of feature dir names or `"all"` |
| `seed_prefer` | features | e.g. `["seed2025","seed2026"]` — which seed wins when a model has several |
| `picks` | features | optional `{feature: [src_stems]}` to curate; omitted feature → all srcs |
| `key_regex` / `prefer_regex` | flat | group(1) = comparison key; optional preferred-seed regex |
| `picks` / `auto_pick` | flat | explicit key list, or int N to evenly sample |
| `captions` | flat | optional `{file, json_path, key_is_index}` to print the prompt under each group |
| `perf` | both | `{note, columns[], auto_from, rows}` → one quantitative table slide right after the cover (rows = models in deck order) |
| `perf.auto_from` | both | feature name (features layout) whose `results/<model_dir>/performance.jsonl` to aggregate; for flat layout just set it truthy — the log is read straight from `models[].dir` |
| `perf.rows` | both | `{model label or dir: {column: value}}`, hand-typed values — **override** the auto-computed ones (so you can add un-measurable columns like "Model size" or fix a bad reading). **Unknown/un-computed cells stay blank — never invent numbers.** |

## Non-obvious gotchas (why this skill exists)
- **5 filename conventions** coexist per model dir: `{src}_prompt{p}_seed{S}` (also matches
  `{src}_prompt{p}_out{i}_seed{S}` — batch runners that record multiple outputs per prompt add
  an `_out{i}_` segment before the seed), `{src}_prompt{p}` (+`_config.json` sidecars), `{src}_p{p}`,
  `{src}_{p}_{n}` (gemini), and gpt_image2's **per-src subdirectory** `{src}/p{p}_{n}.jpg`. The
  script's resolver tries all five — do not write ad-hoc glob/prefix matching (baseline agents that
  sorted `{stem}_*` and took the first file silently pinned prompt0/sample1 everywhere).
- **Enumerate stems from `src/`, not from `meta_data/*.json`.** Most features in this dataset use
  one *shared* `meta_data/prompt.json` for every image rather than a `{stem}.json` each — iterating
  the json files directly treats `"prompt"` as a bogus image stem and silently drops the whole
  feature (this was a real bug here until fixed). Prompts come from `meta_data/{stem}.json`, falling
  back to `meta_data/prompt.json`. `prompt` is a LIST; srcs with 2+ prompts produce 2+ comparison
  groups — reading only `prompt[0]` silently drops groups.
- **features layout: never intersect keys across models.** Coverage is uneven; intersection
  can kill most groups. Missing outputs render as grey "missing" placeholder cells and are
  counted in the summary. **flat layout does the opposite** — it intersects (only keys
  present in every model render) — same rule the old HTML-only tool used, kept for parity.
- **Downscale before embedding.** Naive full-res embedding produced an 80 MB deck from
  9 slides; the script's `thumb_px` keeps ~30 slides ≈ 12 MB.
- Ignore `_logs/` and empty feature dirs; `features: "all"` already filters them.
- **Auto perf only understands specific `performance.jsonl` field names** (`steps`,
  `elapsed_sec`, `vram_peak_gib` — one single VRAM figure, chosen to approximate
  `nvidia-smi`/NVML rather than ComfyUI's own under-counted free-memory report —
  `weight_type`, `device`, `output_width/height`; see `_PERF_AUTO_COLUMNS` in
  `make_comparison_ppt.py`, written by `websocket_edit.py` (this repo) / `websocket_tti.py` via
  the shared `perf_utils.py`). This mapping is a **hand-kept copy** — the two
  scripts import the real aggregation from `perf_utils.aggregate_records()`, but this
  skill stays self-contained rather than importing from the repo root, so if the
  record schema changes again, update `_PERF_AUTO_COLUMNS` here too. A `performance.jsonl`
  recorded before a schema change (e.g. old per-metric fields like `inference_time_s`/
  `memory_used_gib`, or the pre-GiB `vram_peak_mb`) silently produces blank cells for the
  renamed columns, not wrong ones — re-run the batch to refresh it.
- **Batch scripts warm up (default 2x, discarded) before timing** — the first request
  after a model loads pays one-time cache/allocation cost that would otherwise skew
  `avg_elapsed_sec` on small sample counts. They also write a `performance_summary.json`
  next to `performance.jsonl` (aggregated over the *whole* file, including earlier
  resumed runs) — use that for a quick look without opening the deck.
- **`perf.auto_from` path resolution differs by layout**: features → `dataset_root/<auto_from>/results/<model_dir>/performance.jsonl` (so `auto_from` is the *benchmark* feature name, usually different from the qualitative-comparison features in `features[]`); flat → `<model_dir>/performance.jsonl` directly, no `auto_from` value needed beyond truthy.

## Common mistakes
- Comparing a fixed seed across models (features layout) → missing cells; use `seed_prefer` (per-model best effort), not filename equality.
- Editing `models[].dir` with a label-like name → column all-missing; it must be the exact `results/` subdir name (script warns per feature) or the full flat-mode path.
- Huge all-features × all-models decks: use `picks` or lower `thumb_px` instead of splitting the config by hand.
- Expecting `perf.rows` to override with `null`/missing keys — it only overrides keys it explicitly sets; omit a column entirely to keep the auto-computed value.

## Extending to video (not implemented — read this first)
This skill was generalized from an image-only tool (`editing-models-ppt`) with the intent
that video (t2v/i2v) comparisons land here too, rather than in a separate skill. Nothing
video-specific exists yet — don't add speculative config fields for it ahead of a real
need. When there's an actual video dataset to compare:
- **Check the dataset has a stable `results/<model>/` convention first.** As of this
  writing, `data/allen/dataset/wan_animate/I2V_test` does not — it has a
  `Kling_results/<category>/` shape instead of one dir per model. Either the dataset needs
  reorganizing to match the `features`/`flat` layouts, or a third `dataset_layout` needs
  designing around however video comparisons actually get organized — don't force-fit it.
- **`load_thumb()`** is the one function that's inherently image-specific (PIL-based). A
  video variant would need to produce a poster-frame thumbnail (e.g. first frame via
  `ffmpeg`/`imageio`, or a `pptx`-native embed via `shapes.add_movie()` — python-pptx
  1.0.2 here supports it) with the same `(buf, w, h) | None` contract so `place_cell()`
  and `render_group_slide()` don't need to change.
- **The resolver (`resolve_output`/`index_model`) and grouping logic are already
  media-agnostic** — they just match filenames/extensions, so video output naming
  conventions would slot in the same way image ones do (extend `ext` patterns to
  `.mp4`/`.webm` etc.).
- **File size**: video embeds are much larger than downscaled JPEG thumbnails — the
  `thumb_px` sizing strategy won't translate directly; expect to budget deck size
  differently (e.g. short clips, lower bitrate, or poster-frame-only with a link).
